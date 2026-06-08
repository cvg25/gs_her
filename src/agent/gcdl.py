from src.agent.networks.gc_value import GC_DoubleValue
from src.agent.networks.gc_actor import GC_Actor
from src.utils import compute_num_trainable_params, compute_gradient_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy

class GCDL(nn.Module):
    """
    Goal-Conditioned Distance Learning (GCDL) Agent.
    GCDL is a goal-conditioned actor-critic agent designed to learn
    an interpretable distance-to-goal function from offline trajectories,
    then derive a policy using Advantage-Weighted Regression (AWR).

    Value convention
    ----------------
    The value network predicts a normalized negative distance-to-goal:

    V(s,g) in [-1, 0]

    where:
        V(s, g) = 0 means the goal has been reached,
        V(s, g) = -1 means the goal is as far as the maximum task horizon,
        d(s, g) = -V(s,g) * max_task_length

    Thus, the value can be interpreted directly as a remaining-step estimate
    after rescaling. This makes GCDL useful for diagnostic plots that compare
    full-state goal queries against projected/ masked goal queries.

    Goal-Set interface (GoSHER)
    ---------------------------
    The agent implements the GoSHER goal representation:
    z(g,m) = [m * g + (1-m) * e_mask, m]

    where:
        g is the reference goal state,
        m is a binary goal mask indicating which state dimensions define success,
        e_mask is a fixed/ learned/ sampled placeholder token for inactive goal coordinates.

    The concatenated mask is part of the goal specification. It tells the value and actor 
    networks which coordinates should be treated as task-relevant. 

    Value learning
    --------------
    The value network is a double value estimator with an EMA target network.
    the output layer is bounded with -sigmoid so that predictions remain in [-1, 0]. 
    Training uses a one-step bootstrapped temporal-distance target:

        if s_t is at goal:
            target = 0
        elif s_{t+1} is at goal:
            target = step_cost
        else: 
            target = step_cost + V_target(s_{t+1}, g)
    
    where:
        step_cost = -1 / max_task_length

    The target is clamped to [-1, 0]. The minimum of the two target value heads is used
    for pessimistic bootstrapping.

    Actor learning
    --------------
    The policy is trained with Advantage-Weighted Regression. The advantage is computed
    in distance space:

        d_t0 = -V(s_t, g) * mask_task_length
        d_t1 = -V(s_{t+1}, g) * mask_task_length
        adv = d_t0 - d_t1
    
    Therefore, adv > 0 means that the transition moved closer to the queried goal set. 
    The action log-likelihood is weighted by:
        exp(alpha * adv)

    so transitions that make more goal-directed progress receive a larger imitation weight.

    Intended use in the paper
    -------------------------
    GCDL is not the main algorithmic contribution. It is a diagnositce goal-conditioned 
    distance learner used to produce interpretable distance-to-goal curves. The main paper 
    contribution is GoSHER, the goal-set relabeling wrapper. GCDL is useful because its value
    output can be directly plotted as estimated steps-to-goal under different queries, e.g., 
    full-state mask versus cube-position mask.
    """

    def __init__(self, params):
        super().__init__()
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.proj_goal_idxs = params["proj_goal_idxs"]
        self.max_task_length = params['max_task_length']

        # Data
        self.value_p_goal_curr = 0.2
        self.value_p_goal_traj = 0.5
        self.value_p_goal_rand = 0.3
        self.actor_p_goal_curr = 0.0
        self.actor_p_goal_traj = 1.0
        self.actor_p_goal_rand = 0.0

        # HER / GS-HER
        task_query = torch.zeros((self.state_dim,), dtype=torch.float32)
        task_query[self.proj_goal_idxs] = 1.0
        self.register_buffer("task_query", task_query[None, :])

        self.her_type = params["her_type"]        
        if self.her_type in ["gs_her_learned", "gs_her_zeros", "gs_her_random"]:
            self.gs_her_active = True
            self.goal_dim = int(self.state_dim * 3)  # state + goal + query.

            if self.her_type == "gs_her_learned":
                self.goal_emb = nn.Parameter(torch.zeros(1, self.state_dim), requires_grad=True)
                nn.init.trunc_normal_(self.goal_emb, std=1e-6)
                self.state_emb = nn.Parameter(torch.zeros(1, self.state_dim), requires_grad=True)
                nn.init.trunc_normal_(self.state_emb, std=1e-6)

            elif self.her_type == "gs_her_zeros":
                self.goal_emb = nn.Parameter(torch.zeros(1, self.state_dim), requires_grad=False)
                self.state_emb = nn.Parameter(torch.zeros(1, self.state_dim), requires_grad=False)
            
            elif self.her_type == "gs_her_random":
                self.goal_emb = None  # Sample random at training / inference.
                self.state_emb = None 

        elif self.her_type == "her":
            self.gs_her_active = False
            self.goal_dim = 2 * self.state_dim # full_state, full_goal

        elif self.her_type == "her_task":
            self.gs_her_active = False
            self.goal_dim = 2 * len(self.proj_goal_idxs) # task_state, task_goal

        else:
            raise ValueError(f'Invalid her_type param: {params["her_type"]}')
            
        # Value.
        self.step_cost = -1.0 / self.max_task_length
        self.failure_value = -1.0
        
        self.online_value_net = GC_DoubleValue(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            last_layer=lambda x: -1.0 * F.sigmoid(x)
        )

        self.target_value_net = copy.deepcopy(self.online_value_net)
        self.target_value_net.requires_grad_(False)
        self.target_value_net.eval()

        # Actor.
        self.alpha = params['agent']['alpha'] # AWR temperature.

        self.actor_net = GC_Actor(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
        )

        # inference
        self.reset()

    def reset(self):
        self.distance_to_goal = []
    
    def _build_goal_input(self, state, goal_state, goal_query=None):
        B = goal_state.shape[0]
        if self.her_type in ["gs_her_learned", "gs_her_zeros", "gs_her_random"]:
            if goal_query is None:
                raise ValueError("goal_query must be provided for GS-HER variants.")
            goal_query = goal_query.to(device=goal_state.device, dtype=goal_state.dtype)
            if goal_query.ndim == 1:
                goal_query = goal_query[None, :]
            if goal_query.shape[0] == 1 and B > 1:
                goal_query = goal_query.expand(B, -1)
            if self.her_type == "gs_her_random":
                goal_emb = torch.randn_like(goal_state)
                state_emb = torch.randn_like(state)
            else:
                goal_emb = self.goal_emb.to(device=goal_state.device, dtype=goal_state.dtype)
                state_emb = self.state_emb.to(device=goal_state.device, dtype=goal_state.dtype)

            goal_state = goal_query * goal_state + (1.0 - goal_query) * goal_emb
            state = goal_query * state + (1.0 - goal_query) * state_emb
            goal = torch.cat((state, goal_state, goal_query), dim=-1)
        elif self.her_type == "her":
            goal = torch.cat((state, goal_state), dim=-1)
        elif self.her_type == "her_task":
            goal = torch.cat((state[:, self.proj_goal_idxs], goal_state[:, self.proj_goal_idxs]), dim=-1)
        else:
            raise RuntimeError(f"Invalid her_type: {self.her_type}")

        return goal

    @torch.no_grad()
    def act(self, state, goal, compute_distance=False):
        goal = self._build_goal_input(state=state, goal_state=goal, goal_query=self.task_query)
        
        # Compute distance to goal.
        if compute_distance:
            v1, v2 = self.target_value_net(state=state, goal=goal)
            v = (v1 + v2) / 2
            steps_to_goal = -1.0 * v * self.max_task_length
            self.distance_to_goal.append(steps_to_goal.item())

        # Act
        actions = self.actor_net.act(state=state, goal=goal)
        return actions

    def get_trainable_params(self):
        params = {
            'num_params_actor_net': compute_num_trainable_params(model=self.actor_net),
            'num_params_value_net': compute_num_trainable_params(model=self.online_value_net)
        }
        return params

    def clip_grad_norm(self, max_norm, info):
        info['gn_value_net'] = compute_gradient_norm(self.online_value_net.parameters())
        info['gn_actor_net'] = compute_gradient_norm(self.actor_net.parameters())

        params = list(self.online_value_net.parameters()) + list(self.actor_net.parameters())

        if self.her_type == "gs_her_learned":
            params += [self.goal_emb, self.state_emb]

        clip_grad_norm_(params, max_norm)

        return info

    def ema_update(self, ema_value=0.995):
        with torch.no_grad():
            for p, p_targ in zip(self.online_value_net.parameters(), self.target_value_net.parameters()):
                p_targ.data.mul_(ema_value).add_((1-ema_value) * p.data)

    def forward_train(self, data_dict):
        info = dict()
        value_loss, info = self.forward_value_train(
            data_dict=data_dict['value'],
            info_dict=info)
        
        actor_loss, info = self.forward_actor_train(
            data_dict=data_dict['actor'],
            info_dict=info)

        loss = value_loss + actor_loss
        info['total_loss'] = loss.detach().item()

        return loss, info

    def forward_value_train(self, data_dict, info_dict):
        s_t0 = data_dict['s_t0']
        s_t1 = data_dict['s_t1']
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
    
        curr_mask = data_dict['curr_mask']
        traj_mask = data_dict['traj_mask']
        rand_mask = data_dict['rand_mask']
        k_traj = data_dict['k_traj']
        
        goal_st0 = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        with torch.no_grad():
            # Target value at next state
            goal_st1 = self._build_goal_input(state=s_t1, goal_state=goal_state, goal_query=goal_query)
            v1_t1, v2_t1 = self.target_value_net(state=s_t1, goal=goal_st1)
            v_t1 = torch.minimum(v1_t1, v2_t1) # pessimistic
            v_t1 = v_t1.clamp(min=self.failure_value, max=0.0)

            # At goal checks.
            v1_t0, v2_t0 = self.target_value_net(state=s_t0, goal=goal_st0)
            v_t0 = torch.minimum(v1_t0, v2_t0) # pessimistic.
            v_t0 = v_t0.clamp(min=self.failure_value, max=0.0)

            s0_at_goal_boot = (v_t0 > (self.step_cost / 2))
            s1_at_goal_boot = (v_t1 > (self.step_cost / 2))

            s0_at_goal = curr_mask | s0_at_goal_boot
            s1_at_goal = (traj_mask & (k_traj == 1)) | s1_at_goal_boot

            # Bootstrap target:
            # if s_t0 at goal: 0.0
            # elif s_t1 at goal: -1.0
            # else: -1.0 + V(s_t1, g)
            td_target = torch.where(
                s1_at_goal,
                torch.full_like(v_t1, self.step_cost),
                self.step_cost + v_t1
            )

            td_target = torch.where(
                s0_at_goal,
                torch.zeros_like(v_t1),
                td_target
            ).clamp(min=self.failure_value, max=0.0)

            # Demonstration witness:
            # if g == s_{t+k}, then demo proves d(s_t, g) <= k,
            # so V(s_t, g) >= -k
            td_in = k_traj * self.step_cost
            td_in = td_in.clamp(min=self.failure_value, max=0.0)

            td_target = torch.where(
                traj_mask,
                torch.maximum(td_in, td_target),
                td_target
            )

        v1_pred, v2_pred = self.online_value_net(state=s_t0, goal=goal_st0)   

        value_loss = (
            F.smooth_l1_loss(v1_pred, td_target.detach()) +
            F.smooth_l1_loss(v2_pred, td_target.detach())
        ).mean()
        
        with torch.no_grad():
            v = (v1_pred + v2_pred) / 2
            steps = -1.0 * v * self.max_task_length
    
        info_dict['value/value_loss'] = value_loss.detach().item()
        info_dict['value/v_mean'] = v.detach().mean().item()
        info_dict['value/v_max'] = v.detach().max().item()
        info_dict['value/v_min'] = v.detach().min().item()
        info_dict['value/s0_at_goal_pctg'] = s0_at_goal.float().detach().mean().item()
        info_dict['value/s1_at_goal_pctg'] = s1_at_goal.float().detach().mean().item()
        info_dict['value/steps_traj_mean'] = steps[traj_mask].mean().item()
        info_dict['value/steps_rand_mean'] = steps[rand_mask].mean().item()
        info_dict['value/steps_rand_min'] = steps[rand_mask].min().item()
        info_dict['value/steps_rand_max'] = steps[rand_mask].max().item()
        info_dict["value/failure_target_pctg"] = (td_target <= -0.99).float().mean().item()

        return value_loss, info_dict

    def forward_actor_train(self, data_dict, info_dict):
        """
        Compute AWR actor loss
        """
        s_t0 = data_dict['s_t0']
        s_t1 = data_dict['s_t1']
        a_t0 = data_dict['a_t0']
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']

        goal_st0 = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)
        with torch.no_grad():
            v1_t0, v2_t0 = self.target_value_net(state=s_t0, goal=goal_st0)
    
            goal_st1 = self._build_goal_input(state=s_t1, goal_state=goal_state, goal_query=goal_query)
            v1_t1, v2_t1 = self.target_value_net(state=s_t1, goal=goal_st1)

        v_t0 = ((v1_t0 + v2_t0) / 2).clamp(min=-1.0, max=0.0)
        v_t1 = ((v1_t1 + v2_t1) / 2).clamp(min=-1.0, max=0.0)
        # Steps to goal.
        max_task_length = float(self.max_task_length)
        d_t0 = -v_t0 * max_task_length
        d_t1 = -v_t1 * max_task_length
        # Positive if transition moves closer to goal.
        adv = d_t0 - d_t1
        
        exp_a = torch.exp(adv * self.alpha)
        exp_a_clamp = exp_a.clamp_max(100.0)

        dist = self.actor_net(state=s_t0, goal=goal_st0)
        log_prob = dist.log_prob(a_t0)

        actor_loss = -(exp_a_clamp * log_prob).mean()

        info_dict['awr/actor_loss'] = actor_loss.detach().item()
        info_dict['awr/adv_mean'] = adv.detach().mean().item()
        info_dict['awr/adv_max'] = adv.detach().max().item()
        info_dict['awr/adv_min'] = adv.detach().min().item()
        info_dict['awr/bc_log_prob'] = log_prob.detach().mean().item()
        info_dict['awr/mse'] = ((dist.mode - a_t0)**2).detach().mean().item()
        info_dict['awr/std'] = dist.stddev.detach().mean().item()
        info_dict['awr/exp_a_mean'] = exp_a.detach().mean().item()
        info_dict['awr/exp_a_max'] = exp_a.detach().max().item()
        info_dict['awr/exp_a_min'] = exp_a.detach().min().item()

        return actor_loss, info_dict