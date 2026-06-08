from src.agent.networks.gc_value import GC_DoubleValue
from src.agent.networks.gc_actor import GC_Actor
from src.utils import compute_num_trainable_params, compute_gradient_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy

class GCIVL(nn.Module):
    """
    GCIVL Agent
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
        self.expectile = 0.9
        self.discount = 0.99
        
        self.online_value_net = GC_DoubleValue(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            last_layer=lambda x: -1.0 * F.softplus(x)
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
        
        if compute_distance:
            # Compute distance to goal.
            v1, v2 = self.target_value_net(state=state, goal=goal)
            v = (v1 + v2) / 2
            steps_to_goal = self.value_to_steps(value=v)
            self.distance_to_goal.append(steps_to_goal.item())

        # Act
        actions = self.actor_net.act(state=state, goal=goal)
        return actions

    def value_to_steps(self, value):
        """
        Convert value estimate to steps-to-goal.
        If gamma = 1:
            V = -T
        If gamma < 1:
            V = -(1 - gamma^T) / (1 - gamma)
            T = log(1 + V * (1 - gamma)) / log(gamma)
        """
        if abs(self.discount - 1.0) < 1e-6:
            return -value

        gamma = value.new_tensor(self.discount)
        argument = 1.0 + value * (1.0 - gamma)
        argument = argument.clamp(min=1e-6, max=1.0)

        return torch.log(argument) / torch.log(gamma)

    def steps_to_value(self, steps):
        """
        Convert a number of steps-to-goal into the GCIVL value scale.

        If gamma = 1:
            V = -T

        If gamma < 1:
            V = -(1 - gamma^T) / (1 - gamma)
        """
        steps = steps.float()

        if abs(self.discount - 1.0) < 1e-6:
            return -steps

        gamma = steps.new_tensor(self.discount)
        return -(1.0 - torch.pow(gamma, steps)) / (1.0 - gamma)

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

    def _expectile_loss(self, adv, diff, expectile):
        """
        Compute the expectile loss.
        """
        weight = torch.where(adv >= 0, expectile, (1-expectile))
        return weight * (diff**2)

    def forward_value_train(self, data_dict, info_dict):
        s_t0 = data_dict['s_t0']
        s_t1 = data_dict['s_t1']

        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
        goal_st0 = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        curr_mask = data_dict['curr_mask']
        traj_mask = data_dict['traj_mask']
        rand_mask = data_dict['rand_mask']
        k_traj = data_dict['k_traj']

        B = s_t0.shape[0]
        device = s_t0.device

        not_at_goal_mask = 1.0 - curr_mask.float()
        rewards = -1.0 * not_at_goal_mask

        with torch.no_grad():
            # Next state value (min of heads) — for unified q target
            goal_st1 = self._build_goal_input(state=s_t1, goal_state=goal_state, goal_query=goal_query)
            v1_t1, v2_t1 = self.target_value_net(state=s_t1, goal=goal_st1)
            next_v_t = torch.minimum(v1_t1, v2_t1)
            q = rewards + self.discount * not_at_goal_mask * next_v_t
    
            # Per-head q targets (double-critic style)
            q1 = rewards + self.discount * not_at_goal_mask * v1_t1
            q2 = rewards + self.discount * not_at_goal_mask * v2_t1

            # ------------------------------------------------------------
            # Demonstration witness
            #
            # If the relabeled goal is s_{t+k}, then the demonstration proves:
            #
            #     d(s_t, g) <= k
            #
            # Therefore:
            #
            #     V(s_t, g) >= value_of_k_steps
            #
            # Since values are negative, "better" means numerically larger.
            # So we use torch.maximum, not torch.minimum.
            # ------------------------------------------------------------
            demo_v = self.steps_to_value(k_traj)
            q = torch.where(traj_mask, torch.maximum(q, demo_v), q)
            q1 = torch.where(traj_mask, torch.maximum(q1, demo_v), q1)
            q2 = torch.where(traj_mask, torch.maximum(q2, demo_v), q2)

            # Current state value from target net — for advantage
            v1_t0, v2_t0 = self.target_value_net(state=s_t0, goal=goal_st0)
            v_t0 = (v1_t0 + v2_t0) / 2
            adv = q - v_t0
    
        v1, v2 = self.online_value_net(state=s_t0, goal=goal_st0)
        v = (v1 + v2) / 2
    
        value_loss = (
            self._expectile_loss(adv, q1 - v1, self.expectile) +
            self._expectile_loss(adv, q2 - v2, self.expectile)
        ).mean()
        
        with torch.no_grad():
            steps = self.value_to_steps(value=v.detach())

        info_dict['value/value_loss'] = value_loss.detach().item()
        info_dict['value/v_mean'] = v.detach().mean().item()
        info_dict['value/v_max'] = v.detach().max().item()
        info_dict['value/v_min'] = v.detach().min().item()
        info_dict['value/rewards_mean'] = rewards.detach().mean().item()
        info_dict['value/curr_at_goal_pctg'] = curr_mask.float().detach().mean().item()
        info_dict['value/adv_traj_mean'] = adv[traj_mask].detach().mean().item() if traj_mask.any() else 0.0
        info_dict['value/adv_rand_mean'] = adv[rand_mask].detach().mean().item() if rand_mask.any() else 0.0
        info_dict['value/steps_traj_mean'] = steps[traj_mask].mean().item()
        info_dict['value/steps_rand_mean'] = steps[rand_mask].mean().item()
    
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
        
        v_t0 = (v1_t0 + v2_t0) / 2
        v_t1 = (v1_t1 + v2_t1) / 2
        adv = v_t1 - v_t0
        
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