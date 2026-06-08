from src.agent.networks.gc_value import GC_DoubleValue
from src.agent.networks.gc_actor import GC_Actor
from src.utils import compute_num_trainable_params, compute_gradient_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy


class GCIQL(nn.Module):
    """
    Goal-Conditioned Implicit Q-Learning.
    """

    def __init__(self, params):
        super().__init__()

        self.state_dim = params["state_dim"]
        self.action_dim = params["action_dim"]
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
            
            
        # ------------------------------------------------------------------
        # Hyperparameters.
        # ------------------------------------------------------------------
        self.expectile = 0.9
        self.discount = 0.99

        # DDPG+BC coefficient.
        self.alpha = params["agent"]["alpha"]

        # ------------------------------------------------------------------
        # Value V(s, g).
        # ------------------------------------------------------------------
        self.online_value_net = GC_DoubleValue(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            last_layer=lambda x: -1.0 * F.softplus(x)

        )

        # ------------------------------------------------------------------
        # Critic Q(s, g, a).
        #
        # Implemented as GC_DoubleValue over concat(s, a).
        # ------------------------------------------------------------------
        self.online_critic_net = GC_DoubleValue(
            state_dim=self.state_dim + self.action_dim,
            goal_dim=self.goal_dim,
            last_layer=lambda x: x,
        )

        self.target_critic_net = copy.deepcopy(self.online_critic_net)
        self.target_critic_net.requires_grad_(False)
        self.target_critic_net.eval()

        # ------------------------------------------------------------------
        # Actor pi(a | s, g).
        # ------------------------------------------------------------------
        self.actor_net = GC_Actor(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
        )

        self.reset()

    def reset(self):
        self.distance_to_goal = []

    # ----------------------------------------------------------------------
    # Goal utilities.
    # ----------------------------------------------------------------------
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


    def _critic_forward(self, critic_net, state, goal, action):
        state_action = torch.cat((state, action), dim=-1)
        return critic_net(state=state_action, goal=goal)

    def _dist_mode(self, dist):
        mode = dist.mode
        return mode() if callable(mode) else mode

    def value_to_steps(self, value):
        """
        Approximate conversion from negative discounted value to steps-to-goal.

        If gamma == 1:
            V = -T

        If gamma < 1:
            V = -(1 - gamma^T) / (1 - gamma)
        """
        if abs(self.discount - 1.0) < 1e-6:
            return -value

        gamma = value.new_tensor(self.discount)
        argument = 1.0 + value * (1.0 - gamma)
        argument = argument.clamp(min=1e-6, max=1.0)
        return torch.log(argument) / torch.log(gamma)

    def steps_to_value(self, steps):
        """
        Convert demonstrated steps-to-goal into the GCIQL value scale.

        If gamma == 1:
            V = -T

        If gamma < 1:
            V = -(1 - gamma^T) / (1 - gamma)
        """
        steps = steps.float()

        if abs(self.discount - 1.0) < 1e-6:
            return -steps

        gamma = steps.new_tensor(self.discount)
        return -(1.0 - torch.pow(gamma, steps)) / (1.0 - gamma)

    def _expectile_loss(self, adv, diff):
        weight = torch.where(adv >= 0, self.expectile, 1.0 - self.expectile)
        return weight * (diff ** 2)

    # ----------------------------------------------------------------------
    # Inference.
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def act(self, state, goal, compute_distance=False):
        goal = self._build_goal_input(state=state, goal_state=goal, goal_query=self.task_query)

        # Compute distance to goal.
        if compute_distance:
            v1, v2 = self.target_value_net(state=state, goal=goal)
            v = (v1 + v2) / 2.0
            steps_to_goal = self.value_to_steps(v)
            self.distance_to_goal.append(steps_to_goal.mean().item())

        return self.actor_net.act(state=state, goal=goal)

    # ----------------------------------------------------------------------
    # Logging / optimization helpers.
    # ----------------------------------------------------------------------

    def get_trainable_params(self):
        params = {
            "num_params_actor_net": compute_num_trainable_params(model=self.actor_net),
            "num_params_value_net": compute_num_trainable_params(model=self.online_value_net),
            "num_params_critic_net": compute_num_trainable_params(model=self.online_critic_net),
        }

        return params

    def clip_grad_norm(self, max_norm, info):
        info["gn_value_net"] = compute_gradient_norm(self.online_value_net.parameters())
        info["gn_critic_net"] = compute_gradient_norm(self.online_critic_net.parameters())
        info["gn_actor_net"] = compute_gradient_norm(self.actor_net.parameters())

        params = (
            list(self.online_value_net.parameters())
            + list(self.online_critic_net.parameters())
            + list(self.actor_net.parameters())
        )

        if self.her_type == "gs_her_learned":
            params += [self.goal_emb, self.state_emb]

        clip_grad_norm_(params, max_norm)

        return info

    def ema_update(self, ema_value=0.995):
        """
        Same EMA convention as GCIVL:

            target = ema_value * target + (1 - ema_value) * online

        In GCIQL, the target network is the critic.
        """
        with torch.no_grad():
            for p, p_targ in zip(self.online_critic_net.parameters(), self.target_critic_net.parameters()):
                p_targ.data.mul_(ema_value).add_((1.0 - ema_value) * p.data)

    # ----------------------------------------------------------------------
    # Training.
    # ----------------------------------------------------------------------

    def forward_train(self, data_dict):
        info = {}

        s_t0 = data_dict["value"]["s_t0"]
        goal_state=data_dict["value"]['goal']
        goal_query=data_dict["value"]['goal_query']
        goal = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        value_loss, info = self.forward_value_train(
            data_dict=data_dict["value"],
            goal=goal,
            info_dict=info,
        )

        critic_loss, info = self.forward_critic_train(
            data_dict=data_dict["value"],
            goal=goal,
            info_dict=info,
        )

        actor_loss, info = self.forward_actor_train(
            data_dict=data_dict["actor"],
            info_dict=info,
        )

        loss = value_loss + critic_loss + actor_loss
        info["total_loss"] = loss.detach().item()

        return loss, info

    def forward_value_train(self, data_dict, goal, info_dict):
        """
        IQL value loss:

            q = min(Q1_target(s, g, a), Q2_target(s, g, a))
            v = V(s, g)
            L_V = expectile_loss(q - v)

        Since GC_DoubleValue has two heads, both value heads are trained.
        """
        s_t0 = data_dict["s_t0"]
        a_t0 = data_dict["a_t0"]

        with torch.no_grad():
            q1_t, q2_t = self._critic_forward(
                critic_net=self.target_critic_net,
                state=s_t0,
                goal=goal,
                action=a_t0,
            )
            q = torch.minimum(q1_t, q2_t)

        v1, v2 = self.online_value_net(state=s_t0, goal=goal)
        v = (v1 + v2) / 2.0

        adv1 = q - v1
        adv2 = q - v2

        value_loss = 0.5 * (
            self._expectile_loss(adv1, adv1) +
            self._expectile_loss(adv2, adv2)
        ).mean()

        with torch.no_grad():
            steps = self.value_to_steps(v.detach())

        info_dict["value/value_loss"] = value_loss.detach().item()
        info_dict["value/v_mean"] = v.detach().mean().item()
        info_dict["value/v_max"] = v.detach().max().item()
        info_dict["value/v_min"] = v.detach().min().item()
        info_dict["value/q_mean"] = q.detach().mean().item()
        info_dict["value/q_max"] = q.detach().max().item()
        info_dict["value/q_min"] = q.detach().min().item()
        info_dict["value/adv_mean"] = (q - v).detach().mean().item()
        info_dict["value/steps_mean"] = steps.detach().mean().item()

        return value_loss, info_dict

    def forward_critic_train(self, data_dict, goal, info_dict):
        """
        IQL critic loss:

            target_q = r + gamma * mask * V(s', g)

        Using GCIVL-style curr_mask:
            curr_mask = 1 if s_t0 is already at goal
            reward = 0 if at goal else -1
            mask = 0 if at goal else 1
        """
        s_t0 = data_dict["s_t0"]
        s_t1 = data_dict["s_t1"]
        a_t0 = data_dict["a_t0"]

        curr_mask = data_dict["curr_mask"].float()
        not_at_goal_mask = 1.0 - curr_mask
        rewards = -1.0 * not_at_goal_mask
        masks = not_at_goal_mask

        k_traj = data_dict["k_traj"]
        traj_mask = data_dict["traj_mask"]

        with torch.no_grad():
            goal_state=data_dict['goal']
            goal_query=data_dict['goal_query']
            goal_st1 = self._build_goal_input(state=s_t1, goal_state=goal_state, goal_query=goal_query)
            next_v1, next_v2 = self.online_value_net(state=s_t1, goal=goal_st1)
            next_v = (next_v1 + next_v2) / 2.0
            target_q = rewards + self.discount * masks * next_v

            # ------------------------------------------------------------
            # Demonstration witness
            #
            # If the relabeled goal is s_{t+k}, then the dataset transition
            # proves that taking a_t from s_t can reach g within k steps.
            #
            # Therefore:
            #
            #     Q(s_t, a_t, g) >= steps_to_value(k)
            #
            # Since values are negative, use maximum.
            # ------------------------------------------------------------ 
            demo_q = self.steps_to_value(k_traj)

            target_q = torch.where(
                traj_mask,
                torch.maximum(target_q, demo_q),
                target_q
            )

        q1, q2 = self._critic_forward(
            critic_net=self.online_critic_net,
            state=s_t0,
            goal=goal,
            action=a_t0,
        )

        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        info_dict["critic/critic_loss"] = critic_loss.detach().item()
        info_dict["critic/q1_mean"] = q1.detach().mean().item()
        info_dict["critic/q2_mean"] = q2.detach().mean().item()
        info_dict["critic/target_q_mean"] = target_q.detach().mean().item()
        info_dict["critic/target_q_max"] = target_q.detach().max().item()
        info_dict["critic/target_q_min"] = target_q.detach().min().item()
        info_dict["critic/rewards_mean"] = rewards.detach().mean().item()
        info_dict["critic/curr_at_goal_pctg"] = curr_mask.detach().mean().item()

        return critic_loss, info_dict

    def forward_actor_train(self, data_dict, info_dict):
        """
        DDPG+BC actor loss:

            q_loss = -Q(s, g, pi(s, g)) / mean(abs(Q))
            bc_loss = -alpha * log pi(a_dataset | s, g)
            actor_loss = q_loss + bc_loss

        Critic parameters are frozen during q_loss so gradients flow into
        the actor through the action, but not into the critic.
        """
        s_t0 = data_dict["s_t0"]
        a_t0 = data_dict["a_t0"]
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
    
        goal = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        dist = self.actor_net(state=s_t0, goal=goal)

        q_actions = self._dist_mode(dist)
        q_actions = torch.clamp(q_actions, -1.0, 1.0)

        # Freeze critic params for actor update, but keep gradient through q_actions.
        critic_requires_grad = [p.requires_grad for p in self.online_critic_net.parameters()]
        for p in self.online_critic_net.parameters():
            p.requires_grad_(False)

        try:
            q1, q2 = self._critic_forward(
                critic_net=self.online_critic_net,
                state=s_t0,
                goal=goal,
                action=q_actions,
            )
            q = torch.minimum(q1, q2)
        finally:
            for p, req in zip(self.online_critic_net.parameters(), critic_requires_grad):
                p.requires_grad_(req)

        q_loss = -q.mean() / (q.detach().abs().mean() + 1e-6)

        log_prob = dist.log_prob(a_t0)
        bc_loss = -(self.alpha * log_prob).mean()

        actor_loss = q_loss + bc_loss

        info_dict["ddpgbc/actor_loss"] = actor_loss.detach().item()
        info_dict["ddpgbc/q_loss"] = q_loss.detach().item()
        info_dict["ddpgbc/bc_loss"] = bc_loss.detach().item()
        info_dict["ddpgbc/q_mean"] = q.detach().mean().item()
        info_dict["ddpgbc/q_abs_mean"] = q.detach().abs().mean().item()
        info_dict["ddpgbc/bc_log_prob"] = log_prob.detach().mean().item()
        info_dict["ddpgbc/mse"] = ((q_actions.detach() - a_t0) ** 2).mean().item()

        if hasattr(dist, "stddev"):
            info_dict["ddpgbc/std"] = dist.stddev.detach().mean().item()

        return actor_loss, info_dict