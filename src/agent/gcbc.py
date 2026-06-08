from src.agent.networks.gc_actor import GC_Actor
from src.utils import compute_num_trainable_params, compute_gradient_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy

class GCBC(nn.Module):
    """
    Simplest Goal Conditioned Behavioral Cloning (GCBC) actor. 
    """
    
    def __init__(self, params):
        super().__init__()
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.proj_goal_idxs = params["proj_goal_idxs"]
        self.max_task_length = params['max_task_length']

        # Data
        self.value_p_goal_curr = 0.0
        self.value_p_goal_traj = 0.0
        self.value_p_goal_rand = 0.0
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

        # Actor.
        self.actor_net = GC_Actor(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
        )

        # inference
        self.reset()

    def ema_update(self, ema_value=0.995):
        pass

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
        actions = self.actor_net.act(state=state, goal=goal)
        return actions

    def get_trainable_params(self):
        params = {
            'num_params_actor_net': compute_num_trainable_params(model=self.actor_net),
        }
        return params

    def clip_grad_norm(self, max_norm, info):
        info['gn_actor_net'] = compute_gradient_norm(self.actor_net.parameters())

        params = list(self.actor_net.parameters())

        if self.her_type == "gs_her_learned":
            params += [self.goal_emb, self.state_emb]
            
        clip_grad_norm_(params, max_norm)

        return info

    def forward_train(self, data_dict):
        info = dict()

        actor_loss, info = self.forward_actor_train(
            data_dict=data_dict['actor'],
            info_dict=info)

        loss = actor_loss
        info['total_loss'] = loss.detach().item()

        return loss, info

    def forward_actor_train(self, data_dict, info_dict):
        """
        Compute AWR actor loss
        """
        s_t0 = data_dict['s_t0']
        a_t0 = data_dict['a_t0']
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
        goal = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        dist = self.actor_net(state=s_t0, goal=goal)
        log_prob = dist.log_prob(a_t0)

        actor_loss = -log_prob.mean()

        info_dict['bc/actor_loss'] = actor_loss.detach().item()
        info_dict['bc/mse'] = ((dist.mode - a_t0)**2).detach().mean().item()
        info_dict['bc/std'] = dist.stddev.detach().mean().item()

        return actor_loss, info_dict