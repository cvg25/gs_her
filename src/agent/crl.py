from src.agent.networks.gc_actor import GC_Actor
from src.agent.networks.gc_mlp import MLP
from src.utils import compute_num_trainable_params, compute_gradient_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager

class ContrastiveMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=256, num_hidden=3):
        super().__init__()
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, num_hidden=num_hidden)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        return x

class GCBilinearCritic(nn.Module):
    """
    Q(s, a, g) = phi(s, a)^T  psi(g) / sqrt(latent_dim)
 
    Two independent (phi, psi) pairs form an ensemble of size 2,
    matching ensemble=True in the JAX GCBilinearValue.
 
    forward(state, goal, action, info=False)
        info=False -> (q1, q2)  per-sample scalars
        info=True  -> (v, phi, psi) where
                        v   : (B,)      mean diagonal value
                        phi : (2, B, D) stacked embeddings for contrastive loss
                        psi : (2, B, D)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        latent_dim
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = self.latent_dim ** 0.5
        sa_dim = state_dim + action_dim

        self.phi1 = ContrastiveMLP(in_dim=sa_dim, out_dim=latent_dim)
        self.phi2 = ContrastiveMLP(in_dim=sa_dim, out_dim=latent_dim)
        self.psi1 = ContrastiveMLP(in_dim=goal_dim, out_dim=latent_dim)
        self.psi2 = ContrastiveMLP(in_dim=goal_dim, out_dim=latent_dim)

    def forward(self, state, goal, action, info=False):
        sa = torch.cat([state, action], dim=-1)
        
        phi1 = self.phi1(sa) # B, D
        phi2 = self.phi2(sa) # B, D
        psi1 = self.psi1(goal) # B, D
        psi2 = self.psi2(goal) # B, D

        if info:
            # Pack into (E=2, B, D) for the einsum in contrastive_loss
            phi = torch.stack([phi1, phi2], dim=0)  # (2, B, D)
            psi = torch.stack([psi1, psi2], dim=0)  # (2, B, D)
            # Diagonal entries (positive pairs), averaged over ensemble
            v = ((phi1 * psi1).sum(-1) + (phi2 * psi2).sum(-1)) / (2.0 * self.scale)  # (B,)
            return v, phi, psi
        else:
            q1 = (phi1 * psi1).sum(-1) / self.scale  # (B,)
            q2 = (phi2 * psi2).sum(-1) / self.scale  # (B,)
            return q1, q2

@contextmanager
def freeze_params(module):
    old = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(module.parameters(), old):
            p.requires_grad_(r)

class CRL(nn.Module):
    """
    CRL agent.
    """
    def __init__(self, params):
        super().__init__()
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.proj_goal_idxs = params["proj_goal_idxs"]
        self.max_task_length = params['max_task_length']

        # Data
        self.value_p_goal_curr = 0.0
        self.value_p_goal_traj = 1.0
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
            
        # Critic
        self.discount = 0.99
        self.critic_net = GCBilinearCritic(        
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            latent_dim=256)

        # Actor
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
        actions = self.actor_net.act(state=state, goal=goal)
        
        if compute_distance:
            # Compute distance to goal, appends the negative Q-value as a proxy distance.
            q1, q2 = self.critic_net(state, goal, actions)
            q = (q1 + q2) / 2.0
            self.distance_to_goal.append((-q).mean().item())

        return actions

    def _contrastive_loss(self, state, goal, action):
        """
        Sigmoid binary cross-entropy contrastive loss.
 
        Within a batch of size B the (BxB) logit matrix has positive pairs
        on the diagonal and negatives off-diagonal, exactly as in the JAX code.
 
        logits[i, j, e] = phi_e(s_i, a_i)^T psi_e(g_j) / sqrt(D)
        """
        B = state.shape[0]
        v, phi, psi = self.critic_net(state, goal, action, info=True)
        # phi, psi: (E=2, B, D)

        # Replicate JAX einsum 'eik,ejk->ije'
        logits = torch.einsum('eik,ejk->ije', phi, psi) / (self.critic_net.latent_dim ** 0.5)
        # logits: (B, B, E)

        I = torch.eye(B, device=state.device)   # (B, B)
        I_e = I.unsqueeze(-1).expand_as(logits) # (B, B, E)

        loss = F.binary_cross_entropy_with_logits(logits, I_e, reduction='mean')
        
        # Diagnostics (no grad needed)
        with torch.no_grad():
            logits_mean = logits.mean(dim=-1)                  # (B, B)
            correct = logits_mean.argmax(1) == torch.arange(B, device=state.device)
            logits_pos = (logits_mean * I).sum() / I.sum()
            logits_neg = (logits_mean * (1 - I)).sum() / (1 - I).sum()
            v_exp = v.exp()
 
        info = {
            'critic/contrastive_loss'   : loss.item(),
            'critic/v_mean'             : v_exp.mean().item(),
            'critic/v_max'              : v_exp.max().item(),
            'critic/v_min'              : v_exp.min().item(),
            'critic/binary_accuracy'    : ((logits_mean > 0) == I.bool()).float().mean().item(),
            'critic/categorical_accuracy': correct.float().mean().item(),
            'critic/logits_pos'         : logits_pos.item(),
            'critic/logits_neg'         : logits_neg.item(),
            'critic/logits'             : logits_mean.mean().item(),
        }
        return loss, info


    def _ddpgbc_actor_loss(self, state, goal, action):
        """
        DDPG+BC actor loss:
            L = -Q(s, pi(s), g) / |Q|_mean   +   -alpha * log pi(a | s, g)
 
        The Q normalisation makes the scale invariant to the critic's output range.
        """
        dist = self.actor_net(state, goal)
        q_action = dist.mean.clamp(-1.0, 1.0)
 
        with freeze_params(self.critic_net):
            q1, q2 = self.critic_net(state, goal, q_action)
        q = torch.minimum(q1, q2) 
        # Scale-invariant Q loss
        q_loss = -q.mean() / (q.abs().mean().detach() + 1e-6)
 
        # Behaviour-cloning regularisation evaluated on dataset actions
        log_prob = dist.log_prob(action)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(-1)
            
        bc_loss  = -(self.alpha * log_prob).mean()
        actor_loss = q_loss + bc_loss
 
        info = {
            'actor/actor_loss'  : actor_loss.item(),
            'actor/q_loss'      : q_loss.item(),
            'actor/bc_loss'     : bc_loss.item(),
            'actor/q_mean'      : q.mean().item(),
            'actor/q_abs_mean'  : q.abs().mean().item(),
            'actor/bc_log_prob' : log_prob.mean().item(),
            'actor/mse'         : ((dist.mean - action) ** 2).mean().item(),
            'actor/std'         : dist.base_dist.scale.mean().item(),
        }
        return actor_loss, info
 
    def forward_train(self, data_dict):
        """
        Compute total loss and return (loss, info_dict).
 
        Expects data_dict with keys 'value' and 'actor',
        each containing 's_t0', 'a_t0', 'goal'.
        """
        info: dict = {}
 
        critic_loss, info = self.forward_value_train(data_dict['value'], info)
        actor_loss,  info = self.forward_actor_train(data_dict['actor'],  info)
 
        loss = critic_loss + actor_loss
        info['total_loss'] = loss.detach().item()
        return loss, info
 
    def forward_value_train(self, data_dict, info_dict):
        """Contrastive critic update."""
        s_t0 = data_dict['s_t0']
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
        goal_st0 = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        loss, critic_info = self._contrastive_loss(
            state=s_t0,
            goal=goal_st0,
            action = data_dict['a_t0'],
        )
        info_dict.update(critic_info)
        return loss, info_dict
 
    def forward_actor_train(self, data_dict, info_dict):
        """DDPG+BC actor update."""
        s_t0 = data_dict['s_t0']
        goal_state=data_dict['goal']
        goal_query=data_dict['goal_query']
        goal_st0 = self._build_goal_input(state=s_t0, goal_state=goal_state, goal_query=goal_query)

        loss, actor_info = self._ddpgbc_actor_loss(
            state=s_t0,
            goal=goal_st0,
            action = data_dict['a_t0'],
        )
        info_dict.update(actor_info)
        return loss, info_dict
 
    def clip_grad_norm(self, max_norm, info):
        info['gn_critic'] = compute_gradient_norm(self.critic_net.parameters())
        info['gn_actor']  = compute_gradient_norm(self.actor_net.parameters())
        
        params = list(self.critic_net.parameters()) + list(self.actor_net.parameters())

        if self.her_type == "gs_her_learned":
            params += [self.goal_emb, self.state_emb]

        clip_grad_norm_(params, max_norm)
        
        return info
 
    def ema_update(self, ema_value = 0.995):
        """
        CRL uses no target network, so this is a no-op kept for interface compatibility.
        """
        pass
 
    def get_trainable_params(self):
        return {
            'num_params_critic': compute_num_trainable_params(self.critic_net),
            'num_params_actor' : compute_num_trainable_params(self.actor_net),
        }