from src.agent.networks.gc_mlp import GC_MLP
import torch.nn as nn
from torch.distributions import Independent, Normal
import torch

class GC_Actor(nn.Module):

    def __init__(
        self,
        state_dim, 
        goal_dim, 
        action_dim, 
        hidden_dim=256, 
        num_hidden=3
    ):
        super().__init__()
        self.net = GC_MLP(
            state_dim=state_dim, 
            goal_dim=goal_dim, 
            out_dim=action_dim,
            hidden_dim=hidden_dim, 
            num_hidden=num_hidden
        )

    def forward(self, state, goal, temperature=1.0):
        means = self.net(state=state, goal=goal)
        log_stds = torch.zeros_like(means)
        stds  = torch.exp(log_stds) * temperature
        dist = Independent(Normal(loc=means, scale=stds), 1)
        return dist

    def act(self, state, goal, temperature=1e-8):
        dist = self.forward(state, goal, temperature)
        actions = dist.sample()
        return actions