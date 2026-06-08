from src.agent.networks.gc_mlp import GC_MLP
import torch.nn as nn

class GC_ValueHead(nn.Module):

    def __init__(self, state_dim, goal_dim, out_dim, last_layer, num_hidden, hidden_dim):
        super().__init__() 
        self.net = GC_MLP(
            state_dim=state_dim, 
            goal_dim=goal_dim, 
            out_dim=out_dim,
            hidden_dim=hidden_dim, 
            num_hidden=num_hidden
        )
        self.last_layer = last_layer
    
    def forward(self, state, goal):
        v = self.net(state=state, goal=goal)
        v = self.last_layer(v)
        return v.squeeze(-1)

class GC_DoubleValue(nn.Module):

    def __init__(self, state_dim, goal_dim, last_layer, hidden_dim=256, num_hidden=3):
        super().__init__()
        out_dim = 1

        self.v1 = GC_ValueHead(
            state_dim=state_dim,
            goal_dim=goal_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            last_layer=last_layer
        )

        self.v2 = GC_ValueHead(
            state_dim=state_dim,
            goal_dim=goal_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            last_layer=last_layer
        )

    def forward(self, state, goal):
        v1 = self.v1(state, goal)
        v2 = self.v2(state, goal)
        return v1, v2
    