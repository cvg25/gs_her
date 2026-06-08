import torch.nn as nn
import torch

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, act_layer=nn.GELU):
        super().__init__()
        create_hidden_layer = lambda hidden_dim: nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_layer()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            act_layer(),
            *[create_hidden_layer(hidden_dim=hidden_dim) for _ in range(num_hidden)],
            nn.Linear(in_features=hidden_dim, out_features=out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

class GC_MLP(nn.Module):

    def __init__(self, state_dim, goal_dim, out_dim, hidden_dim, num_hidden, act_layer=nn.GELU):
        super().__init__()
        self.goal_emb_dim = max(8, goal_dim//2)
        self.goal_encoder = MLP(
            in_dim=goal_dim,
            out_dim=self.goal_emb_dim,
            hidden_dim=hidden_dim,
            num_hidden=1
        )
        self.norm = nn.LayerNorm(self.goal_emb_dim)
        self.mlp = MLP(
            in_dim=state_dim + self.goal_emb_dim, 
            out_dim=out_dim, 
            hidden_dim=hidden_dim, 
            num_hidden=num_hidden
        )

    def forward(self, state, goal):
        g = self.goal_encoder(goal)
        g = self.norm(g)
        s_g = torch.cat((state,g), dim=-1)
        x = self.mlp(s_g)
        return x 