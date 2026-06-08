import torch.nn as nn
import torch

class EMANormalize(nn.Module):
    """
    EMA running mean/std for (possibly batched) tensors.

    - Tracks per-dimension stats over the last dimension(s) given by `shape`.
    - If x has extra leading dims (e.g. [n_envs, *shape]), update reduces over those dims.
    - Bias-corrected EMA for mean and E[x^2].
    """
    def __init__(self, shape, alpha=1e-3, eps=1e-8, dtype=torch.float32):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)

        self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
        self.register_buffer("m2",   torch.zeros(shape, dtype=dtype))  # EMA of E[x^2]
        self.register_buffer("t",    torch.zeros((), dtype=torch.long))  # steps for bias correction

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.to(dtype=self.mean.dtype, device=self.mean.device)

        # Reduce over all leading dims beyond the tracked shape
        # Example: tracked shape (obs_dim,) and x is (n_envs, obs_dim) -> reduce over dim 0
        reduce_dims = tuple(range(x.ndim - self.mean.ndim))  # () if same ndim
        if len(reduce_dims) > 0:
            x_mean = x.mean(dim=reduce_dims)
            x_m2 = (x * x).mean(dim=reduce_dims)
        else:
            x_mean = x
            x_m2 = x * x

        a = self.alpha
        self.mean.mul_(1.0 - a).add_(a * x_mean)
        self.m2.mul_(1.0 - a).add_(a * x_m2)
        self.t.add_(1)

    @torch.no_grad()
    def stats(self):
        # Bias correction: divide by (1 - (1-a)^t)
        if self.t.item() == 0:
            var = (self.m2 - self.mean * self.mean).clamp_min(self.eps)
            return self.mean, var.sqrt()

        b = (1.0 - self.alpha) ** int(self.t.item())
        denom = (1.0 - b)

        mean_bc = self.mean / denom
        m2_bc   = self.m2   / denom
        var = (m2_bc - mean_bc * mean_bc).clamp_min(self.eps)
        return mean_bc, var.sqrt()

    def forward(self, x, update_stats):
        # Update with RAW x before normalization
        if update_stats:
            self.update(x)

        mean, std = self.stats()
        x = x.to(dtype=mean.dtype, device=mean.device)
        y = (x - mean) / std
        
        return y
    
    def undo(self, x: torch.Tensor):
        mean, std = self.stats()
        x = x * std + mean
        return x