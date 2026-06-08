import torch
import math

class GCDataSampler():
    """
    DataSampler class for Goal Conditioned RL.
    The goals are sampled from the current state, future states in the same trajectory and random states.
    """

    def __init__(self, datasampler, device, keep_dataset_on_device):
        self.datasampler = datasampler
        self.state_dim = self.datasampler.obs_dim
        self.action_dim = self.datasampler.act_dim
        self.max_task_length = self.datasampler.max_task_length
        self.proj_goal_idxs = self.datasampler.proj_goal_idxs
        self.state_semantics = self.datasampler.state_semantics

        self.device = device
        self.keep_dataset_on_device = keep_dataset_on_device
        if keep_dataset_on_device:
            self.datasampler.to(self.device)
        
        self.states = self.datasampler.states
        self.actions = self.datasampler.actions

        self.num_trajectories = self.actions.shape[0]
        self.num_steps = self.actions.shape[1]
        self.queries_cache = None 

    def compute_normalization_stats(self, normalize_state, normalize_action):
        self.states = normalize_state(self.states, update_stats=True)
        self.actions = normalize_action(self.actions, update_stats=True)
        return normalize_state, normalize_action

    def sample(
        self,
        batch_size,
        value_p_goal_curr,
        value_p_goal_traj,
        value_p_goal_rand,
        actor_p_goal_curr,
        actor_p_goal_traj,
        actor_p_goal_rand,
        query_strategy
    ):
        B = batch_size
        T = self.num_steps
        N = self.num_trajectories
        
        batch_idxs = torch.randint(0, N, (B,), device=self.device)
        t0_idxs = torch.randint(0, T - 1, (B,), device=self.device)

        data_dict = {}

        # Hindsight Goal Relabeling
        value_p = value_p_goal_curr + value_p_goal_traj + value_p_goal_rand
        if value_p > 0.0:
            assert abs(value_p - 1.0) < 1e-6, f"Error {value_p} != 1.0"

            data_dict['value'] = self.sample_data_dict(
                batch_idxs=batch_idxs,
                t0_idxs=t0_idxs,
                p_goal_curr=value_p_goal_curr,
                p_goal_traj=value_p_goal_traj,
                p_goal_rand=value_p_goal_rand,
                query_strategy=query_strategy)
        else:
            data_dict['value'] = {}

        actor_p = actor_p_goal_curr + actor_p_goal_traj + actor_p_goal_rand
        if actor_p > 0.0:
            assert actor_p == 1.0, f'Error {actor_p} != 1.0'
            
            data_dict['actor'] = self.sample_data_dict(
                batch_idxs=batch_idxs,
                t0_idxs=t0_idxs,
                p_goal_curr=actor_p_goal_curr,
                p_goal_traj=actor_p_goal_traj,
                p_goal_rand=actor_p_goal_rand,
                query_strategy=query_strategy)
        else:
            data_dict['actor'] = {}
            
        return data_dict
    
    def sample_data_dict(
        self,
        batch_idxs,
        t0_idxs,
        p_goal_curr,
        p_goal_traj,
        p_goal_rand,
        query_strategy):
        B = batch_idxs.shape[0]
        T = self.num_steps
        Ds = self.states.shape[-1]

        # Sample Hindsight Goal labels.
        
        # Current goal: g = s_t0
        g_curr = self.states[batch_idxs, t0_idxs]

        # Same trajectory future goal
        # k_offset = torch.randint(1, self.max_task_length, (B,), device=self.device)
        alpha = 2.0
        k_offset = torch.floor((torch.rand((B,), device=self.device) ** alpha) * self.max_task_length).long() + 1
        future_t = torch.minimum(t0_idxs + k_offset, torch.full_like(t0_idxs, T))
        g_traj = self.states[batch_idxs, future_t]
        k_traj = future_t - t0_idxs

        # Random goal from the whole dataset.
        rand_B = torch.randint(0, self.num_trajectories, (B,), device=self.device)
        rand_T = torch.randint(0, T + 1, (B,), device=self.device)
        g_rand = self.states[rand_B, rand_T]        

        # Sample goal type.
        probs = torch.tensor(
            [p_goal_curr, p_goal_traj, p_goal_rand],
            dtype=torch.float,
            device=self.device,
        )
        probs = probs / probs.sum()

        goal_type = torch.multinomial(probs, num_samples=B, replacement=True)
        # 0 -> curr, 1 -> traj, 2 -> rand
        # Masks.
        curr_mask = goal_type == 0
        traj_mask = goal_type == 1
        rand_mask = goal_type == 2

        goal = torch.empty((B, Ds), dtype=self.states.dtype, device=self.device)
        goal[curr_mask] = g_curr[curr_mask]
        goal[traj_mask] = g_traj[traj_mask]
        goal[rand_mask] = g_rand[rand_mask]

        # GS-HER random goal queries.
        if query_strategy is None:
            goal_query = None
        elif query_strategy == 'blockwise':
            goal_query = self.sample_random_block_goal_query(batch_size=B)
        elif query_strategy == 'semantic':
            goal_query = self.sample_random_semantic_goal_query(batch_size=B)
        elif query_strategy == 'uniform': 
            goal_query = self.sample_random_uniform_goal_query(batch_size=B)      
        else:
            raise Exception(f'Goal query strategy not available: {query_strategy}')

        s_t0 = self.states[batch_idxs, t0_idxs]
        s_t1 = self.states[batch_idxs, t0_idxs + 1]
        a_t0 = self.actions[batch_idxs, t0_idxs]

        data_dict = {
            's_t0': s_t0,
            'a_t0': a_t0,
            's_t1': s_t1,
            'goal': goal,
            'goal_query': goal_query,
            'curr_mask': curr_mask,
            'traj_mask': traj_mask,
            'rand_mask': rand_mask,
            'k_traj': k_traj.float(),
        }

        return data_dict

    def sample_random_block_goal_query(self, batch_size):
        if self.queries_cache is None:
            self.queries_cache = self._build_random_block_goal_query_cache()

        query_idxs = torch.randint(0, len(self.queries_cache), (batch_size,), device=self.device)
        return self.queries_cache[query_idxs]

    def sample_random_semantic_goal_query(self, batch_size):
        if self.queries_cache is None:
            self.queries_cache = self._build_semantic_block_goal_query_cache()

        query_idxs = torch.randint(0, len(self.queries_cache), (batch_size,), device=self.device)
        return self.queries_cache[query_idxs]

    def _build_random_block_goal_query_cache(self, cache_size: int = 131_072):
        """
        Builds a binary query/target mask cache for SSL state masking.
        Samples one or more contiguous coordinate spans. Uses state-vector locality, but no semantic labels.

        Three query types (sampled per-element):
          • full    all dimensions active (p_full)
          • random  random independent coordinates (p_random)
          • block   M independent contiguous spanse (remainder)

        Block sampling follows:
          - Scale   ~ Uniform(s_min, s_max)  as a *fraction* of D
          - Start   ~ Uniform(0, D - length)
          - M       ~ Uniform(1, max_blocks)
          - Blocks are sampled independently; target = union of spans
        """
        C      = cache_size
        D      = self.state_dim
        device = self.device

        # ── Hyperparameters ──────────────────────────────────────────────────────
        p_full        = 0.15
        p_random      = 0.15
        max_blocks    = 4
        s_min, s_max  = 0.15, 0.40   # block scale as fraction of D

        # ── Query-type assignment ────────────────────────────────────────────────
        r         = torch.rand(C, device=device)
        is_full   = r < p_full
        is_random = (r >= p_full) & (r < p_full + p_random)
        is_block  = ~(is_full | is_random)

        query = torch.zeros(C, D, device=device)

        # ── Full queries ─────────────────────────────────────────────────────────
        query[is_full] = 1.0

        # ── Random-coordinate queries ─────────────────────────────────────────────
        if is_random.any():
            r_idx = is_random.nonzero(as_tuple=True)[0]
            query[r_idx] = self.sample_random_uniform_goal_query(batch_size=r_idx.shape[0])

        # ── Block queries ──────────────────────────────────────────
        if is_block.any():
            b_idx = is_block.nonzero(as_tuple=True)[0]    # (K,)
            K     = b_idx.shape[0]

            pos      = torch.arange(D, device=device)     # (D,)
            blk_mask = torch.zeros(K, D, dtype=torch.bool, device=device)

            # Number of blocks per sample, drawn once
            n_blocks = torch.randint(1, max_blocks + 1, (K,), device=device)

            for k in range(max_blocks):
                active = n_blocks > k                    # (K,) which samples use block k
                if not active.any():
                    break

                # ── Scale ~ Uniform(s_min, s_max) as fraction of D ──────────────
                scale  = torch.rand(K, device=device) * (s_max - s_min) + s_min
                length = (scale * D).long().clamp(min=1, max=D)           # (K,)

                # ── Start ~ Uniform(0, D - length) ──────────────────────────────
                max_start = (D - length).clamp(min=0)                     # (K,)
                # uniform in {0, ..., max_start}; +1 makes max_start reachable
                start = (torch.rand(K, device=device)
                         * (max_start + 1).float()).long().clamp(0, D - 1)
                end   = (start + length).clamp(max=D)                     # (K,)

                span = (pos.unsqueeze(0) >= start.unsqueeze(1)) & \
                       (pos.unsqueeze(0) <  end.unsqueeze(1))  & \
                       active.unsqueeze(1)                                 # (K, D)

                blk_mask |= span

            # ── Guarantee non-empty (rare edge case) ─────────────────────────────
            empty = ~blk_mask.any(dim=1)
            if empty.any():
                e_idx = empty.nonzero(as_tuple=True)[0]
                rand_pos = torch.randint(0, D, (e_idx.shape[0],), device=device)
                blk_mask[e_idx, rand_pos] = True

            query[b_idx] = blk_mask.float()

        return query
    
    def sample_random_uniform_goal_query(self, batch_size):
        """
        Returns a [batch_size, dim] mask with 1.0 = visible, 0.0 = dropped.

        For each row: 
            - sample N_drop uniformly from {0, ..., dim-1}
            - choose N_drop positions uniformly at random
        """
        
        B = batch_size
        D = self.state_dim
        n_drop = torch.randint(0, D, size=(B,), device=self.device)
        perm = torch.rand(size=(B, D), device=self.device).argsort(dim=1)
        
        # In permutation order, first n_drop entries are dropped.
        visible_in_perm_order = torch.arange(D, device=self.device)[None, :] >= n_drop[:, None]

        mask = torch.empty(B, D, dtype=torch.bool, device=self.device)

        # Scatter visibility values back to original token positions.
        mask.scatter_(dim=1, index=perm, src=visible_in_perm_order)
        return mask.float()

    def _build_semantic_block_goal_query_cache(self, cache_size: int = 131_072) -> torch.Tensor:
        """
        Oracle semantic-block query cache.

        K is drawn from a mixture of two distributions:
          - Log-uniform  P(K=k) ∝ 1/k  — single-group prior (sparse tasks)
          - Uniform      P(K=k) = 1/G  — flat coverage    (multi-group tasks)

        The mixture weight p_sparse is a hyperparameter (default 0.6).
        This avoids both:
          - Uniform's under-representation of single-group targets
          - Log-uniform's under-representation of multi-group targets

        Returns:
            Float tensor (cache_size, D) with values in {0, 1}.
        """
        C      = cache_size
        D      = self.state_dim
        device = self.device

        semantic_keys = list(self.state_semantics.keys())
        G             = len(semantic_keys)

        # ── Group membership matrix (G, D) ────────────────────────────────────────
        group_basis = torch.zeros(G, D, device=device)
        for g, key in enumerate(semantic_keys):
            group_basis[g, self.state_semantics[key]] = 1.0

        # ── Sample K from mixture ─────────────────────────────────────────────────
        p_sparse   = 0.6                                                   # mixture weight
        use_sparse = torch.rand(C, device=device) < p_sparse              # (C,) bool

        if G == 1:
            num_active = torch.ones(C, dtype=torch.long, device=device)
        else:
            # Log-uniform branch: P(K=k) ∝ 1/k
            log_k        = torch.rand(C, device=device) * math.log(G + 1)
            k_log_uniform = log_k.exp().long().clamp(min=1, max=G)        # (C,)

            # Uniform branch: P(K=k) = 1/G
            k_uniform = torch.randint(1, G + 1, (C,), device=device)     # (C,)

            num_active = torch.where(use_sparse, k_log_uniform, k_uniform)  # (C,)

        # ── Without-replacement group selection ───────────────────────────────────
        group_order = torch.rand(C, G, device=device).argsort(dim=1)      # (C, G)
        rank        = torch.arange(G, device=device).unsqueeze(0)         # (1, G)
        in_rank     = rank < num_active.unsqueeze(1)                      # (C, G)

        selection = torch.zeros(C, G, dtype=torch.bool, device=device)
        selection.scatter_(1, group_order, in_rank)                       # (C, G)

        # ── Expand to state dims ──────────────────────────────────────────────────
        queries = (selection.float() @ group_basis).clamp(max=1.0)        # (C, D)

        return queries