"""CMC-inspired memory consolidation for SAC.

Changes vs. the original:
  * FIX 1: the actor anchor uses agent.actor.mean_action() (differentiable).
    The old code used actor.deterministic(), which is @torch.no_grad(), so the
    actor regularizer produced zero gradient and CMC never constrained the actor.
  * FIX 4: candidate states come from the most recent `recent_frac` of the task's
    replay buffer (skips warmup / early-training states).
  * FIX 4: the priority score is fixed and made scale-free:
       - policy disagreement  |mu(s) - a~pi(s)|^2
       - Q gap                |Q(s, mu(s)) - Q(s, a~)|
       - real TD error        |r + g(1-term)(minQt(s', a') - alpha log pi(a'|s')) - minQ(s, a)|
         (the old proxy bootstrapped from s instead of s', so it was not a TD error)
     Each term is rank-normalized to [0, 1] within the task, then averaged, so
     no single term (or task reward scale) dominates.
  * selection="uniform" gives the random-memory control (ablation).
  * Separate actor_loss() / critic_loss() so each update samples memory once.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _rank01(x):
    """Rank-normalize a 1-D tensor to [0, 1] (ties broken arbitrarily)."""
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    ranks = torch.empty_like(x)
    ranks[torch.argsort(x)] = torch.arange(n, dtype=x.dtype, device=x.device)
    return ranks / (n - 1)


class CounterfactualMemoryConsolidator:
    def __init__(self, memory, actor_lambda=1.0, critic_lambda=0.25,
                 selection="priority", recent_frac=0.3, gamma=0.99):
        assert selection in ("priority", "uniform")
        self.memory = memory
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.selection = selection
        self.recent_frac = recent_frac
        self.gamma = gamma

    # ----------------------------------------------------------- consolidation
    @torch.no_grad()
    def consolidate_task(self, agent, replay, task_id, max_candidates=5000,
                         device="cpu"):
        pool = replay.recent_indices(self.recent_frac)
        if len(pool) == 0 or self.memory.capacity <= 0:
            return {"candidates": 0, "kept": 0,
                    "per_task": self.memory.counts_per_task()}

        n = min(max_candidates, len(pool))
        idx = np.random.choice(pool, n, replace=False)

        obs_np = replay.obs[idx]
        nobs_np = replay.next_obs[idx]
        rew_np = replay.rewards[idx]
        finite = (np.isfinite(obs_np).all(1) & np.isfinite(nobs_np).all(1)
                  & np.isfinite(rew_np))
        idx = idx[finite]
        if len(idx) == 0:
            return {"candidates": 0, "kept": 0,
                    "per_task": self.memory.counts_per_task()}

        t = lambda a: torch.as_tensor(a, device=device)
        obs = t(replay.obs[idx])
        act = t(replay.actions[idx])
        rew = t(replay.rewards[idx])
        nobs = t(replay.next_obs[idx])
        term = t(replay.terminals[idx])

        old_action = agent.actor.deterministic(obs)
        old_q = torch.minimum(agent.q1(obs, old_action),
                              agent.q2(obs, old_action)).squeeze(-1)

        if self.selection == "uniform":
            priority = torch.rand(len(idx), device=device)
        else:
            stoch_a, _ = agent.actor.sample(obs)
            q_stoch = torch.minimum(agent.q1(obs, stoch_a),
                                    agent.q2(obs, stoch_a)).squeeze(-1)
            disagreement = ((old_action - stoch_a) ** 2).mean(-1)
            q_gap = (old_q - q_stoch).abs()

            na, nlogp = agent.actor.sample(nobs)
            tq = torch.minimum(agent.q1_target(nobs, na),
                               agent.q2_target(nobs, na)).squeeze(-1)
            target = rew + self.gamma * (1.0 - term) * (tq - agent.alpha * nlogp.squeeze(-1))
            q_sa = torch.minimum(agent.q1(obs, act), agent.q2(obs, act)).squeeze(-1)
            td = (target - q_sa).abs()

            priority = (_rank01(disagreement) + _rank01(q_gap) + _rank01(td)) / 3.0

        kept = self.memory.add_task(
            task_id,
            obs.cpu().numpy(), old_action.cpu().numpy(),
            old_q.cpu().numpy(), priority.cpu().numpy(),
        )
        return {"candidates": int(len(idx)), "kept": int(kept),
                "per_task": self.memory.counts_per_task()}

    # ------------------------------------------------------------------ losses
    def actor_loss(self, agent, batch_size, device):
        """MSE between the CURRENT (differentiable) mean action and the stored one."""
        batch = self.memory.sample(batch_size, device)
        if batch is None:
            return torch.zeros((), device=device)
        mem_obs, old_action, _, _ = batch
        new_action = agent.actor.mean_action(mem_obs)
        return F.mse_loss(new_action, old_action)

    def critic_loss(self, agent, batch_size, device):
        """Anchor both critics at (s_mem, a_old) to the stored Q value."""
        batch = self.memory.sample(batch_size, device)
        if batch is None:
            return torch.zeros((), device=device)
        mem_obs, old_action, old_q, _ = batch
        q1 = agent.q1(mem_obs, old_action).squeeze(-1)
        q2 = agent.q2(mem_obs, old_action).squeeze(-1)
        return 0.5 * (F.mse_loss(q1, old_q) + F.mse_loss(q2, old_q))

    def loss(self, agent, obs):
        """Backward-compatible API: returns (actor_loss, critic_loss)."""
        device = obs.device
        return (self.actor_loss(agent, len(obs), device),
                self.critic_loss(agent, len(obs), device))
