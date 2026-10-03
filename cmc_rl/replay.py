"""Replay buffer and CMC memory.

Changes vs. the original:
  * ReplayBuffer stores `terminals` (true environment termination ONLY). Time-limit
    truncation must NOT be stored here, otherwise the critic bootstraps to zero at
    every 500-step timeout.                                          (FIX 2)
    `dones` is kept as a read-only alias so old scripts still import.
  * ReplayBuffer.recent_indices(frac): indices of the most recent `frac` of data,
    wrap-around aware (used to pick consolidation candidates).       (FIX 4)
  * CMCMemory keeps a per-task store with equal quotas
    (capacity // n_tasks), rebalanced at every task boundary. Priorities are only
    compared WITHIN a task, so tasks with large reward/Q scales cannot evict other
    tasks. Sampling is uniform over stored states, which, with equal quotas, is
    balanced across tasks.                                           (FIX 4)
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros((capacity, act_dim), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminals = np.zeros(capacity, np.float32)
        self.size = 0
        self.ptr = 0

    @property
    def dones(self):  # backward-compat alias; these are TERMINALS, not timeouts
        return self.terminals

    def add(self, obs, action, reward, next_obs, terminal):
        """`terminal` must be float(terminated), never terminated-or-truncated."""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminals[i] = terminal
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, batch_size)
        return tuple(torch.as_tensor(x[idx], device=device)
                     for x in [self.obs, self.actions, self.rewards,
                               self.next_obs, self.terminals])

    def sample_mixed(self, batch_size, device, n_prefix, ratio):
        """Draw round(ratio * batch_size) samples from the first `n_prefix`
        transitions (the demonstration episodes, stored first) and the rest
        uniformly from the whole buffer. Requires that the buffer never wrapped,
        so the prefix is still the demonstration data."""
        if n_prefix <= 0 or ratio <= 0:
            return self.sample(batch_size, device)
        assert self.size < self.capacity or self.ptr == 0, \
            "replay buffer wrapped: demonstration transitions were overwritten"
        k = int(round(ratio * batch_size))
        idx = np.concatenate([np.random.randint(0, n_prefix, k),
                              np.random.randint(0, self.size, batch_size - k)])
        return tuple(torch.as_tensor(x[idx], device=device)
                     for x in [self.obs, self.actions, self.rewards,
                               self.next_obs, self.terminals])

    def recent_indices(self, frac):
        """Indices of the most recent ceil(frac * size) transitions (oldest first)."""
        if self.size == 0:
            return np.zeros(0, dtype=np.int64)
        k = max(1, int(np.ceil(frac * self.size)))
        k = min(k, self.size)
        # The newest item is at ptr-1; walk back k items with wrap-around.
        return (self.ptr - k + np.arange(k)) % self.capacity


class CMCMemory:
    """Small protected memory of old-task states with equal per-task quotas."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._store = {}   # task_id -> dict of arrays, sorted by priority (desc)
        self._rebuild()

    # ---------------------------------------------------------------- storage
    def add_task(self, task_id, obs, old_action, old_q, priority):
        """Store candidates for `task_id`, then rebalance all tasks to equal quotas.

        Within each task, the highest-priority states are kept. Priorities of
        different tasks are never compared.
        """
        if self.capacity <= 0 or len(obs) == 0:
            return 0
        order = np.argsort(-np.asarray(priority), kind="stable")
        self._store[int(task_id)] = {
            "obs": np.asarray(obs, np.float32)[order],
            "old_action": np.asarray(old_action, np.float32)[order],
            "old_q": np.asarray(old_q, np.float32)[order],
            "priority": np.asarray(priority, np.float32)[order],
        }
        self._rebalance()
        return self.counts_per_task().get(int(task_id), 0)

    # backward-compatible name
    def add_batch(self, obs, old_action, old_q, task_id, priority):
        return self.add_task(task_id, obs, old_action, old_q, priority)

    def _rebalance(self):
        n_tasks = len(self._store)
        quota = self.capacity // n_tasks
        spare = self.capacity - quota * n_tasks
        # Hand spare slots to the earliest tasks so total == capacity when possible.
        for k, tid in enumerate(sorted(self._store)):
            q = quota + (1 if k < spare else 0)
            d = self._store[tid]
            for key in d:
                d[key] = d[key][:q]
        self._rebuild()

    def _rebuild(self):
        tids = sorted(self._store)
        if tids:
            self.obs = np.concatenate([self._store[t]["obs"] for t in tids])
            self.old_action = np.concatenate([self._store[t]["old_action"] for t in tids])
            self.old_q = np.concatenate([self._store[t]["old_q"] for t in tids])
            self.priority = np.concatenate([self._store[t]["priority"] for t in tids])
            self.task_id = np.concatenate(
                [np.full(len(self._store[t]["obs"]), t, np.int32) for t in tids])
        else:
            self.obs = np.zeros((0, self.obs_dim), np.float32)
            self.old_action = np.zeros((0, self.act_dim), np.float32)
            self.old_q = np.zeros(0, np.float32)
            self.priority = np.zeros(0, np.float32)
            self.task_id = np.zeros(0, np.int32)
        self.size = len(self.obs)

    def counts_per_task(self):
        return {int(t): int(len(d["obs"])) for t, d in self._store.items()}

    # --------------------------------------------------------------- sampling
    def sample(self, batch_size, device):
        if self.size == 0:
            return None
        idx = np.random.randint(0, self.size, min(batch_size, self.size))
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.old_action[idx], device=device),
            torch.as_tensor(self.old_q[idx], device=device),
            torch.as_tensor(self.task_id[idx], device=device),
        )
