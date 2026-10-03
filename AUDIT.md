# Project audit — why the agent showed 0% success, and what was changed

Scope: every file in the project plus the Meta-World source at the commit pinned in
`requirements.txt` (c822f28) and Meta-World 3.0.0, and Gymnasium 0.29.1 / 1.0.0
(the libraries the envs inherit from). Findings marked **[source]** were verified
by reading that library code, not assumed.

## 1. Bugs that corrupted training, evaluation or the comparison

| # | Problem | Evidence | Effect | Fix |
|---|---|---|---|---|
| 1 | **Goal/object positions come from the global `np.random`**, not the env's seeded RNG. `make_env(seed=…)`, `env.seed()` and the "held-out eval seed" never controlled which goals were used. | **[source]** `SawyerXYZEnv._get_state_rand_vec` calls `np.random.uniform` (v2 and v3). | Evaluation episodes differed between runs. CMC consumes extra global random numbers (memory sampling), so **baseline and CMC were evaluated on different goals** — the comparison was not like-for-like. | `tasks.reset_env(env, episode_seed)` seeds the global RNG only during `reset()`. Evaluation uses fixed per-task episode seeds shared by every seed and both arms. |
| 2 | **CMC actor regularizer had no gradient** (`Actor.deterministic` is `@torch.no_grad`). | `cmc.py`, `networks.py` | "WITH CMC" never constrained the policy. | Differentiable `Actor.mean_action`; training asserts the anchor has a gradient. |
| 3 | **500-step timeouts stored as terminal states.** | `run_experiment.py`: `done = terminated or truncated` | Critic bootstraps to 0 on every timeout — biased Q-values. | Only `terminated` is stored. **[source]** Meta-World never sets `terminated`. |
| 4 | **Success read only at the last step.** | `evaluate()` | Undercounts episodes that succeed and then nudge the puck away. | Success = reached at any step; final-step success also logged. |
| 5 | **Physics blow-ups went into the replay buffer** (`MUJOCO_LOG.TXT`: NaN QACC at DOF 9 = puck free joint). | **[source]** Meta-World's `_did_see_sim_exception` guard is never set in this version. | Corrupt transitions, TD spikes. | `SafeEnv` drops non-finite / BADQACC / auto-reset steps and resets. |
| 6 | **Videos did not show the episode correctly.** Frames captured *before* each step and the last state never rendered; FPS 50 though the env runs at 80 Hz (62.5% speed); camera `"track"` does not exist so MuJoCo used a free camera; goals random and not the evaluated ones; nothing checked the checkpoint was loaded. | **[source]** Gymnasium asserts `1/dt == render_fps == 80`; Gymnasium falls back to camera id −1 when `"track"` is missing. | Videos could not be trusted to show the trained policy. | `render_agent.py` rewritten (see §3). |
| 7 | **Wrong library could be installed silently.** README said `pip install metaworld` (3.x → v3 tasks, mujoco≥3); `requirements.txt` pins a v2 commit with mujoco 2.3.7; `diag_benchmark.py` used v3. `tasks.py` silently switched v2→v3. | README, requirements, `diag_benchmark.py`, `tasks.py` | Possibly not running the tasks you think; version mismatch is a likely cause of the MuJoCo instability. | Pinned v2 stack; loud warning + versions saved in `config.json`. |
| 8 | Results tooling hard-coded to the old 6-task list, one seed, retention on return. | `compare_results.py`, `reeval_before_cmc.py`, `render_all_videos.py` | Could not report the intended experiment. | Rewritten for T1→T2→T3, success-primary, multi-seed, paired arms, protocol check. |
| 9 | Priority score mixed units and its "TD error" bootstrapped from `s` not `s'`; one task with large Q values could evict the others from memory. | `cmc.py`, `replay.py` | Memory dominated by reward scale. | Rank-normalized priority with a real TD error; equal per-task memory quotas. |

## 2. Why success was 0 even without bugs

* **Budget.** The default was 100k steps per task with randomized goals. SAC usually
  needs several hundred thousand steps for `push`, and `shelf-place` (grasp + lift +
  place) is one of the hardest Meta-World tasks.
* **Reward ≈ 0 early is expected.** **[source]** `push`/`push-wall` rewards are dominated
  by a gripper "caging" term with 5 mm tolerances until the gripper is at the puck;
  `shelf-place` reward is *grasp × placement* (a product), so it stays near 0 until
  grasping is learned. A return near 0 means "has not reached the puck yet", not a bug.
* **Success thresholds** **[source]**: push 0.05 m, push-wall 0.07 m, shelf-place 0.07 m
  (puck to target). Episodes never terminate on success; they run 500 steps.

**Consequence:** the pipeline now has a **base-SAC gate** — each task is first trained
alone; the continual experiment is not run unless plain SAC reaches the success
threshold (default 0.8) on every task. Without that, "forgetting" is undefined.

## 3. Verified correct (no change needed)

* Observation: 39-d goal-observable (hand, gripper, puck pose, previous frame, goal).
* Action: 4-d in [-1, 1]; xyz × 0.01 m per step + gripper (positive = close); the actor's
  tanh scaling matches the env bounds.
* SAC: twin critics, target networks, entropy tuning, tanh log-prob correction.
* Checkpoint save/load: all networks + α. Rendering now verifies the loaded weights.
* Episode length 500 with truncation; reset re-samples goals when unfrozen.

## 4. Files

**Changed/added:** `cmc_rl/tasks.py` (3 tasks, reproducible resets, camera),
`cmc_rl/metrics.py` (new), `cmc_rl/networks.py`, `cmc_rl/cmc.py`, `cmc_rl/replay.py`,
`run_experiment.py`, `compare_results.py`, `render_agent.py`, `render_all_videos.py`,
`run_full_experiment.py`, `sanity_check.py`, `requirements.txt`, `README.md`.

**Obsolete — delete:** `train_cmc.py`, `Untitled.ipynb`, `evaluate_results.py`
(10/20-task legacy code), `reeval_before_cmc.py` (6 tasks, wrong seeding),
`diag_benchmark.py` (v3 API; `run_experiment.py` now logs steps/s), `final_report.md`
(several statements in it are wrong: 50 Hz, "SAC issues: none", 6 tasks).

## 5. Remaining limitations (be explicit in any write-up)

* No task identity in the observation; the three tasks share one policy. This is part
  of the setup being studied, but it limits how much any method can retain.
* CMC's importance score is a heuristic, not a counterfactual comparison with a frozen
  old policy. Run `--memory-selection uniform` as the control.
* Fewer than 3 seeds per arm does not support a claim that CMC reduces forgetting.

## 6. First base-SAC check (seed 1, 300k steps per task, plain SAC, random goals)

| Task | Eval success | What the log shows | Cause |
|---|---|---|---|
| push-v2 | 0.10 | \|Q\| climbed to ~480 and TD error to ~4000 while episode returns were only 20–140; α jumped from 0.01 to 0.38; learning then collapsed and slowly recovered | **critic divergence** (Q-values far above any achievable return) |
| push-wall-v2 | 0.00 | return rose steadily 25 → ~470, Q stable | **still learning** — reaches and touches the puck, not yet pushing it past the wall; budget too small |
| shelf-place-v2 | 0.00 | reward exactly 0.0 for all 300k steps; Q decayed to 0; α → 0 | **no reward signal** (see below) |

**Why shelf-place gives reward = 0** **[source]**: its reward is
`hamacher_product(object_grasped, in_place)`, and `object_grasped` is exactly 0 unless
the gripper "caging" score exceeds 0.97 *and* the gripper is closing. A hamacher product
with a 0 term is 0. So the task pays nothing until the puck is almost perfectly between
the fingers. The random-policy check measured 0.000, and SAC saw no reward in 300k steps.
This is a property of the shelf-place-v2 reward, not a bug in this project.

**Changes:** optional critic LayerNorm (`--layer-norm`), optional scripted-expert
demonstration episodes at the start of each task (`--demo-episodes`), optional reward
scaling, `--device`, fewer GPU synchronisations per step, and `check_expert.py`, which
runs Meta-World's scripted expert through this project's code to verify the environment
pipeline (Meta-World's own tests require > 0.8 expert success).

## 7. Second base-SAC check (seed 1, 300k steps, `--layer-norm --demo-episodes 10 --warmup 5000`)

| Task | Final eval (30 ep.) | Best learning-curve point (10 ep.) | Run 1 |
|---|---|---|---|
| push-v2 | 0.57 | 0.90 at 270k | 0.10 |
| push-wall-v2 | 0.23 | 0.80 at 150k and 200k, then fell to 0.10–0.30 | 0.00 |
| shelf-place-v2 | 0.07 | 0.30 at 220k–270k | 0.00 |

* The expert demos succeeded in 10/10 episodes on every task (with action noise), which
  confirms the environment, goal, reward and success pipeline end to end.
* Q-values now track the returns (mean Q ≈ 300–700 with ≈ 5–6 reward per step, γ = 0.99)
  instead of diverging, so LayerNorm fixed the run-1 blow-up.
* All three tasks are learned, but not to 0.8 within 300k steps, and the policy is unstable
  late in training (push-wall peaked and then lost most of its success). The demo
  transitions shrink to under 2% of the data by the end; `--demo-ratio` keeps a fixed share.
