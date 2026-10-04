# CMC for continual Meta-World RL — 3-task study

Does a SAC agent keep what it learned on earlier manipulation tasks while it learns
related new ones, and does Counterfactual Memory Consolidation (CMC) reduce forgetting?

| Stage | Task | What it adds |
|---|---|---|
| T1 | `push-v2` | push the puck to a target (manipulation) |
| T2 | `push-wall-v2` | navigate around a wall, then push (manipulation + navigation) |
| T3 | `shelf-place-v2` | grasp, lift and place the puck on a shelf (pick + place) |

Two arms, identical protocol: **A) SAC baseline** and **B) SAC + CMC**.
Primary metric: **success rate**; secondary: return. See `AUDIT.md` for what was
wrong before and why success was 0.

## Install

```bash
pip install -r requirements.txt     # pinned Meta-World v2 stack (mujoco 2.3.7)
```
Headless Linux rendering: `export MUJOCO_GL=egl` (or `osmesa`).

## 1. Check the pipeline (minutes)

```bash
python sanity_check.py                  # logic checks on a fake env
python sanity_check.py --env --render   # real tasks: versions, reproducible goals, one frame each
python check_expert.py --video          # Meta-World's scripted expert through this code:
                                        # success should be > 0.8 per task; also writes
                                        # videos/expert/*.mp4 showing each task completed
python run_experiment.py --smoke-test --no-cmc
python run_experiment.py --smoke-test
```

## 2. Base-SAC gate — do this before the continual experiment

Each task trained **alone** with plain SAC. Continue only if every task reaches the
success threshold (default 0.8) on the fixed evaluation episodes.

```bash
python run_full_experiment.py --gate-only --seeds 1 --layer-norm --demo-episodes 10 --warmup 5000
# or one task at a time:
python run_experiment.py --no-cmc --tasks push-v2 --layer-norm --demo-episodes 10 --warmup 5000
```
Any flag `run_full_experiment.py` does not know is forwarded to every run, so the
gate, the baseline and CMC always share identical settings.

| Option | What it does | Why |
|---|---|---|
| `--layer-norm` | LayerNorm in both critics | plain SAC's Q-values blew up on push-v2 (mean Q ≈ 480, TD error ≈ 4000) |
| `--demo-episodes N` | first N episodes of each task use Meta-World's scripted expert (+ `--demo-noise`, default 0.1) | shelf-place-v2 reward is exactly 0 until the puck is caged between the fingers; random exploration never gets there |
| `--demo-ratio r` | draw a fraction r of every batch from the demo transitions (e.g. 0.25) | without it the 5k demo steps shrink to under 2% of the data by 300k steps |
| `--reward-scale x` | scale stored rewards (training only) | optional, if Q-values still grow large |
| `--device cpu/cuda` | choose the device | small networks can run faster on CPU |

These change the training setup, so report them; they apply identically to both arms,
and the replay buffer is still reset per task (no old-task data is replayed).
The log prints `train_succ`, `|Q|`, `alpha`, unstable steps and **steps/s** — use
steps/s to estimate the runtime of the main experiment on your machine.
Tip: `--goal-mode fixed` (one configuration per task) is a fast way to confirm the
pipeline can learn at all before the harder randomized-goal setting.

## 3. Main experiment

```bash
python run_full_experiment.py --seeds 1 2 3 --skip-gate     # after the gate passed
```
or by hand, per seed:
```bash
python run_experiment.py --no-cmc --seed 1    # A) results/before_cmc/seed_1/
python run_experiment.py          --seed 1    # B) results/after_cmc/seed_1/
python compare_results.py --results results --out-dir results/comparison
python render_all_videos.py --results results --seed 1
```

After every stage, **all tasks learned so far** are evaluated on the same 30 fixed
episodes per task (identical for both arms and all seeds):

|          | T1 | T2 | T3 |
|----------|----|----|----|
| after T1 | new learning | – | – |
| after T2 | retention / forgetting | new learning | – |
| after T3 | retention / forgetting | retention / forgetting | new learning |

Forgetting of task j after stage i = success(j after j) − success(j after i).

## Outputs

```
results/
  single_<task>/seed_1/          base-SAC gate runs
  before_cmc/seed_k/  after_cmc/seed_k/
      config.json                arguments, library versions, evaluation episode seeds
      results.json               every (stage, task) evaluation
      learning_curves.json       success of all seen tasks during training
      summary.json               new-task learning, retention, forgetting
      checkpoints/stage_T1.pt …  agent after each stage
  comparison/
      report.md  metrics.json  matrix_success.png  stage_success.png
      forgetting.png  learning_curves.png
videos/<results folder>/
  <run>/seed_k/T1_push-v2__after_T2.mp4      every task learned so far, after every stage
  compare/seed_k/T1_push-v2__after_T3__baseline_vs_cmc.mp4
                                             baseline (left) vs CMC (right), same start state
  index.md                                   every video with its success count
```
Each video has a `.json` with per-episode success (`render_agent.py` also saves start/end PNGs).
Videos are 80 fps (Meta-World's control rate), from the `corner2` camera, on the
same episodes used for evaluation, and the checkpoint weights are verified on load.

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | train + evaluate one arm (or one task with `--tasks`) |
| `run_full_experiment.py` | gate → main runs → report → videos |
| `compare_results.py` | tables and figures, baseline vs CMC |
| `render_agent.py`, `render_all_videos.py` | videos of the trained policies |
| `sanity_check.py` | pipeline checks |
| `check_expert.py` | scripted expert through this code: verifies env, success and videos |
| `cmc_rl/tasks.py` | the 3 tasks, reproducible resets, camera |
| `cmc_rl/networks.py`, `replay.py`, `cmc.py`, `metrics.py` | SAC, buffers, CMC, metrics |
| `cmc_rl/experts.py` | Meta-World's scripted expert policies for the 3 tasks |
