"""
render_all_videos.py
====================
Videos of the trained policies for the continual-learning experiment.

For every run folder under --results (before_cmc, after_cmc, and single-task
gate runs single_*), and every seed, it renders checkpoint x task videos on the
SAME fixed evaluation episodes the experiment scores:

  --mode full (default)   every task learned so far, after every stage
                          (the same grid as the evaluation matrix):
                            after T1: T1
                            after T2: T1, T2
                            after T3: T1, T2, T3
  --mode key              each task right after it was learned + after the last stage

Side-by-side videos (left: SAC baseline, right: SAC + CMC) are made for every
(checkpoint, task) pair present in both arms of a seed. Both robots start from
the identical puck/goal position, so differences are due to the policy only.

Every video shows the start state, every action, and an overlay with step,
reward, return and the step at which success was first reached. Checkpoint
weights are verified on load (see render_agent.load_verified_agent).

Outputs (under --out/<results folder name>/):
    <run>/seed_k/T1_push-v2__after_T3.mp4      + .json (per-episode success)
    compare/seed_k/T1_push-v2__after_T3__baseline_vs_cmc.mp4
    index.md, index.json                       what was rendered, with success

    python render_all_videos.py --results results_main --seed 1
    python render_all_videos.py --results results_main --seeds 1 2 3 --mode key
    python render_all_videos.py --results results --runs single_push_dr25
Headless Linux: set MUJOCO_GL=egl (or osmesa). Windows needs nothing.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from cmc_rl.tasks import TASK_SEQUENCE, TASK_LABELS, MAX_EPISODE_STEPS, make_env, reset_env, env_dt
from render_agent import CAMERAS, load_verified_agent, _overlay, _sha256
from run_experiment import eval_episode_seeds

ARMS = {"before_cmc": "SAC baseline", "after_cmc": "SAC + CMC"}


def task_label(task):
    return TASK_LABELS[TASK_SEQUENCE.index(task)]


def label_task(label):
    return TASK_SEQUENCE[TASK_LABELS.index(label)]


def run_title(run_name):
    for prefix, title in ARMS.items():
        if run_name == prefix:
            return title
        if run_name.startswith(prefix + "_"):
            return f"{title} [{run_name[len(prefix) + 1:]}]"
    return f"SAC alone [{run_name}]" if run_name.startswith("single_") else run_name


# ───────────────────────────────────────────────────────────────── discovery
def discover(results, seeds=None, only_runs=None):
    """Find <results>/<run>/seed_k/ folders that contain stage checkpoints."""
    runs = []
    for run_dir in sorted(p for p in Path(results).iterdir() if p.is_dir()):
        if run_dir.name == "comparison" or (only_runs and run_dir.name not in only_runs):
            continue
        for seed_dir in sorted(run_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            if seeds and seed not in seeds:
                continue
            ckpts = sorted((seed_dir / "checkpoints").glob("stage_T*.pt"))
            if not ckpts:
                continue
            cfg_path = seed_dir / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            args = cfg.get("args", {})
            runs.append({
                "name": run_dir.name, "seed": seed, "dir": seed_dir,
                "tasks": args.get("tasks") or list(TASK_SEQUENCE),
                "goal_mode": args.get("goal_mode", "random"),
                "stages": {p.stem.split("_", 1)[1]: p for p in ckpts},
            })
    return runs


def plan(run, mode):
    """(stage, task_label) pairs to render for one run."""
    labels = [task_label(t) for t in run["tasks"]]
    stages = [lbl for lbl in labels if lbl in run["stages"]]
    if not stages:
        return []
    final = stages[-1]
    jobs = []
    for st in stages:
        for tl in labels[: labels.index(st) + 1]:
            if mode == "full" or tl == st or st == final:
                jobs.append((st, tl))
    return jobs


def pairs(runs):
    """Match before_cmc[_x] with after_cmc[_x] runs of the same seed."""
    by_key = {(r["name"], r["seed"]): r for r in runs}
    out = []
    for r in runs:
        if r["name"] == "before_cmc" or r["name"].startswith("before_cmc_"):
            other = by_key.get(("after_cmc" + r["name"][len("before_cmc"):], r["seed"]))
            if other is not None:
                out.append((r, other))
    return out


# ───────────────────────────────────────────────────────────────── rendering
class EpisodeRunner:
    """Steps one policy through one episode, producing an annotated frame after
    reset (start state) and after every env.step (state the action produced)."""

    def __init__(self, agent, env, task, title):
        self.agent, self.env, self.task, self.title = agent, env, task, title

    def start(self, episode_seed, ep_idx):
        self.obs, _ = reset_env(self.env, episode_seed)
        self.seed, self.ep_idx = episode_seed, ep_idx
        self.t, self.ret, self.first_success, self.done = 0, 0.0, -1, False
        self.frame = self._annotate(self._grab(), 0.0)
        if np.std(self.frame) < 1.0:
            raise RuntimeError("rendered frame is blank: check MUJOCO_GL / camera")
        return self.frame

    def step(self):
        """Advance one step; once the episode is over, keep returning the last frame."""
        if self.done:
            return self.frame
        self.obs, r, terminated, truncated, info = self.env.step(
            self.agent.act(self.obs, deterministic=True))
        self.t += 1
        self.ret += float(r)
        if info.get("success", 0.0) > 0 and self.first_success < 0:
            self.first_success = self.t
        self.done = bool(terminated or truncated or self.t >= MAX_EPISODE_STEPS)
        self.frame = self._annotate(self._grab(), float(r))
        return self.frame

    def stats(self):
        return {"episode_seed": self.seed, "return": round(self.ret, 2),
                "success": self.first_success > 0, "first_success_step": self.first_success,
                "steps": self.t}

    def _grab(self):
        f = self.env.render()
        if f is None:
            raise RuntimeError("env.render() returned None (render_mode not set?)")
        return np.ascontiguousarray(f, dtype=np.uint8)

    def _annotate(self, frame, r):
        if self.first_success > 0:
            status = f"SUCCESS at step {self.first_success}"
        elif self.t == 0:
            status = "start state"
        else:
            status = "no success yet" if not self.done else "episode over: no success"
        return _overlay(frame, [f"{self.title} | {self.task} | episode {self.ep_idx + 1}",
                                f"step {self.t:3d}/{MAX_EPISODE_STEPS}  reward {r:5.2f}  "
                                f"return {self.ret:7.1f}  {status}"])


def open_writer(path, fps):
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                              macro_block_size=1)


def _side_by_side(left, right, gap=6):
    h = max(left.shape[0], right.shape[0])
    pad = lambda f: np.pad(f, ((0, h - f.shape[0]), (0, 0), (0, 0)))
    sep = np.full((h, gap, 3), 255, np.uint8)
    return np.concatenate([pad(left), sep, pad(right)], axis=1)


def render_single(run, stage, tl, a, root):
    task = label_task(tl)
    ckpt = run["stages"][stage]
    agent = load_verified_agent(str(ckpt), task, run["goal_mode"])
    env = make_env(task, goal_mode=run["goal_mode"], render_mode="rgb_array", camera=a.camera)
    fps = a.fps or int(round(1.0 / env_dt(env)))
    out = root / run["name"] / f"seed_{run['seed']}" / f"{tl}_{task}__after_{stage}.mp4"
    runner = EpisodeRunner(agent, env, task, f"{run_title(run['name'])} | after {stage}")
    episodes = []
    try:
        with open_writer(out, fps) as w:
            for i, ep_seed in enumerate(eval_episode_seeds(task, a.episodes)):
                w.append_data(runner.start(ep_seed, i))
                while not runner.done:
                    w.append_data(runner.step())
                episodes.append(runner.stats())
    finally:
        env.close()
    out.with_suffix(".json").write_text(json.dumps(
        {"run": run["name"], "seed": run["seed"], "checkpoint": str(ckpt),
         "sha256": _sha256(ckpt), "task": task, "after_stage": stage,
         "goal_mode": run["goal_mode"], "camera": a.camera, "fps": fps,
         "episodes": episodes}, indent=2))
    return out, episodes


def render_compare(base, cmc, stage, tl, a, root):
    task = label_task(tl)
    goal_mode = base["goal_mode"]
    if cmc["goal_mode"] != goal_mode:
        raise ValueError("baseline and CMC runs use different goal modes")
    runners = []
    envs = []
    for run in (base, cmc):
        agent = load_verified_agent(str(run["stages"][stage]), task, goal_mode)
        env = make_env(task, goal_mode=goal_mode, render_mode="rgb_array", camera=a.camera)
        envs.append(env)
        runners.append(EpisodeRunner(agent, env, task,
                                     f"{run_title(run['name'])} | after {stage}"))
    fps = a.fps or int(round(1.0 / env_dt(envs[0])))
    out = root / "compare" / f"seed_{base['seed']}" / \
        f"{tl}_{task}__after_{stage}__baseline_vs_cmc.mp4"
    episodes = {"baseline": [], "cmc": []}
    try:
        with open_writer(out, fps) as w:
            for i, ep_seed in enumerate(eval_episode_seeds(task, a.episodes)):
                w.append_data(_side_by_side(runners[0].start(ep_seed, i),
                                            runners[1].start(ep_seed, i)))
                while not (runners[0].done and runners[1].done):
                    w.append_data(_side_by_side(runners[0].step(), runners[1].step()))
                episodes["baseline"].append(runners[0].stats())
                episodes["cmc"].append(runners[1].stats())
    finally:
        for env in envs:
            env.close()
    out.with_suffix(".json").write_text(json.dumps(
        {"seed": base["seed"], "task": task, "after_stage": stage, "left": base["name"],
         "right": cmc["name"], "goal_mode": goal_mode, "camera": a.camera, "fps": fps,
         "episodes": episodes}, indent=2))
    return out, episodes


def _succ(eps):
    return f"{sum(e['success'] for e in eps)}/{len(eps)}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="folder holding the run folders")
    p.add_argument("--seed", type=int, default=None, help="one seed (same as --seeds k)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all found")
    p.add_argument("--runs", nargs="+", default=None,
                   help="only these run folders, e.g. before_cmc after_cmc single_push_dr25")
    p.add_argument("--mode", choices=["full", "key"], default="full")
    p.add_argument("--what", choices=["both", "individual", "compare"], default="both")
    p.add_argument("--episodes", type=int, default=2,
                   help="evaluation episodes per video (the first N of the fixed 30)")
    p.add_argument("--camera", default="corner2", choices=CAMERAS)
    p.add_argument("--fps", type=int, default=None, help="default: 1/env.dt = 80")
    p.add_argument("--out", default="videos")
    a = p.parse_args()

    seeds = a.seeds or ([a.seed] if a.seed is not None else None)
    if not Path(a.results).is_dir():
        sys.exit(f"results folder not found: {a.results}")
    runs = discover(a.results, seeds, a.runs)
    if not runs:
        sys.exit(f"no checkpoints found under {a.results} "
                 f"(expected <run>/seed_k/checkpoints/stage_T*.pt)")
    root = Path(a.out) / Path(a.results).resolve().name

    todo = []
    if a.what in ("both", "individual"):
        todo += [("single", run, None, st, tl) for run in runs for st, tl in plan(run, a.mode)]
    if a.what in ("both", "compare"):
        for base, cmc in pairs(runs):
            common = [j for j in plan(base, a.mode) if j in plan(cmc, a.mode)]
            todo += [("compare", base, cmc, st, tl) for st, tl in common]

    print(f"Found {len(runs)} run(s): " + ", ".join(f"{r['name']}/seed_{r['seed']}" for r in runs))
    print(f"Rendering {len(todo)} video(s), {a.episodes} episode(s) each, camera {a.camera}, "
          f"into {root}\n")
    rows, failures = [], []
    t0 = time.time()
    for k, (kind, run, other, st, tl) in enumerate(todo, 1):
        label = (f"{run['name']}/seed_{run['seed']}" if kind == "single"
                 else f"compare/seed_{run['seed']}")
        print(f"[{k}/{len(todo)}] {label}: {tl} {label_task(tl)} after {st}", flush=True)
        try:
            if kind == "single":
                out, eps = render_single(run, st, tl, a, root)
                summary = f"success {_succ(eps)}"
                row = {"kind": kind, "run": run["name"], "success": _succ(eps)}
            else:
                out, eps = render_compare(run, other, st, tl, a, root)
                summary = f"baseline {_succ(eps['baseline'])} | CMC {_succ(eps['cmc'])}"
                row = {"kind": kind, "run": f"{run['name']} vs {other['name']}",
                       "success": f"{_succ(eps['baseline'])} vs {_succ(eps['cmc'])}"}
            row.update({"seed": run["seed"], "task": f"{tl} {label_task(tl)}",
                        "after": st, "video": str(out.relative_to(root))})
            rows.append(row)
            print(f"    {summary}  ->  {out}", flush=True)
        except Exception as e:  # noqa: BLE001 — keep going; report at the end
            failures.append((label, tl, st, repr(e)))
            print(f"    FAILED: {e!r}", flush=True)
            traceback.print_exc(limit=2)

    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(rows, indent=2))
    md = ["# Rendered videos", "",
          f"Results: `{a.results}` · camera `{a.camera}` · {a.episodes} fixed evaluation "
          "episode(s) per video · success = episodes in which the task was completed", "",
          "| kind | run | seed | task | checkpoint | success | video |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['kind']} | {r['run']} | {r['seed']} | {r['task']} | after {r['after']} | "
           f"{r['success']} | `{r['video']}` |" for r in rows]
    (root / "index.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    m, s = divmod(int(time.time() - t0), 60)
    print(f"\nDone: {len(rows)} video(s) in {m}m{s:02d}s. Index: {root / 'index.md'}")
    if failures:
        print(f"{len(failures)} video(s) FAILED:")
        for f in failures:
            print("   ", *f)
        sys.exit(1)


if __name__ == "__main__":
    main()