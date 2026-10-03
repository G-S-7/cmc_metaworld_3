"""
run_full_experiment.py
======================
The whole study, in the order the evidence requires:

  Phase 1  BASE-SAC GATE: plain SAC trained on each task ALONE.
           If SAC cannot reliably solve a task by itself (success < threshold on
           the fixed evaluation episodes), the continual experiment is NOT run:
           forgetting cannot be measured for a task that was never learned.
  Phase 2  MAIN: T1 push-v2 -> T2 push-wall-v2 -> T3 shelf-place-v2 for every
           seed, A) SAC baseline and B) SAC + CMC, identical protocol.
  Phase 3  REPORT: compare_results.py (tables + figures).
  Phase 4  VIDEOS: render_all_videos.py for the first seed.

    python run_full_experiment.py --seeds 1 2 3
    python run_full_experiment.py --seeds 1 --skip-gate        # gate already passed
    python run_full_experiment.py --gate-only                  # just check base SAC
    python run_full_experiment.py --gate-only --layer-norm --demo-episodes 10 --warmup 5000
        (unknown flags are forwarded unchanged to every run_experiment.py call)

Runtime is dominated by CPU SAC updates; the logs print steps/s, so measure on
your machine during the gate before launching the main runs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from cmc_rl.tasks import TASK_SEQUENCE, TASK_LABELS

PY = sys.executable


def run(cmd, label):
    print(f"\n{'#' * 72}\n  {label}\n  $ {' '.join(cmd)}\n{'#' * 72}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    m, s = divmod(int(time.time() - t0), 60)
    print(f"  [{label}] {'OK' if rc == 0 else f'FAILED rc={rc}'} in {m // 60}h{m % 60:02d}m{s:02d}s",
          flush=True)
    return rc == 0


def common_args(a):
    # a.extra = any run_experiment.py flags not known here (e.g. --layer-norm
    # --demo-episodes 10 --warmup 5000 --device cpu); forwarded to EVERY run so the
    # gate, the baseline and CMC always use identical settings.
    return ["--goal-mode", a.goal_mode, "--eval-episodes", str(a.eval_episodes),
            "--learned-threshold", str(a.threshold), "--output", a.output] + a.extra


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps-per-task", type=int, default=300_000)
    p.add_argument("--gate-steps", type=int, nargs=3, default=None,
                   metavar=("PUSH", "PUSH_WALL", "SHELF_PLACE"),
                   help="per-task budget for the gate (default: --steps-per-task)")
    p.add_argument("--goal-mode", choices=["random", "fixed"], default="random")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--output", default="results")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="run the main experiment even if the gate fails")
    p.add_argument("--no-videos", action="store_true")
    a, a.extra = p.parse_known_args()
    if a.extra:
        print(f"Forwarding to every run_experiment.py call: {' '.join(a.extra)}")
    t_all = time.time()

    # ── Phase 1: base-SAC gate ───────────────────────────────────────────
    if not a.skip_gate:
        gate_steps = a.gate_steps or [a.steps_per_task] * len(TASK_SEQUENCE)
        verdict = {}
        for task, label, steps in zip(TASK_SEQUENCE, TASK_LABELS, gate_steps):
            ok = run([PY, "run_experiment.py", "--no-cmc", "--tasks", task,
                      "--seed", str(a.seeds[0]), "--steps-per-task", str(steps)]
                     + common_args(a), f"GATE {label} {task}: SAC alone, {steps} steps")
            tag = a.extra[a.extra.index("--tag") + 1] if "--tag" in a.extra else ""
            summ = (Path(a.output) / (f"single_{task.replace('-v2', '')}" + (f"_{tag}" if tag else ""))
                    / f"seed_{a.seeds[0]}" / "summary.json")
            succ = json.load(open(summ))["success"]["new_task"].get(label) if ok and summ.exists() else None
            verdict[task] = succ
        print("\nBASE-SAC GATE (success on the fixed evaluation episodes):")
        passed = True
        for task, succ in verdict.items():
            good = succ is not None and succ >= a.threshold
            passed &= good
            print(f"  {task:<16} {'n/a' if succ is None else f'{succ:.2f}'}  "
                  f"{'PASS' if good else 'FAIL'}")
        json.dump(verdict, open(Path(a.output) / "gate.json", "w"), indent=2)
        if a.gate_only:
            return
        if not passed and not a.force:
            print("\nGate FAILED: base SAC does not reliably solve every task on its own.\n"
                  "Do not interpret a continual-learning run yet. Options: raise the\n"
                  "budget for the failing task (--gate-steps), check the training log\n"
                  "(train_succ, |Q|, alpha, unstable), or verify the pipeline first with\n"
                  "--goal-mode fixed. Use --force only to run anyway.")
            sys.exit(2)

    # ── Phase 2: main continual experiment ───────────────────────────────
    ok = True
    for seed in a.seeds:
        for arm, flags in (("A) SAC baseline", ["--no-cmc"]), ("B) SAC + CMC", [])):
            ok &= run([PY, "run_experiment.py", *flags, "--seed", str(seed),
                       "--steps-per-task", str(a.steps_per_task)] + common_args(a),
                      f"MAIN seed {seed}: {arm}")

    # ── Phase 3 + 4 ──────────────────────────────────────────────────────
    if ok:
        run([PY, "compare_results.py", "--results", a.output,
             "--out-dir", str(Path(a.output) / "comparison"),
             "--learned-threshold", str(a.threshold)], "REPORT")
        if not a.no_videos:
            run([PY, "render_all_videos.py", "--results", a.output,
                 "--seed", str(a.seeds[0])], "VIDEOS")
    h, rem = divmod(int(time.time() - t_all), 3600)
    print(f"\nTotal wall time {h}h{rem // 60:02d}m. Report: {Path(a.output) / 'comparison' / 'report.md'}")


if __name__ == "__main__":
    main()
