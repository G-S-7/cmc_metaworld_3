"""
check_expert.py
===============
Runs Meta-World's scripted expert for each task through THIS project's code
(make_env, reset_env, SafeEnv, evaluate: success at any step), on the same
fixed evaluation episodes the experiment uses.

  * Expert success > 0.8 on a task (the bar Meta-World's own test suite uses
    for these scripted policies) -> observation, goal, reset, reward and success
    handling are correct; a 0% SAC result there is a LEARNING problem.
  * Expert success near 0 -> something in the environment pipeline is wrong;
    do not train until it is fixed.

It also prints the expert's return, i.e. the return a solved episode gets, as a
reference for reading SAC's training logs, and can render videos of the expert
completing each task (this also checks the rendering pipeline).

    python check_expert.py
    python check_expert.py --episodes 30 --video
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


from cmc_rl.tasks import TASK_SEQUENCE, TASK_LABELS, make_env, env_dt
from cmc_rl.experts import get_expert
from run_experiment import evaluate, eval_episode_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--goal-mode", choices=["random", "fixed"], default="random")
    p.add_argument("--video", action="store_true", help="render 2 expert episodes per task")
    p.add_argument("--camera", default="corner2")
    p.add_argument("--out", default="videos/expert")
    a = p.parse_args()

    print(f"Scripted expert on {a.episodes} fixed evaluation episodes per task "
          f"(goal_mode={a.goal_mode})\n")
    ok = True
    for label, task in zip(TASK_LABELS, TASK_SEQUENCE):
        expert = get_expert(task)
        m = evaluate(expert, task, eval_episode_seeds(task, a.episodes), a.goal_mode)
        good = m["mean_success"] > 0.8   # Meta-World's own scripted-policy test bar
        ok &= good
        steps = m["mean_steps_to_success"]
        print(f"  {label} {task:<15} success={m['mean_success']:.2f}  "
              f"return={m['mean_return']:7.1f} ± {m['std_return']:.1f}  "
              f"steps-to-success={'n/a' if steps is None else f'{steps:.0f}'}  "
              f"unstable={m['n_unstable_episodes']}  {'OK' if good else 'PROBLEM'}")

        if a.video:
            from render_agent import render_episode, save_video
            env = make_env(task, goal_mode=a.goal_mode, render_mode="rgb_array", camera=a.camera)
            frames = []
            for i, ep_seed in enumerate(eval_episode_seeds(task, 2)):
                f, ret, t_succ = render_episode(expert, env, task, ep_seed, i, "scripted expert")
                frames += f
            out = save_video(frames, Path(a.out) / f"{label}_{task}__expert.mp4",
                             int(round(1.0 / env_dt(env))))
            env.close()
            print(f"      video: {out}")

    print("\n" + ("Environment pipeline verified: the expert solves every task through this code."
                  if ok else
                  "At least one task is NOT solved by the scripted expert: check the env setup "
                  "before training (paste this output)."))


if __name__ == "__main__":
    main()
