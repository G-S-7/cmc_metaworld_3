"""
reeval_before_cmc.py
====================
Re-runs evaluation only (no training) for the BEFORE CMC experiment,
loading saved checkpoints from results/before_cmc/checkpoints/.

This is needed when the training was complete but the evaluation loop was
interrupted (e.g. server restart).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import numpy as np
import torch

from cmc_rl.tasks import SIX_TASK_SEQUENCE, make_env
from cmc_rl.networks import SACAgent

TASK_LABELS = ["T1", "T2", "T3", "T4", "T5", "T6"]
TASKS       = SIX_TASK_SEQUENCE
CKPT_DIR    = Path("results/before_cmc/checkpoints")
OUT_DIR     = Path("results/before_cmc")
EVAL_EPISODES = 10
SEED = 1
DEVICE = torch.device("cpu")


@torch.no_grad()
def evaluate(agent, task_name, episodes, seed=None):
    env = make_env(task_name, seed=seed)
    returns, successes = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        info = {}
        for _ in range(500):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_ret)
        successes.append(float(info.get("success", 0.0)))
    env.close()
    return {
        "mean_return":  float(np.mean(returns)),
        "std_return":   float(np.std(returns)),
        "mean_success": float(np.mean(successes)),
        "std_success":  float(np.std(successes)),
        "n_episodes":   episodes,
    }


def probe_dims():
    env = make_env(TASKS[0], seed=SEED)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low  = env.action_space.low.copy()
    high = env.action_space.high.copy()
    env.close()
    return obs_dim, act_dim, low, high


def main():
    print("=" * 60)
    print("  RE-EVALUATION: BEFORE CMC (no training)")
    print("=" * 60)

    obs_dim, act_dim, low, high = probe_dims()
    all_metrics = {}
    results_log = []

    for stage_idx, (task_label, task_name) in enumerate(zip(TASK_LABELS, TASKS)):
        ckpt_path = CKPT_DIR / f"stage_{task_label}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n  Loading checkpoint after Task {task_label}: {ckpt_path.name}")
        agent = SACAgent(obs_dim, act_dim, low, high, DEVICE)
        agent.load(str(ckpt_path))
        agent.actor.eval()
        agent.q1.eval()
        agent.q2.eval()

        print(f"  --- Evaluation after Task {task_label} ---")
        for eval_label in TASK_LABELS[: stage_idx + 1]:
            eval_task = TASKS[TASK_LABELS.index(eval_label)]
            t0 = time.time()
            m = evaluate(agent, eval_task, EVAL_EPISODES, seed=SEED)
            elapsed = time.time() - t0
            all_metrics[(task_label, eval_label)] = m
            print(f"    Task {eval_label} ({eval_task}): "
                  f"return={m['mean_return']:.2f} ± {m['std_return']:.2f}  "
                  f"success={m['mean_success']:.3f}  [{elapsed:.1f}s]")
            results_log.append({
                "trained_task_label": task_label,
                "trained_task_name":  task_name,
                "eval_task_label":    eval_label,
                "eval_task_name":     eval_task,
                "use_cmc":            False,
                "seed":               SEED,
                **m,
            })

        # Save incrementally
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(results_log, f, indent=2)

    # Forgetting summary
    print(f"\n{'='*60}")
    print("  FORGETTING SUMMARY  (BEFORE CMC)")
    print(f"{'='*60}")

    forgetting = {}
    for i, t_label in enumerate(TASK_LABELS):
        base_key = (t_label, t_label)
        if base_key not in all_metrics:
            continue
        base_perf = all_metrics[base_key]["mean_return"]
        base_succ = all_metrics[base_key]["mean_success"]
        print(f"  Task {t_label} ({TASKS[i]})")
        print(f"    Baseline: return={base_perf:.2f}, success={base_succ:.3f}")
        for j in range(i + 1, len(TASK_LABELS)):
            eval_label = TASK_LABELS[j]
            curr_key = (eval_label, t_label)
            if curr_key not in all_metrics:
                continue
            curr_perf = all_metrics[curr_key]["mean_return"]
            curr_succ = all_metrics[curr_key]["mean_success"]
            retention = (curr_perf / base_perf) if base_perf > 0 else float("nan")
            forgetting[f"Retention_{t_label}_after_{eval_label}"] = retention
            print(f"    After {eval_label}: return={curr_perf:.2f}, "
                  f"success={curr_succ:.3f}, retention={retention:.1%}")

    with open(OUT_DIR / "forgetting.json", "w") as f:
        json.dump(forgetting, f, indent=2)

    print(f"\n  Saved: {OUT_DIR / 'results.json'}")
    print(f"  Saved: {OUT_DIR / 'forgetting.json'}")
    print("\n  Done. Now run: python compare_results.py "
          "--before results/before_cmc/results.json "
          "--after results/after_cmc/results.json")


if __name__ == "__main__":
    main()
