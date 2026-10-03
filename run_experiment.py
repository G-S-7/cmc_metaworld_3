"""
Continual SAC (+ optional CMC) on Meta-World:  T1 push-v2 -> T2 push-wall-v2 -> T3 shelf-place-v2
==============================================================================================

Protocol (identical for the baseline and CMC runs)
--------------------------------------------------
* Tasks are trained in order; one shared SAC agent; replay buffer reset per task.
* After each stage, EVERY task learned so far is evaluated on the same fixed set
  of evaluation episodes (object/goal positions come from per-task episode seeds
  that do not depend on the run seed or on CMC). Primary metric: success rate
  (success reached at any step of the 500-step episode). Secondary: return.
* Learning curves: every --eval-every steps, all tasks seen so far are evaluated
  on the first --curve-episodes of those episodes. This shows new-task learning
  and old-task forgetting while it happens.

Examples
--------
    # 1) Base-SAC gate: can plain SAC learn each task on its own?
    python run_experiment.py --no-cmc --tasks push-v2        --steps-per-task 300000
    python run_experiment.py --no-cmc --tasks push-wall-v2   --steps-per-task 300000
    python run_experiment.py --no-cmc --tasks shelf-place-v2 --steps-per-task 500000

    # 1b) Same, with stabilised critics and 10 scripted-expert episodes per task
    python run_experiment.py --no-cmc --tasks push-v2 --layer-norm --demo-episodes 10 --warmup 5000

    # 2) Main experiment (same seed for both arms)
    python run_experiment.py --no-cmc --seed 1      # A) SAC baseline
    python run_experiment.py          --seed 1      # B) SAC + CMC
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cmc_rl.tasks import (TASK_SEQUENCE, TASK_LABELS, MAX_EPISODE_STEPS,
                          make_env, reset_env, get_env_dict, resolve_name, env_dt)
from cmc_rl.networks import SACAgent
from cmc_rl.replay import ReplayBuffer, CMCMemory
from cmc_rl.cmc import CounterfactualMemoryConsolidator
from cmc_rl.metrics import build_matrix, cl_metrics, fmt
from cmc_rl.experts import get_expert

EVAL_SEED_BASE = 9_000_000   # evaluation episodes: identical for every run


# ════════════════════════════════════════════════════════════════════ utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def label_of(task_name):
    return TASK_LABELS[TASK_SEQUENCE.index(task_name)]


def eval_episode_seeds(task_name, n):
    """Fixed per-task evaluation episodes, shared by every run and both arms."""
    base = EVAL_SEED_BASE + 10_000 * TASK_SEQUENCE.index(task_name)
    return [base + k for k in range(n)]


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


def environment_report(goal_mode):
    rep = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "mujoco": _pkg_version("mujoco"),
        "metaworld": _pkg_version("metaworld"), "gymnasium": _pkg_version("gymnasium"),
    }
    _, api = get_env_dict()
    rep["metaworld_api"] = api
    rep["envs"] = {}
    for t in TASK_SEQUENCE:
        env = make_env(t, goal_mode=goal_mode)
        u = getattr(env, "unwrapped", env)
        rep["envs"][t] = {"resolved": resolve_name(t), "class": type(u).__name__,
                          "dt": env_dt(env),
                          "max_path_length": int(getattr(u, "max_path_length", -1)),
                          "obs_dim": int(np.prod(env.observation_space.shape)),
                          "act_dim": int(np.prod(env.action_space.shape))}
        env.close()
    warnings = []
    mj = rep["mujoco"].split(".")[0]
    if api == "v2" and mj.isdigit() and int(mj) >= 3:
        warnings.append("Meta-World v2 envs expect mujoco<3; install mujoco==2.3.7.")
    if api == "v3" and mj.isdigit() and int(mj) < 3:
        warnings.append("Meta-World v3 envs expect mujoco>=3.")
    if api != "v2":
        warnings.append("Not running the v2 tasks the experiment specifies.")
    for t, e in rep["envs"].items():
        if e["max_path_length"] != MAX_EPISODE_STEPS:
            warnings.append(f"{t}: max_path_length={e['max_path_length']}")
        if (e["obs_dim"], e["act_dim"]) != (39, 4):
            warnings.append(f"{t}: obs/act dims {e['obs_dim']}/{e['act_dim']} != 39/4")
    rep["warnings"] = warnings

    print("Environment report:")
    for k in ("python", "torch", "mujoco", "metaworld", "gymnasium", "metaworld_api"):
        print(f"  {k:<14} {rep[k]}")
    for t, e in rep["envs"].items():
        print(f"  {t:<16} -> {e['resolved']}  dt={e['dt']:.4f}s ({1 / e['dt']:.0f} Hz)  "
              f"T={e['max_path_length']}  obs={e['obs_dim']} act={e['act_dim']}")
    for w in warnings:
        print(f"  [WARNING] {w}")
    return rep


# ═══════════════════════════════════════════════════════════ physics guard
def _mj_data(env):
    return getattr(getattr(env, "unwrapped", env), "data", None)


def _bad_qacc_count(data):
    try:
        import mujoco
        return int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    except Exception:
        return 0


class SafeEnv:
    """Flags unstable steps: non-finite obs/reward, a MuJoCo BADQACC warning, or
    simulator time going backwards (MuJoCo auto-reset after a blow-up)."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.n_bad = 0
        self._prev_cnt, self._prev_time = 0, 0.0

    def _snapshot(self):
        data = _mj_data(self.env)
        if data is not None:
            self._prev_cnt, self._prev_time = _bad_qacc_count(data), float(data.time)

    def reset(self, episode_seed):
        out = reset_env(self.env, episode_seed)
        self._snapshot()
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bad = (not np.all(np.isfinite(obs))) or (not np.isfinite(reward))
        data = _mj_data(self.env)
        if data is not None:
            cnt, tnow = _bad_qacc_count(data), float(data.time)
            if cnt != self._prev_cnt or tnow < self._prev_time:
                bad = True
            self._prev_cnt, self._prev_time = cnt, tnow
        if bad:
            self.n_bad += 1
        return obs, reward, terminated, truncated, info, bad

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════════ evaluation
@torch.no_grad()
def evaluate(agent, task_name, episode_seeds, goal_mode, env_fn=None):
    """Deterministic policy; success = success reached at ANY step."""
    env_fn = env_fn or make_env
    env = SafeEnv(env_fn(task_name, goal_mode=goal_mode))
    returns, successes, final_succ, first_succ_step, unstable = [], [], [], [], 0
    for ep_seed in episode_seeds:
        obs, _ = env.reset(ep_seed)
        ep_ret, ep_succ, last, t_succ = 0.0, 0.0, 0.0, -1
        for t in range(MAX_EPISODE_STEPS):
            obs, reward, terminated, truncated, info, bad = env.step(
                agent.act(obs, deterministic=True))
            if bad:
                unstable += 1
                break
            ep_ret += float(reward)
            last = float(info.get("success", 0.0))
            if last > 0 and t_succ < 0:
                t_succ = t + 1
            ep_succ = max(ep_succ, last)
            if terminated or truncated:
                break
        returns.append(ep_ret)
        successes.append(ep_succ)
        final_succ.append(last)
        first_succ_step.append(t_succ)
    env.close()
    solved = [s for s in first_succ_step if s > 0]
    return {
        "mean_success": float(np.mean(successes)),
        "std_success": float(np.std(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_final_step_success": float(np.mean(final_succ)),
        "mean_steps_to_success": float(np.mean(solved)) if solved else None,
        "n_episodes": len(episode_seeds),
        "n_unstable_episodes": unstable,
    }


# ═════════════════════════════════════════════════════════════════ training
def train_one_task(agent, task_name, stage_tasks, stage_idx, args, replay, cmc,
                   use_cmc, device, curve_log, env_fn=None, expert_fn=None):
    env_fn = env_fn or make_env
    n_demo = args.demo_episodes
    expert = (expert_fn or get_expert)(task_name) if n_demo > 0 else None
    demo_rng = np.random.default_rng([args.seed, TASK_SEQUENCE.index(task_name), 11])
    label = label_of(task_name)
    steps, warmup, bs, gamma = (args.steps_per_task, args.warmup,
                                args.batch_size, args.gamma)
    print(f"\n{'=' * 64}\n  Stage {stage_idx + 1}: Task {label} {task_name}  "
          f"({steps} steps, CMC {'ON' if use_cmc else 'OFF'})"
          + (f"  memory/task={cmc.memory.counts_per_task()}" if use_cmc else "")
          + (f"\n  first {n_demo} episodes: scripted expert + N(0,{args.demo_noise}) noise"
             + (f"; {args.demo_ratio:.0%} of every batch drawn from them" if args.demo_ratio else "")
             if n_demo else "")
          + f"\n{'=' * 64}")

    env = SafeEnv(env_fn(task_name, goal_mode=args.goal_mode))
    ep_rng = np.random.default_rng([args.seed, TASK_SEQUENCE.index(task_name), 7])
    new_ep_seed = lambda: int(ep_rng.integers(0, 2 ** 31 - 1))

    obs, _ = env.reset(new_ep_seed())
    ep_ret, ep_succ, ep_len, n_eps = 0.0, 0.0, 0, 0
    recent_ret, recent_succ = [], []
    # Last loss values kept as tensors; .item() (a GPU sync) only at log time.
    zero = torch.zeros((), device=device)
    td_t = q_t = act_t = cA_t = cC_t = zero
    anchor_checked = False
    demo_succ = []
    n_demo_transitions = 0
    use_mem = lambda: use_cmc and cmc.memory.size > 0
    seen = stage_tasks[: stage_idx + 1]
    t0 = time.time()

    for step in range(steps):
        in_demo = n_eps < n_demo
        if in_demo:
            action = expert.get_action(obs)
            if args.demo_noise > 0:
                action = action + demo_rng.normal(0.0, args.demo_noise, action.shape)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        elif step < warmup:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        nobs, reward, terminated, truncated, info, bad = env.step(action)
        if bad:                                    # drop transition, new episode
            obs, _ = env.reset(new_ep_seed())
            ep_ret, ep_succ, ep_len = 0.0, 0.0, 0
        else:
            ep_len += 1
            # bootstrap mask = terminated only; reward scaling affects training only
            replay.add(obs, action, float(reward) * args.reward_scale, nobs, float(terminated))
            if in_demo:
                n_demo_transitions += 1
            obs = nobs
            ep_ret += float(reward)
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            if terminated or truncated or ep_len >= MAX_EPISODE_STEPS:
                if in_demo:
                    demo_succ.append(ep_succ)
                    if len(demo_succ) == n_demo:
                        print(f"  [demo] {n_demo} expert episodes collected, "
                              f"success={np.mean(demo_succ):.2f}, buffer={replay.size}", flush=True)
                else:
                    recent_ret = (recent_ret + [ep_ret])[-20:]
                    recent_succ = (recent_succ + [ep_succ])[-20:]
                n_eps += 1
                obs, _ = env.reset(new_ep_seed())
                ep_ret, ep_succ, ep_len = 0.0, 0.0, 0

        # -------------------------------------------------------- SAC update
        if replay.size >= max(bs, warmup) and not in_demo:
            s, a, r, ns, term = replay.sample_mixed(bs, device, n_demo_transitions,
                                                    args.demo_ratio)
            with torch.no_grad():
                na, nlogp = agent.actor.sample(ns)
                tq = (torch.minimum(agent.q1_target(ns, na), agent.q2_target(ns, na)).squeeze(-1)
                      - agent.alpha.detach() * nlogp.squeeze(-1))
                target = r + gamma * (1.0 - term) * tq
            q1 = agent.q1(s, a).squeeze(-1)
            q2 = agent.q2(s, a).squeeze(-1)
            td = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            closs = td
            if use_mem():
                cC = cmc.critic_loss(agent, bs, device)
                closs = closs + cmc.critic_lambda * cC
                cC_t = cC.detach()
            agent.critic_opt.zero_grad()
            closs.backward()
            agent.critic_opt.step()
            td_t, q_t = td.detach(), q1.detach().abs().mean()

            if step % 2 == 0:
                pi, logp = agent.actor.sample(s)
                qpi = torch.minimum(agent.q1(s, pi), agent.q2(s, pi)).squeeze(-1)
                aloss = (agent.alpha.detach() * logp.squeeze(-1) - qpi).mean()
                if use_mem():
                    cA = cmc.actor_loss(agent, bs, device)
                    if not anchor_checked:
                        assert cA.requires_grad, "CMC actor anchor has no gradient"
                        anchor_checked = True
                    aloss = aloss + cmc.actor_lambda * cA
                    cA_t = cA.detach()
                agent.actor_opt.zero_grad()
                aloss.backward()
                agent.actor_opt.step()
                act_t = aloss.detach()
                alpha_loss = -(agent.log_alpha
                               * (logp.detach().squeeze(-1) + agent.target_entropy)).mean()
                agent.alpha_opt.zero_grad()
                alpha_loss.backward()
                agent.alpha_opt.step()
                agent.soft_update()

        # ------------------------------------------------------------ logging
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            sps = (step + 1) / (time.time() - t0)
            td_v, q_v, act_v, cA_v, cC_v = (float(x) for x in (td_t, q_t, act_t, cA_t, cC_t))
            msg = (f"[{label} {step + 1:>7}] eps={n_eps} train_ret={np.mean(recent_ret or [0]):7.1f} "
                   f"train_succ={np.mean(recent_succ or [0]):.2f} TD={td_v:8.2f} |Q|={q_v:6.1f} "
                   f"A={act_v:7.2f} alpha={agent.alpha.item():.4f} unstable={env.n_bad} "
                   f"{sps:.0f} steps/s")
            if use_mem():
                msg += f" | cmcA={cA_v:.4f} cmcC={cC_v:.2f}"
            print(msg, flush=True)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0 and step + 1 < steps:
            point = {"stage": label, "step_in_stage": step + 1,
                     "global_step": stage_idx * steps + step + 1}
            parts = []
            for t in seen:
                m = evaluate(agent, t, eval_episode_seeds(t, args.curve_episodes),
                             args.goal_mode, env_fn)
                point[label_of(t)] = {"success": m["mean_success"], "return": m["mean_return"]}
                parts.append(f"{label_of(t)} succ={m['mean_success']:.2f}")
            curve_log.append(point)
            print("  >>> curve: " + "  ".join(parts), flush=True)

    n_bad = env.n_bad
    env.close()
    cons = None
    if use_cmc and stage_idx < len(stage_tasks) - 1:   # nothing to protect after the last task
        cons = cmc.consolidate_task(agent, replay, stage_idx,
                                    max_candidates=min(args.cmc_candidates, replay.size),
                                    device=device)
        print(f"  [CMC] consolidated {label}: candidates={cons['candidates']} "
              f"kept={cons['kept']} memory/task={cons['per_task']}")
    return {"unstable_transitions": n_bad, "consolidation": cons,
            "demo_success": float(np.mean(demo_succ)) if demo_succ else None,
            "train_success_last20": float(np.mean(recent_succ or [0])),
            "wall_time_s": time.time() - t0}


# ═══════════════════════════════════════════════════════════════ experiment
def run_experiment(args, out_dir, device):
    use_cmc = not args.no_cmc
    stage_tasks = args.tasks
    labels = [label_of(t) for t in stage_tasks]
    arm = "SAC + CMC" if use_cmc else "SAC baseline (no CMC)"
    print(f"\n{'#' * 70}\n  {arm}   seed={args.seed}   tasks={stage_tasks}\n"
          f"  steps/task={args.steps_per_task}  goal_mode={args.goal_mode}  "
          f"eval episodes/task={args.eval_episodes}\n"
          f"  critic LayerNorm={args.layer_norm}  reward_scale={args.reward_scale}  "
          f"demo_episodes={args.demo_episodes}  demo_ratio={args.demo_ratio}\n"
          f"  output: {out_dir}\n{'#' * 70}")

    seed_everything(args.seed)
    env_info = environment_report(args.goal_mode)

    probe = make_env(stage_tasks[0], goal_mode=args.goal_mode)
    obs_dim = int(np.prod(probe.observation_space.shape))
    act_dim = int(np.prod(probe.action_space.shape))
    low, high = probe.action_space.low.copy(), probe.action_space.high.copy()
    probe.close()

    agent = SACAgent(obs_dim, act_dim, low, high, device, layer_norm=args.layer_norm)
    cmc = CounterfactualMemoryConsolidator(
        CMCMemory(args.cmc_memory_size if use_cmc else 0, obs_dim, act_dim),
        actor_lambda=args.cmc_actor_lambda, critic_lambda=args.cmc_critic_lambda,
        selection=args.memory_selection, recent_frac=args.recent_frac, gamma=args.gamma)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    _dump({"args": vars(args), "arm": arm, "use_cmc": use_cmc, "labels": labels,
           "eval_episode_seeds": {t: eval_episode_seeds(t, args.eval_episodes)
                                  for t in stage_tasks},
           "environment": env_info, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
          out_dir / "config.json")

    rows, curve_log, stage_info = [], [], {}
    for k, task in enumerate(stage_tasks):
        if args.reset_alpha_per_task and k > 0:
            agent.reset_alpha_and_optimizers()
        replay = ReplayBuffer(args.replay_size or args.steps_per_task, obs_dim, act_dim)
        stage_info[labels[k]] = train_one_task(agent, task, stage_tasks, k, args, replay,
                                               cmc, use_cmc, device, curve_log)
        del replay
        agent.save(str(out_dir / "checkpoints" / f"stage_{labels[k]}.pt"))

        print(f"\n  --- Evaluation after {labels[k]} ({args.eval_episodes} fixed episodes/task) ---")
        for t in stage_tasks[: k + 1]:
            m = evaluate(agent, t, eval_episode_seeds(t, args.eval_episodes), args.goal_mode)
            tag = "new" if t == task else "old"
            print(f"    {label_of(t)} {t:<15} [{tag}] success={m['mean_success']:.2f}  "
                  f"return={m['mean_return']:8.1f}  unstable_eps={m['n_unstable_episodes']}")
            rows.append({"trained_task_label": labels[k], "trained_task_name": task,
                         "eval_task_label": label_of(t), "eval_task_name": t,
                         "use_cmc": use_cmc, "seed": args.seed, **m})
        _dump(rows, out_dir / "results.json")
        _dump(curve_log, out_dir / "learning_curves.json")
        _dump(stage_info, out_dir / "stage_info.json")

    succ = cl_metrics(build_matrix(rows, "mean_success"), labels)
    ret = cl_metrics(build_matrix(rows, "mean_return"), labels)
    learned = {lbl: succ["new_task"].get(lbl, 0.0) >= args.learned_threshold for lbl in labels}
    summary = {"arm": arm, "seed": args.seed, "labels": labels,
               "success": succ, "return": ret,
               "learned_threshold": args.learned_threshold, "task_learned": learned}
    _dump(summary, out_dir / "summary.json")

    print(f"\n{'=' * 64}\n  SUMMARY ({arm}, seed {args.seed}) — success rate\n{'=' * 64}")
    for lbl in labels:
        line = f"  {lbl}: learned={fmt(succ['new_task'].get(lbl))}"
        for later in labels[labels.index(lbl) + 1:]:
            key = f"{lbl}_after_{later}"
            line += (f" | after {later}: forgetting={fmt(succ['forgetting'].get(key))}"
                     f" retention={fmt(succ['retention'].get(key))}")
        line += "" if learned[lbl] else f"   <-- below {args.learned_threshold} (not reliably learned)"
        print(line)
    print(f"  average final success={fmt(succ['avg_final'])}  "
          f"average forgetting={fmt(succ['avg_forgetting_final'])}")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=TASK_SEQUENCE, choices=TASK_SEQUENCE,
                   help="subset in training order (single task = base-SAC check)")
    p.add_argument("--goal-mode", choices=["random", "fixed"], default="random")
    p.add_argument("--steps-per-task", type=int, default=300_000)
    p.add_argument("--warmup", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--replay-size", type=int, default=0, help="0 = steps-per-task")
    p.add_argument("--cmc-memory-size", type=int, default=20_000)
    p.add_argument("--cmc-candidates", type=int, default=5_000)
    p.add_argument("--cmc-actor-lambda", type=float, default=1.0)
    p.add_argument("--cmc-critic-lambda", type=float, default=0.25)
    p.add_argument("--memory-selection", choices=["priority", "uniform"], default="priority")
    p.add_argument("--recent-frac", type=float, default=0.3)
    p.add_argument("--reset-alpha-per-task", action="store_true")
    p.add_argument("--demo-episodes", type=int, default=0,
                   help="start each task with N scripted-expert episodes in the replay buffer")
    p.add_argument("--demo-noise", type=float, default=0.1,
                   help="Gaussian action noise added to the expert during demo episodes")
    p.add_argument("--demo-ratio", type=float, default=0.0,
                   help="fraction of every training batch drawn from the demo transitions")
    p.add_argument("--layer-norm", action="store_true", help="LayerNorm in the critics")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="multiply rewards stored for training (logs/eval use raw reward)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--curve-episodes", type=int, default=10)
    p.add_argument("--learned-threshold", type=float, default=0.8)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="results")
    p.add_argument("--tag", default="")
    p.add_argument("--no-cmc", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args(argv)
    if [t for t in TASK_SEQUENCE if t in args.tasks] != args.tasks:
        p.error(f"--tasks must follow the order {TASK_SEQUENCE}")
    if not 0.0 <= args.demo_ratio < 1.0:
        p.error("--demo-ratio must be in [0, 1)")
    if args.demo_ratio > 0 and args.demo_episodes == 0:
        p.error("--demo-ratio needs --demo-episodes > 0")
    if args.demo_ratio > 0 and args.replay_size and args.replay_size < args.steps_per_task:
        p.error("--demo-ratio needs --replay-size >= --steps-per-task (demos must not be overwritten)")
    return args


def output_dir(args):
    sub = "before_cmc" if args.no_cmc else "after_cmc"
    if args.tasks != TASK_SEQUENCE:
        sub = "single_" + "+".join(t.replace("-v2", "") for t in args.tasks) + \
              ("" if args.no_cmc else "_cmc")
    if args.tag:
        sub += f"_{args.tag}"
    if args.smoke_test:
        sub = "smoke_" + sub
    return Path(args.output) / sub / f"seed_{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_test:
        args.steps_per_task, args.warmup = 1500, 300
        args.eval_every, args.curve_episodes, args.eval_episodes = 750, 2, 2
        args.cmc_memory_size, args.log_every = 300, 500
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    return run_experiment(args, output_dir(args), device)


if __name__ == "__main__":
    main()
