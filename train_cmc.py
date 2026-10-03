"""
Continual SAC + Counterfactual Memory Consolidation on Meta-World.

Default sequence:
10 Meta-World tasks repeated twice, matching mikelma/componet.
For a quick test, use --steps-per-task 10000 --warmup 1000.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cmc_rl.tasks import TASKS, make_task
from cmc_rl.networks import SACAgent
from cmc_rl.replay import ReplayBuffer, CMCMemory
from cmc_rl.cmc import CounterfactualMemoryConsolidator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(agent, task_id, episodes, device):
    env = make_task(task_id)
    returns, successes = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            if terminated or truncated:
                returns.append(ep_ret)
                successes.append(float(info.get("success", 0.0)))
                break
    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def train_one_task(agent, task_id, steps, warmup, batch_size, replay, cmc,
                   device, eval_every, eval_episodes, output_dir, stage):
    env = make_task(task_id)
    obs, _ = env.reset()
    episode_return = 0.0

    for step in range(steps):
        if step < warmup:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_return += reward

        if done:
            obs, _ = env.reset()
            episode_return = 0.0

        if replay.size >= max(batch_size, warmup):
            s, a, r, ns, d = replay.sample(batch_size, device)

            with torch.no_grad():
                na, nlogp = agent.actor.sample(ns)
                target_q = torch.minimum(
                    agent.q1_target(ns, na), agent.q2_target(ns, na)
                ).squeeze(-1) - agent.alpha.detach() * nlogp.squeeze(-1)
                target = r + 0.99 * (1-d) * target_q

            q1 = agent.q1(s, a).squeeze(-1)
            q2 = agent.q2(s, a).squeeze(-1)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            if cmc.memory.size:
                _, c_critic = cmc.loss(agent, s)
                critic_loss = critic_loss + cmc.critic_lambda * c_critic

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            if step % 2 == 0:
                pi, logp = agent.actor.sample(s)
                qpi = torch.minimum(agent.q1(s, pi), agent.q2(s, pi)).squeeze(-1)
                actor_loss = (agent.alpha.detach() * logp.squeeze(-1) - qpi).mean()

                if cmc.memory.size:
                    c_actor, _ = cmc.loss(agent, s)
                    actor_loss = actor_loss + cmc.actor_lambda * c_actor

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()

                # Entropy temperature
                alpha_loss = -(
                    agent.log_alpha * (logp.detach().squeeze(-1) + agent.target_entropy)
                ).mean()
                agent.alpha_opt.zero_grad()
                alpha_loss.backward()
                agent.alpha_opt.step()

                agent.soft_update()

        if eval_every and (step + 1) % eval_every == 0:
            ret, succ = evaluate(agent, task_id, eval_episodes, device)
            print(f"task={task_id} step={step+1} return={ret:.2f} success={succ:.3f}")

    env.close()

    # After finishing the task, select a small protected CMC memory.
    added = cmc.consolidate_task(
        agent, replay, task_id,
        max_candidates=min(5000, replay.size),
        device=device
    )
    print(f"CMC consolidated task {task_id}: candidates={added}, memory={cmc.memory.size}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps-per-task", type=int, default=1_000_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--replay-size", type=int, default=1_000_000)
    p.add_argument("--cmc-memory-size", type=int, default=10_000)
    p.add_argument("--cmc-actor-lambda", type=float, default=1.0)
    p.add_argument("--cmc-critic-lambda", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", type=str, default="runs/cmc_metaworld")
    p.add_argument("--no-cmc", action="store_true",
                   help="Run ordinary continual SAC for the baseline.")
    p.add_argument("--short-sequence", action="store_true",
                   help="Run only the first 5 tasks, then repeat them.")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    sequence = TASKS if not args.short_sequence else TASKS[:5] + TASKS[:5]
    first_env = make_task(0)
    obs_dim = int(np.prod(first_env.observation_space.shape))
    act_dim = int(np.prod(first_env.action_space.shape))
    low = first_env.action_space.low
    high = first_env.action_space.high
    first_env.close()

    agent = SACAgent(obs_dim, act_dim, low, high, device)
    replay = ReplayBuffer(args.replay_size, obs_dim, act_dim)
    memory = CMCMemory(args.cmc_memory_size, obs_dim, act_dim)
    cmc = CounterfactualMemoryConsolidator(
        memory,
        actor_lambda=args.cmc_actor_lambda,
        critic_lambda=args.cmc_critic_lambda,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for stage, task_name in enumerate(sequence):
        task_id = stage if not args.short_sequence else [0,1,2,3,4,0,1,2,3,4][stage]
        print(f"\n=== STAGE {stage+1}/{len(sequence)}: {task_name} (id={task_id}) ===")

        # Important continual-learning behavior: retain the same agent.
        # Replay is reset at task boundaries; CMC memory persists.
        replay = ReplayBuffer(args.replay_size, obs_dim, act_dim)

        train_one_task(
            agent, task_id, args.steps_per_task, args.warmup, args.batch_size,
            replay, cmc if not args.no_cmc else CounterfactualMemoryConsolidator(
                CMCMemory(0, obs_dim, act_dim)
            ),
            device, args.eval_every, args.eval_episodes, str(out), stage
        )

        # Evaluate on every task seen so far, including the current repeated task.
        seen_ids = list(dict.fromkeys(
            ([0,1,2,3,4] if args.short_sequence else list(range(10)))
            [:min(stage + 1, 10)]
        ))
        for eval_id in seen_ids:
            ret, succ = evaluate(agent, eval_id, args.eval_episodes, device)
            results.append({
                "stage": stage,
                "trained_task_id": task_id,
                "evaluated_task_id": eval_id,
                "trained_task": task_name,
                "evaluated_task": TASKS[eval_id],
                "return": ret,
                "success": succ,
            })

        agent.save(out / f"stage_{stage:02d}_task_{task_id}.pt")

        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\nFinished. Results:", out / "results.json")


if __name__ == "__main__":
    main()
