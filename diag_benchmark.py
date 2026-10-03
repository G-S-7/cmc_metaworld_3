import os, time, torch, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import metaworld

# Check episode length
env_cls = metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE['push-v3-goal-observable']
env = env_cls(seed=42)
obs, _ = env.reset()

step_count = 0
total_reward = 0
while True:
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    total_reward += r
    step_count += 1
    if term or trunc:
        print(f'Episode ended at step {step_count}: terminated={term} truncated={trunc}')
        print(f'Final success={info["success"]}, total_reward={total_reward:.2f}')
        break
    if step_count > 600:
        print(f'Did NOT terminate/truncate after {step_count} steps')
        break
env.close()

# Benchmark speed without SAC
env2 = env_cls(seed=42)
obs, _ = env2.reset()
t0 = time.time()
N = 2000
for _ in range(N):
    a = env2.action_space.sample()
    obs, r, term, trunc, info = env2.step(a)
    if term or trunc:
        obs, _ = env2.reset()
elapsed = time.time() - t0
env2.close()
steps_per_sec = N / elapsed
print(f'\nRaw env benchmark: {N} steps in {elapsed:.2f}s = {steps_per_sec:.0f} steps/sec')

# Benchmark with SAC forward pass
from cmc_rl.networks import SACAgent
device = torch.device('cpu')
agent = SACAgent(39, 4, np.array([-1.,-1.,-1.,-1.]), np.array([1.,1.,1.,1.]), device)

env3 = env_cls(seed=42)
obs, _ = env3.reset()
t0 = time.time()
M = 500
for _ in range(M):
    a = agent.act(obs)
    obs, r, term, trunc, info = env3.step(a)
    if term or trunc:
        obs, _ = env3.reset()
elapsed2 = time.time() - t0
env3.close()
sac_steps_per_sec = M / elapsed2
print(f'SAC collect benchmark: {M} steps in {elapsed2:.2f}s = {sac_steps_per_sec:.0f} steps/sec')
print()
for budget in [20000, 50000, 100000]:
    secs = budget * 6 * 2 / sac_steps_per_sec
    print(f'  {budget} steps/task x 6 tasks x 2 exps = {secs/60:.1f} min (collect only)')
