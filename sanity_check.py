"""
sanity_check.py
===============
Fast checks that the pipeline does what the experiment needs. Run this first.

    python sanity_check.py                 # Part A: logic checks on a fake env (~1 min)
    python sanity_check.py --env           # + Part B: the 3 real Meta-World tasks
    python sanity_check.py --env --render  # + render one frame per task (needs a GL backend)

Part A needs only torch + numpy. Part B needs metaworld + mujoco.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import traceback

import numpy as np
import torch

from cmc_rl.networks import SACAgent
from cmc_rl.replay import ReplayBuffer, CMCMemory
from cmc_rl.cmc import CounterfactualMemoryConsolidator
from cmc_rl.metrics import cl_metrics
from cmc_rl.tasks import TASK_SEQUENCE, reset_env
import run_experiment as rx

OBS, ACT = 39, 4
LOW, HIGH = -np.ones(ACT, np.float32), np.ones(ACT, np.float32)
DEV = torch.device("cpu")
RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, True))
                print(f"  PASS  {name}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False))
                print(f"  FAIL  {name}\n        {e!r}")
                traceback.print_exc(limit=3)
        return run
    return deco


# ───────────────────────────────────────────────────────────── fake env
class Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)


class FakeData:
    time = 0.0


class FakeEnv:
    """Like Meta-World: goal drawn from GLOBAL np.random at reset, never
    terminates, truncates at ep_len. Optional NaN / sim-rewind / success window."""

    def __init__(self, ep_len=50, nan_every=None, rewind_every=None, fault_step=7,
                 success_window=None):
        self.action_space = Box(LOW, HIGH)
        self.observation_space = Box(-np.inf * np.ones(OBS), np.inf * np.ones(OBS))
        self.ep_len, self.nan_every, self.rewind_every = ep_len, nan_every, rewind_every
        self.fault_step, self.success_window = fault_step, success_window
        self.data = FakeData()
        self.t, self.ep, self.goal = 0, -1, np.zeros(3)

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        o = np.random.randn(OBS).astype(np.float32)
        o[-3:] = self.goal
        return o

    def reset(self):
        self.t, self.ep, self.data.time = 0, self.ep + 1, 0.0
        self.goal = np.random.uniform(-1, 1, 3)        # global RNG, like Meta-World
        return self._obs(), {}

    def step(self, a):
        self.t += 1
        self.data.time += 0.0125
        obs = self._obs()
        if self.nan_every and self.ep % self.nan_every == 1 and self.t == self.fault_step:
            obs[0] = np.nan
        if self.rewind_every and self.ep % self.rewind_every == 1 and self.t == self.fault_step:
            self.data.time = 0.001
        w = self.success_window
        succ = 1.0 if (w and w[0] <= self.t < w[1]) else 0.0
        return obs, 1.0, False, self.t >= self.ep_len, {"success": succ}

    def close(self):
        pass


def fake_factory(**kw):
    return lambda task_name, goal_mode="random": FakeEnv(**kw)


def make_agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return SACAgent(OBS, ACT, LOW, HIGH, DEV)


def filled_memory(capacity=300, tasks=2):
    mem = CMCMemory(capacity, OBS, ACT)
    for t in range(tasks):
        mem.add_task(t, np.random.randn(200, OBS), np.random.uniform(-1, 1, (200, ACT)),
                     np.random.randn(200) * 10, np.random.rand(200))
    return mem


# ═══════════════════════════════════════════════════════════════ Part A
@check("Only the 3 tasks, in order push -> push-wall -> shelf-place")
def t_tasks():
    assert TASK_SEQUENCE == ["push-v2", "push-wall-v2", "shelf-place-v2"]
    assert rx.parse_args([]).tasks == TASK_SEQUENCE
    try:
        rx.parse_args(["--tasks", "shelf-place-v2", "push-v2"])
        raise AssertionError("out-of-order --tasks was accepted")
    except SystemExit:
        pass
    try:
        rx.parse_args(["--tasks", "hammer-v2"])
        raise AssertionError("a removed task was accepted")
    except SystemExit:
        pass


@check("reset_env: same episode seed -> same goal; global RNG untouched")
def t_reset_env():
    env = FakeEnv()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    g1 = reset_env(env, 42)[0][-3:]
    g2 = reset_env(env, 42)[0][-3:]
    g3 = reset_env(env, 43)[0][-3:]
    assert np.allclose(g1, g2) and not np.allclose(g1, g3)
    assert np.array_equal(np.random.get_state()[1], before), "global RNG was disturbed"


@check("Evaluation episodes are identical for every run seed and both arms")
def t_eval_seeds():
    s = {t: rx.eval_episode_seeds(t, 30) for t in TASK_SEQUENCE}
    assert len({tuple(v) for v in s.values()}) == 3          # different per task
    assert s["push-v2"] == rx.eval_episode_seeds("push-v2", 30)  # deterministic
    a = rx.parse_args(["--seed", "1"]); b = rx.parse_args(["--seed", "2", "--no-cmc"])
    assert a.eval_episodes == b.eval_episodes                # same default protocol


@check("Evaluation: same seeds -> same goals; success counted at ANY step")
def t_eval():
    agent = make_agent()
    seeds = rx.eval_episode_seeds("push-v2", 3)
    fn = fake_factory(success_window=(3, 5))
    m = rx.evaluate(agent, "push-v2", seeds, "random", env_fn=fn)
    assert m["mean_success"] == 1.0 and m["mean_final_step_success"] == 0.0, m
    assert m["mean_steps_to_success"] == 3, m
    goals = []
    for _ in range(2):
        env = rx.SafeEnv(fn("push-v2"))
        goals.append([env.reset(s)[0][-3:] for s in seeds])
    assert np.allclose(goals[0], goals[1])


@check("CMC actor anchor carries gradient and pulls the policy to stored actions")
def t_anchor():
    agent = make_agent()
    mem = CMCMemory(256, OBS, ACT)
    mem.add_task(0, np.random.randn(256, OBS), np.full((256, ACT), 0.5), np.zeros(256),
                 np.ones(256))
    cmc = CounterfactualMemoryConsolidator(mem)
    l0 = cmc.actor_loss(agent, 256, DEV)
    assert l0.requires_grad
    for _ in range(200):
        l = cmc.actor_loss(agent, 256, DEV)
        agent.actor_opt.zero_grad(); l.backward(); agent.actor_opt.step()
    assert cmc.actor_loss(agent, 256, DEV).item() < 0.5 * l0.item()
    x = torch.randn(4, OBS)
    assert not agent.actor.deterministic(x).requires_grad


@check("Training loop: no timeout stored as terminal, no NaN stored, curves logged")
def t_train():
    agent = make_agent()
    args = rx.parse_args(["--steps-per-task", "1500", "--warmup", "200", "--batch-size", "32",
                          "--eval-every", "500", "--curve-episodes", "1",
                          "--log-every", "100000", "--cmc-candidates", "500"])
    rb = ReplayBuffer(5000, OBS, ACT)
    mem = filled_memory(tasks=1)                 # T1 already consolidated
    cmc = CounterfactualMemoryConsolidator(mem)
    curves = []
    info = rx.train_one_task(agent, "push-wall-v2", TASK_SEQUENCE, 1, args, rb, cmc,
                             True, DEV, curves,
                             env_fn=fake_factory(nan_every=3, rewind_every=4))
    assert rb.size > 0 and rb.terminals[:rb.size].sum() == 0
    assert np.isfinite(rb.obs[:rb.size]).all() and np.isfinite(rb.next_obs[:rb.size]).all()
    assert info["unstable_transitions"] > 0
    assert mem.counts_per_task().get(1, 0) > 0, "T2 was not consolidated"
    assert len(curves) == 2 and "T1" in curves[0] and "T2" in curves[0], curves


@check("Demo episodes: expert actions stored, no SAC updates during demos, reward scale applied")
def t_demos():
    class FakeExpert:
        def get_action(self, obs):
            return np.full(ACT, 0.5, np.float32)
    agent = make_agent()
    before = [p.detach().clone() for p in agent.actor.parameters()]
    args = rx.parse_args(["--steps-per-task", "100", "--warmup", "0", "--batch-size", "16",
                          "--demo-episodes", "2", "--demo-noise", "0", "--reward-scale", "2.0",
                          "--eval-every", "0", "--log-every", "100000"])
    rb = ReplayBuffer(1000, OBS, ACT)
    info = rx.train_one_task(agent, "push-v2", TASK_SEQUENCE, 0, args, rb,
                             CounterfactualMemoryConsolidator(CMCMemory(0, OBS, ACT)),
                             False, DEV, [], env_fn=fake_factory(ep_len=50),
                             expert_fn=lambda t: FakeExpert())
    assert rb.size == 100 and np.allclose(rb.actions[:100], 0.5), "expert actions not stored"
    assert np.allclose(rb.rewards[:100], 2.0), "reward scale not applied to stored reward"
    assert all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())), \
        "SAC updated during demo episodes"
    assert info["demo_success"] == 0.0


@check("Demo ratio: half of each batch comes from the demo transitions")
def t_demo_ratio():
    rb = ReplayBuffer(1000, 1, 1)
    for i in range(600):
        rb.add([i], [0], 0, [i], 0)
    obs = rb.sample_mixed(256, DEV, 100, 0.5)[0][:, 0].numpy()
    assert (obs[:128] < 100).all() and (obs < 100).sum() >= 128
    try:
        rx.parse_args(["--demo-ratio", "0.5"])
        raise AssertionError("--demo-ratio accepted without --demo-episodes")
    except SystemExit:
        pass


@check("Critic LayerNorm: save/load round trip; mismatched architecture refused")
def t_layernorm():
    import tempfile
    a = SACAgent(OBS, ACT, LOW, HIGH, DEV, layer_norm=True)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in a.q1.modules())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in a.actor.modules())
    path = os.path.join(tempfile.mkdtemp(), "ln.pt")
    a.save(path)
    b = SACAgent(OBS, ACT, LOW, HIGH, DEV, layer_norm=True)
    b.load(path)
    x = torch.randn(3, OBS)
    assert torch.allclose(a.actor.deterministic(x), b.actor.deterministic(x))
    try:
        SACAgent(OBS, ACT, LOW, HIGH, DEV, layer_norm=False).load(path)
        raise AssertionError("loaded a LayerNorm checkpoint into a plain agent")
    except ValueError:
        pass


@check("Consolidation skipped after the last task (nothing left to protect)")
def t_no_last_consolidation():
    agent = make_agent()
    args = rx.parse_args(["--steps-per-task", "400", "--warmup", "100", "--batch-size", "32",
                          "--eval-every", "0", "--log-every", "100000"])
    mem = filled_memory()
    cmc = CounterfactualMemoryConsolidator(mem)
    info = rx.train_one_task(agent, "shelf-place-v2", TASK_SEQUENCE, 2, args,
                             ReplayBuffer(1000, OBS, ACT), cmc, True, DEV, [],
                             env_fn=fake_factory())
    assert info["consolidation"] is None and 2 not in mem.counts_per_task()


@check("Memory: equal per-task quotas regardless of priority scale")
def t_quotas():
    mem = CMCMemory(300, OBS, ACT)
    for t, scale in enumerate([1.0, 1e3, 1e6]):
        mem.add_task(t, np.random.randn(500, OBS), np.zeros((500, ACT)), np.zeros(500),
                     np.random.rand(500) * scale)
    assert mem.counts_per_task() == {0: 100, 1: 100, 2: 100}


@check("Metrics: forgetting / retention / averages on a known matrix")
def t_metrics():
    m = {"T1": {"T1": 0.9}, "T2": {"T1": 0.4, "T2": 0.8}, "T3": {"T1": 0.2, "T2": 0.5, "T3": 0.0}}
    r = cl_metrics(m, ["T1", "T2", "T3"])
    assert abs(r["forgetting"]["T1_after_T2"] - 0.5) < 1e-9
    assert abs(r["forgetting"]["T1_after_T3"] - 0.7) < 1e-9
    assert abs(r["retention"]["T2_after_T3"] - 0.625) < 1e-9
    assert abs(r["avg_forgetting_final"] - 0.5) < 1e-9          # (0.7 + 0.3) / 2
    m0 = {"T1": {"T1": 0.0}, "T2": {"T1": 0.0, "T2": 0.9}, "T3": {"T1": 0.0, "T2": 0.9, "T3": 0.1}}
    r0 = cl_metrics(m0, ["T1", "T2", "T3"])
    assert r0["tasks_counted_in_avg_forgetting"] == ["T2"]      # unlearned T1 excluded


@check("Output folders: per arm and per seed; single-task gate runs kept apart")
def t_outdir():
    assert rx.output_dir(rx.parse_args(["--seed", "7", "--no-cmc"])).parts[-2:] == ("before_cmc", "seed_7")
    assert rx.output_dir(rx.parse_args(["--seed", "7"])).parts[-2:] == ("after_cmc", "seed_7")
    assert rx.output_dir(rx.parse_args(["--no-cmc", "--tasks", "push-v2"])).parts[-2] == "single_push"


# ═══════════════════════════════════════════════════════════════ Part B
def env_checks(render):
    from cmc_rl.tasks import make_env
    print("\n[Part B] Real Meta-World environments")
    rep = rx.environment_report("random")
    ok = not rep["warnings"]
    for task in TASK_SEQUENCE:
        try:
            env = make_env(task, goal_mode="random")
            g = [reset_env(env, s)[0][-3:].copy() for s in (5, 5, 6)]
            reproducible = np.allclose(g[0], g[1]) and not np.allclose(g[0], g[2])
            envf = make_env(task, goal_mode="fixed")
            gf = [reset_env(envf, s)[0][-3:].copy() for s in (5, 6)]
            fixed_ok = np.allclose(gf[0], gf[1])
            senv = rx.SafeEnv(env)
            senv.reset(1)
            rewards, n_trunc = [], 0
            for t in range(600):
                *_, r, term, trunc, info, bad = senv.step(senv.action_space.sample())
                rewards.append(r)
                if term:
                    raise AssertionError("Meta-World returned terminated=True")
                if trunc or bad:
                    n_trunc += trunc
                    senv.reset(100 + t)
            line = (f"  {task:<16} goals reproducible={reproducible} fixed-mode={fixed_ok} "
                    f"truncations in 600 steps={n_trunc} unstable={senv.n_bad} "
                    f"random-policy reward mean={np.mean(rewards):.3f}")
            ok &= reproducible and fixed_ok and n_trunc == 1
            if render:
                renv = make_env(task, goal_mode="random", render_mode="rgb_array", camera="corner2")
                reset_env(renv, 1)
                frame = renv.render()
                good = frame is not None and np.std(frame) > 1
                line += f" frame={None if frame is None else frame.shape} ok={good}"
                ok &= good
                try:
                    import imageio
                    imageio.imwrite(f"sanity_{task}.png", frame)
                except Exception:  # noqa: BLE001
                    pass
                renv.close()
            print(line)
            env.close(); envf.close()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {task:<16} FAILED: {e!r}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", action="store_true")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    print("[Part A] Logic checks (fake environment)")
    for t in [t_tasks, t_reset_env, t_eval_seeds, t_eval, t_anchor, t_train,
              t_demos, t_demo_ratio, t_layernorm, t_no_last_consolidation, t_quotas, t_metrics, t_outdir]:
        t()
    env_ok = env_checks(a.render) if a.env else True
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} logic checks passed"
          + ("" if env_ok else "  |  environment checks reported problems"))
    sys.exit(1 if (n_fail or not env_ok) else 0)


if __name__ == "__main__":
    main()
