"""
render_agent.py
===============
Render a trained checkpoint performing one task and save an MP4.

What was wrong before (and is fixed here):
  * Frames were captured BEFORE each env.step and the loop stopped on
    truncation, so the final state (often the success) was never recorded.
    Now: one frame right after reset (start state) + one frame after EVERY step.
  * Video FPS was 50; Meta-World runs at 1/dt = 80 Hz, so videos played at
    62.5% speed. Now the FPS is read from env.dt.
  * Meta-World asks for a camera named "track" that does not exist, so MuJoCo
    used a free camera. Now a fixed scene camera is used (default "corner2").
  * Episodes used random, unreproducible goals. Now the rendered episodes are
    the SAME episodes used by run_experiment.py's evaluation (same seeds).
  * Nothing proved the checkpoint was used. Now every actor parameter is
    compared with the file, and the deterministic actions are checked to differ
    from an untrained network's.

Usage:
    python render_agent.py --checkpoint results/after_cmc/seed_1/checkpoints/stage_T3.pt \
                           --task push-v2 --episodes 3 --out videos/push_after_T3.mp4
Headless Linux: set MUJOCO_GL=egl (or osmesa) before running.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from cmc_rl.tasks import TASK_SEQUENCE, MAX_EPISODE_STEPS, make_env, reset_env, env_dt
from cmc_rl.networks import SACAgent
from run_experiment import eval_episode_seeds

CAMERAS = ["corner2", "corner3", "corner", "topview", "behindGripper", "gripperPOV"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_verified_agent(checkpoint, task, goal_mode, device="cpu"):
    """Build an agent, load the checkpoint, and prove the weights are in use."""
    probe = make_env(task, goal_mode=goal_mode)
    obs_dim = int(np.prod(probe.observation_space.shape))
    act_dim = int(np.prod(probe.action_space.shape))
    low, high = probe.action_space.low.copy(), probe.action_space.high.copy()
    obs0, _ = reset_env(probe, eval_episode_seeds(task, 1)[0])
    probe.close()

    ckpt = torch.load(checkpoint, map_location=device)
    agent = SACAgent(obs_dim, act_dim, low, high, torch.device(device),
                     layer_norm=bool(ckpt.get("layer_norm", False)))
    x = torch.as_tensor(np.asarray(obs0)[None], dtype=torch.float32)
    a_untrained = agent.actor.deterministic(x).numpy()[0]
    agent.load(checkpoint)
    for name, p in agent.actor.state_dict().items():
        if not torch.equal(p.cpu(), ckpt["actor"][name].cpu()):
            raise RuntimeError(f"actor parameter {name} does not match the checkpoint")
    a_trained = agent.actor.deterministic(x).numpy()[0]
    if np.allclose(a_trained, a_untrained):
        raise RuntimeError("trained and untrained actions are identical: checkpoint not used")
    agent.actor.eval(); agent.q1.eval(); agent.q2.eval()
    print(f"  checkpoint {checkpoint}  sha256={_sha256(checkpoint)}")
    print(f"  verified: actor weights match file; first action trained={np.round(a_trained, 3)} "
          f"vs untrained={np.round(a_untrained, 3)}")
    return agent


def _overlay(frame, lines):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h = 14 * len(lines) + 8
    draw.rectangle([0, 0, img.width, h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 4 + 14 * i), line, fill=(255, 255, 255))
    return np.asarray(img)


@torch.no_grad()
def render_episode(agent, env, task, episode_seed, ep_idx, ckpt_name, overlay=True):
    obs, _ = reset_env(env, episode_seed)
    frames, ret, first_success = [], 0.0, -1

    def grab(t, r, succ):
        f = env.render()
        if f is None:
            raise RuntimeError("env.render() returned None (render_mode not set?)")
        f = np.ascontiguousarray(f, dtype=np.uint8)
        if overlay:
            status = (f"SUCCESS at step {first_success}" if first_success > 0
                      else ("success: no" if t > 0 else "start state"))
            f = _overlay(f, [f"{task} | {ckpt_name} | episode {ep_idx + 1}",
                             f"step {t:3d}/{MAX_EPISODE_STEPS}  reward {r:5.2f}  "
                             f"return {ret:7.1f}  {status}"])
        frames.append(f)

    grab(0, 0.0, 0.0)                                  # starting state
    for t in range(1, MAX_EPISODE_STEPS + 1):
        obs, r, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
        ret += float(r)
        if info.get("success", 0.0) > 0 and first_success < 0:
            first_success = t
        grab(t, float(r), info.get("success", 0.0))    # state AFTER the action
        if terminated or truncated:
            break
    return frames, ret, first_success


def save_video(frames, path, fps):
    import imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        return path
    except Exception as e:  # noqa: BLE001
        gif = path.with_suffix(".gif")
        print(f"  [WARNING] mp4 failed ({e}); writing {gif}")
        imageio.mimsave(str(gif), frames, duration=1000 / fps, loop=0)
        return gif


def render(checkpoint, task, out, episodes=3, goal_mode=None, camera="corner2",
           fps=None, overlay=True):
    checkpoint = Path(checkpoint)
    if goal_mode is None:   # use the training run's setting when available
        cfg = checkpoint.parent.parent / "config.json"
        goal_mode = (json.load(open(cfg))["args"].get("goal_mode", "random")
                     if cfg.exists() else "random")
    agent = load_verified_agent(str(checkpoint), task, goal_mode)
    env = make_env(task, goal_mode=goal_mode, render_mode="rgb_array", camera=camera)
    fps = fps or int(round(1.0 / env_dt(env)))

    all_frames, summary = [], []
    for i, ep_seed in enumerate(eval_episode_seeds(task, episodes)):
        frames, ret, t_succ = render_episode(agent, env, task, ep_seed, i,
                                             checkpoint.name, overlay)
        if np.std(frames[0]) < 1.0:
            raise RuntimeError("rendered frame is blank; check MUJOCO_GL / camera")
        all_frames += frames
        summary.append({"episode_seed": ep_seed, "return": ret,
                        "success": t_succ > 0, "first_success_step": t_succ,
                        "frames": len(frames)})
        print(f"  episode {i + 1}: return={ret:7.1f}  success={t_succ > 0}"
              + (f" (step {t_succ})" if t_succ > 0 else ""))
    env.close()

    path = save_video(all_frames, out, fps)
    try:
        import imageio
        imageio.imwrite(str(Path(out).with_suffix(".start.png")), all_frames[0])
        imageio.imwrite(str(Path(out).with_suffix(".end.png")), all_frames[-1])
    except Exception:  # noqa: BLE001
        pass
    with open(Path(out).with_suffix(".json"), "w") as f:
        json.dump({"checkpoint": str(checkpoint), "sha256": _sha256(checkpoint),
                   "task": task, "goal_mode": goal_mode, "camera": camera, "fps": fps,
                   "episodes": summary}, f, indent=2)
    print(f"  saved {path}  ({len(all_frames)} frames @ {fps} fps, camera {camera})")
    return summary


def main():
    p = argparse.ArgumentParser(description="Render a trained checkpoint to video")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True, choices=TASK_SEQUENCE)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default=None, help="output .mp4 (default videos/<task>.mp4)")
    p.add_argument("--goal-mode", choices=["random", "fixed"], default=None,
                   help="default: read from the run's config.json")
    p.add_argument("--camera", default="corner2", choices=CAMERAS)
    p.add_argument("--fps", type=int, default=None, help="default: 1/env.dt (80)")
    p.add_argument("--no-overlay", action="store_true")
    a = p.parse_args()
    render(a.checkpoint, a.task, a.out or f"videos/{a.task}.mp4", a.episodes,
           a.goal_mode, a.camera, a.fps, not a.no_overlay)


if __name__ == "__main__":
    main()
