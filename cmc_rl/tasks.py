"""Meta-World helpers for the 3-task continual-learning experiment.

Task sequence (fixed order):
    T1  push-v2         push the puck to the target
    T2  push-wall-v2    push the puck around a wall to the target
    T3  shelf-place-v2  pick the puck up and place it on a shelf

Facts verified in the Meta-World source (v2 commit c822f28 and v3.0.0):
  * Per-episode object/goal positions are drawn from the GLOBAL `np.random`
    inside reset(), not from the env's seeded RNG. `env.seed()` and the
    constructor seed therefore do NOT control which goals you get.
    -> `reset_env(env, episode_seed)` seeds the global RNG only for the
       duration of reset(), so every episode is reproducible and evaluation
       episodes are identical for every run.
  * Episodes never terminate; they truncate at max_path_length = 500.
  * Control rate is 1/dt = 80 Hz (render_fps = 80).
  * Default rendering asks for a camera called "track", which does not exist
    in the Meta-World scene, so MuJoCo falls back to a free camera.
    -> `set_camera(env, name)` selects one of the scene's fixed cameras.
"""

import re

import numpy as np

TASK_SEQUENCE = ["push-v2", "push-wall-v2", "shelf-place-v2"]
TASK_LABELS = ["T1", "T2", "T3"]
TASK_DESCRIPTIONS = {
    "push-v2": "Push puck to target",
    "push-wall-v2": "Navigate around wall + push puck to target",
    "shelf-place-v2": "Pick + place puck on shelf",
}
MAX_EPISODE_STEPS = 500

# Constructor seeds. In goal_mode="fixed" they define the single configuration
# of each task; they are the same for every seed and for baseline and CMC.
TASK_CONSTRUCT_SEEDS = {"push-v2": 101, "push-wall-v2": 102, "shelf-place-v2": 103}

_API_CACHE = {}


def get_env_dict():
    """Return (env_dict, api_version). Prefers the v2 API (the tasks are v2)."""
    if "dict" not in _API_CACHE:
        import metaworld
        v2 = getattr(metaworld, "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE", None)
        if v2 is None:
            # Meta-World 2.0 (the pinned commit) exports it from metaworld.envs only.
            try:
                from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as v2
            except ImportError:
                v2 = None
        if v2 is not None:
            _API_CACHE["dict"] = (v2, "v2")
        elif hasattr(metaworld, "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE"):
            print("[tasks] WARNING: installed Meta-World has no v2 envs; using v3 "
                  "equivalents (different reward/physics from the v2 tasks).")
            _API_CACHE["dict"] = (metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, "v3")
        else:
            raise ImportError("Installed metaworld exposes neither v2 nor v3 "
                              "goal-observable environments.")
    return _API_CACHE["dict"]


def resolve_name(task_name):
    """'push-v2' -> the key actually used with the installed API."""
    _, api = get_env_dict()
    return task_name if api == "v2" else re.sub(r"-v\d+$", f"-{api}", task_name)


def make_env(task_name, goal_mode="random", render_mode=None, camera=None):
    """Create a goal-observable Meta-World env for one of the 3 tasks.

    goal_mode="random": object/goal positions are re-sampled at every reset
                        (controlled by reset_env's episode_seed).
    goal_mode="fixed":  a single fixed configuration per task.
    """
    if task_name not in TASK_SEQUENCE:
        raise ValueError(f"{task_name} is not part of this experiment {TASK_SEQUENCE}")
    if goal_mode not in ("random", "fixed"):
        raise ValueError(f"goal_mode must be 'random' or 'fixed', got {goal_mode}")
    env_dict, _ = get_env_dict()
    key = resolve_name(task_name) + "-goal-observable"
    env = env_dict[key](seed=TASK_CONSTRUCT_SEEDS[task_name])
    # Goal-observable envs freeze the configuration after construction.
    env._freeze_rand_vec = (goal_mode == "fixed")
    if render_mode is not None:
        env.render_mode = render_mode
        if camera:
            set_camera(env, camera)
    return env


def reset_env(env, episode_seed):
    """Reset with a reproducible object/goal configuration.

    Meta-World samples positions from the global np.random during reset(), so
    the global RNG is seeded only for this call and then restored.
    """
    state = np.random.get_state()
    np.random.seed(int(episode_seed) % (2 ** 32))
    try:
        obs, info = env.reset()
    finally:
        np.random.set_state(state)
    return obs, info


def set_camera(env, name):
    """Render from a named scene camera (works with Gymnasium 0.29 and 1.x)."""
    import mujoco
    u = getattr(env, "unwrapped", env)
    cam_id = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        names = [mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                 for i in range(u.model.ncam)]
        raise ValueError(f"camera '{name}' not found; available: {names}")
    u.camera_name = None          # Gymnasium 0.29 reads env.camera_id/name
    u.camera_id = cam_id
    renderer = getattr(u, "mujoco_renderer", None)
    if renderer is not None:      # Gymnasium 1.x reads renderer.camera_id
        renderer.camera_id = cam_id
    return cam_id


def env_dt(env):
    return float(getattr(getattr(env, "unwrapped", env), "dt"))