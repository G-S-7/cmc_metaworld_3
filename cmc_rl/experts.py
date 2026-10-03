"""Meta-World's scripted expert controllers for the 3 tasks.

Used for two things:
  1. Verifying the environment pipeline end to end (check_expert.py): if the
     expert succeeds through OUR reset/step/success code, then observations,
     goals, rewards and success are wired correctly, and a 0% result is a
     learning problem, not an environment bug.
  2. Optional demonstration episodes at the start of each task's training
     (--demo-episodes), to give SAC non-zero reward on tasks whose shaped reward
     is ~0 under random actions (shelf-place-v2).

The scripted policies read the goal from the observation, so they need the
goal-observable envs (which is what cmc_rl.tasks.make_env builds).
"""

import warnings

import numpy as np

from cmc_rl.tasks import TASK_SEQUENCE

_POLICY_CLASSES = {
    "push-v2": "SawyerPushV2Policy",
    "push-wall-v2": "SawyerPushWallV2Policy",
    "shelf-place-v2": "SawyerShelfPlaceV2Policy",
}

# The scripted controllers use large gains and print a warning on almost every
# step ("Constant(s) may be too high"); the env clips actions to [-1, 1] anyway.
warnings.filterwarnings("ignore", message=r"Constant\(s\) may be too high")


class ExpertPolicy:
    """Wraps a scripted policy. Actions are clipped to [-1, 1], exactly what the
    env executes, so they are safe to store in the replay buffer."""

    def __init__(self, task_name):
        if task_name not in _POLICY_CLASSES:
            raise ValueError(f"no expert for {task_name}; tasks are {TASK_SEQUENCE}")
        import metaworld.policies as mwp
        self.task_name = task_name
        self.policy = getattr(mwp, _POLICY_CLASSES[task_name])()

    def get_action(self, obs):
        a = np.asarray(self.policy.get_action(np.asarray(obs, dtype=np.float64)),
                       dtype=np.float32)
        return np.clip(a, -1.0, 1.0)

    # Same interface as SACAgent, so evaluate()/render_episode() accept it.
    def act(self, obs, deterministic=True):
        return self.get_action(obs)


def get_expert(task_name):
    return ExpertPolicy(task_name)
