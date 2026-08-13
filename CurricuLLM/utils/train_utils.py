import copy
import os
import sys

try:
    from farama_notifications import notifications

    notifications.pop("gymnasium_robotics", None)
except Exception:
    pass

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3.common.utils import set_random_seed

ENVIRONMENTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "environments")
)
if ENVIRONMENTS_ROOT not in sys.path:
    sys.path.insert(0, ENVIRONMENTS_ROOT)

import Curriculum
import importlib

def make_env(
    env_id: str,
    rank: int,
    task=None,
    seed: int = 0,
    render_mode: str = None,
    env_kwargs=None,
):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        importlib.reload(Curriculum)
        kwargs = copy.deepcopy(env_kwargs) if env_kwargs is not None else {}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode

        if render_mode is None:
            env = gym.make(env_id, **kwargs)
        else:
            env = gym.make(env_id, **kwargs)
            
        if task is not None:
            env.set_task(task)
        # check_env(env) # check the environment
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init
