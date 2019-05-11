import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

class Ant2DTaskConfig(object):
    goal_velocity: np.array

    def __init__(self):
        self.goal_velocity = [3.0, 0]

    def sample_goal_velocity(self, lo=np.array([-3, -3]), hi=np.array([3.0, 3.0])) -> float:
        return np.random.uniform(lo, hi)

    def sample(self) -> None:
        self.goal_velocity = self.sample_goal_velocity()

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity}'

class Ant2DTaskEnv(AntEnv):
    """
        An ant environment with a configurable goal velocity.
    """
    _task_config: Ant2DTaskConfig

    def __init__(self, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = None

        super(Ant2DTaskEnv, self).__init__(*args, **kwargs)

    @overrides
    def step(self, action):
        """
        Same as AntTaskEnv except two-dimensional goal velocity.
        """
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = -np.abs(comvel[0] - self._task_config.goal_velocity[0]) + 1.0
        forward_reward += -np.abs(comvel[1] - self._task_config.goal_velocity[1]) + 1.0
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)
