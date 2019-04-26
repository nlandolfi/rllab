import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.misc.overrides import overrides

class HalfCheetahTaskConfig(object):
    """
    Remember, goal_velocity is really a speed, but we follow terminology of the MAML paper.
    """
    goal_velocity: float

    def __init__(self):
        self.goal_velocity = 2.0

    def sample_goal_velocity(self, lo=-2.0, hi=-2.0) -> float:
        return np.random.uniform(lo, hi)

    def sample(self) -> None:
        self.goal_velocity = self.sample_goal_velocity()

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

class HalfCheetahTaskEnv(HalfCheetahEnv):
    """
        A half cheetah environment with a configurable goal velocity.
    """
    _task_config: HalfCheetahTaskConfig

    def __init__(self, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HalfCheetahTaskEnv, self).__init__(*args, **kwargs)

    @overrides
    def step(self, action):
        """
        Same as HalfCheetahEnv except run_cost is |actual - target| rather than actual.
        Cross-reference with Chelsea's implementation, in particular run_cost computation:
        https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand.py#L62

        The special case of goal_velocity +/- inf corresponds to half_cheetah_env_rand_direc.py environment.
        See: https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand_direc.py#L70
        """
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))

        if self._task_config.goal_velocity == -math.inf:
            run_cost = self.get_body_comvel("torso")[0]
        elif self._task_config.goal_velocity == math.inf:
            run_cost = -1 * self.get_body_comvel("torso")[0]
        else:
            run_cost = np.abs(self.get_body_comvel("torso")[0] - self._task_config.goal_velocity)
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)
