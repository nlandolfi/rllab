import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.misc.overrides import overrides

class HalfCheetahTaskConfig(object):
    goal_velocity: float

    def __init__(self):
        self.goal_velocity = 2.0

    def sample_goal_velocity(self) -> float:
        return np.random.uniform(0.0, 2.0)

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
        """
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = np.abs(self.get_body_comvel("torso")[0] - self._task_config.goal_velocity)
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)
