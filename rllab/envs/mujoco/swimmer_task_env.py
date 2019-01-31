from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .swimmer_env import SwimmerEnv
import numpy as np

class SwimmerTaskConfig(object):
    goal_velocity: float

    def __init__(self):
        self.goal_velocity = 1.0

    def sample_goal_velocity(self) -> float:
        return np.random.uniform(0.0, 1.0)

    def sample(self) -> None:
        self.goal_velocity = self.sample_goal_velocity()

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'


class SwimmerTaskEnv(SwimmerEnv):
    """
        A swimmer environment with a configurable goal velocity
    """

    def __init__(self, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            self._task_config = None

        super(SwimmerTaskEnv, self).__init__(*args, **kwargs)

    @overrides
    def step(self, action):
        """
        Same as SwimmerEnv except run_cost is |actual - target| rather than actual.
        Cross-reference with Chelsea's implementation, in particular forward_reward computation:
        https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/swimmer_randgoal_env.py#L52
        """
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = -1.5 * np.abs(self.get_body_comvel("torso")[0] - self._task_config.goal_velocity)
        reward = forward_reward - ctrl_cost
        done = False
        return Step(next_obs, reward, done)
