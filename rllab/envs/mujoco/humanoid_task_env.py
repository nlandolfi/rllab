import math
import numpy as np
from .humanoid_env import HumanoidEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

class HumanoidTaskConfig(object):
    goal_velocity: float

    def __init__(self):
        self.goal_velocity = 3.0

    def sample_goal_velocity(self, lo=-3.0, hi=3.0) -> float:
        return np.random.uniform(lo, hi)

    def sample(self) -> None:
        self.goal_velocity = self.sample_goal_velocity()

    def __str__(self):
        if self.goal_velocity == -math.inf:
            return 'Goal Velocity = BACKWARD (-inf)'
        elif self.goal_velocity == math.inf:
            return 'Goal Velocity = FORWARD (+inf)'
        else:
            return f'Goal Velocity = {self.goal_velocity:.4f}'

class HumanoidTaskEnv(HumanoidEnv):
    _task_config: HumanoidTaskConfig

    def __init__(self, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HumanoidTaskEnv, self).__init__(*args, **kwargs)
        
    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        alive_bonus = self.alive_bonus
        data = self.model.data

        comvel = self.get_body_comvel("torso")

        if self._task_config.goal_velocity == -math.inf:
            lin_vel_reward = -comvel[0]
        elif self._task_config.goal_velocity == math.inf:
            lin_vel_reward = comvel[0]
        else:
            lin_vel_reward = -np.abs(comvel[0] - self._task_config.goal_velocity)

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost
        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0

        return Step(next_obs, reward, done)