import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

class AntTaskConfig(object):
    """
    Remember, goal_velocity is really a speed, but we follow terminology of the MAML paper.
    """
    goal_velocity: float

    def __init__(self):
        self.goal_velocity = 3.0

    def sample_goal_velocity(self, lo=-3.0, hi=3.0) -> float:
        return np.random.uniform(lo, hi)

    def sample(self) -> None:
        self.goal_velocity = self.sample_goal_velocity()

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

class AntTaskEnv(AntEnv):
    """
        An ant environment with a configurable goal velocity.

        The velocity is really a speed.
    """
    _task_config: AntTaskConfig

    def __init__(self, *args, **kwargs):
        if 'task_config' in kwargs['task_config']:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = None

        super(AntTaskEnv, self).__init__(*args, **kwargs)

    @overrides
    def step(self, action):
        """
        Same as AntEnv except forward_reward is |actual - target| + 1, rather than actual.
        Cross-reference with Chelsea's implementation, in particular forward_reward computaion:
        https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/ant_env_rand.py#L52

        The special case of goal_velocity +/- inf corresponds to ant_env_rand_direc.py environment.
        See: https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/ant_env_rand_direc.py#L51
        """
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        if self._task_config.goal_velocity == -math.inf:
            forward_reward = -1 * comvel[0]
        elif self._task_config.goal_velocity == math.inf:
            forward_reward = comvel[0]
        else:
            forward_reward = -np.abs(comvel[0] - self._task_config.goal_velocity) + 1.0 
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