from .ant_task_env import AntTaskEnv


class AntMissingLegTaskEnv(AntTaskEnv):
    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        a = action.copy()
        a[0] = 0
        a[1] = 0
        return a + noise