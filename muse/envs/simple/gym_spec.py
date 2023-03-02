from muse.envs.env_spec import EnvSpec


class GymEnvSpec(EnvSpec):

    @property
    def output_observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['reward']

    @property
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['obs']

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return []

    @property
    def action_names(self):
        """
        Returns:
            list(str)
        """
        return ['action']
