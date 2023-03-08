from attrdict.utils import get_with_default

from muse.envs.param_spec import ParamEnvSpec
from muse.utils.general_utils import strlist


class BiModalParamEnvSpec(ParamEnvSpec):
    """
    Adds mode0, mode1 action specs
    """

    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)
        self.mode0_action_names = strlist(get_with_default(params, "mode0_action_names", []))
        self.mode1_action_names = strlist(get_with_default(params, "mode1_action_names", []))

        # used in dynamics
        self.dynamics_state_names = strlist(get_with_default(params, "dynamics_state_names", self.observation_names))

        # dynamics using a waypoint (mode 0)
        self.wp_dynamics_fn = params << "wp_dynamics_fn"

        # check all names are in action_names
        not_present = list(set(self.mode0_action_names).difference(self.action_names))
        assert len(not_present) == 0, f"Action names missing mode 0 names: {not_present}"
        not_present = list(set(self.mode1_action_names).difference(self.action_names))
        assert len(not_present) == 0, f"Action names missing mode 1 names: {not_present}"

        # check for state names
        not_present = list(set(self.dynamics_state_names).difference(self.observation_names))
        assert len(not_present) == 0, f"Obs names missing dynamics names: {not_present}"
