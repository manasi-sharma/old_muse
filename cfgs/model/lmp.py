from attrdict import AttrDict as d

from muse.models.basic_model import DefaultMLPModel
from muse.models.bc.lmp.lmp import LMPBaseGCBC

from configs.fields import Field as F
from muse.models.bc.action_decoders import RNNActionDecoder
from muse.models.bc.lmp.play_helpers import get_plan_dist_fn
from muse.models.model import Model
from muse.models.rnn_model import DefaultRnnModel
from muse.utils.loss_utils import get_default_mae_action_loss_fn, mse_err_fn, get_default_nll_loss_fn

export = d(
    exp_name='_{?use_goal::ng}lmp-l2-p{plan_size}_hs{hidden_size}_bt{beta}',
    cls=LMPBaseGCBC,
    use_goal=True,

    # goals will be parsed from observation
    use_last_state_goal=True,

    normalize_states=False,
    save_action_normalization=False,
    use_policy_dist=False,

    # macros
    plan_size=64,
    hidden_size=400,
    beta=1e-4,
    replan_horizon=10,

    # names
    goal_names=['goal/object'],
    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],

    # encoders
    state_encoder_order=['proprio_encoder'],
    model_order=['proprio_encoder', 'prior', 'posterior', 'action_decoder'],
    proprio_encoder=d(
        # normalizes and replaces these keys
        cls=Model,
        normalize_inputs=F('../normalize_states'),
        normalization_inputs=F('../state_names'),
    ),

    prior=d(
        cls=DefaultMLPModel,
        use_dist=True,
        mlp_size=F('../plan_size', lambda ps: 4 * ps),
        out_size=F('../plan_size'),
        model_inputs=F(['../state_names', '../goal_names'], lambda s, g: s + g),
        model_output='plan_dist',
    ),

    posterior=d(
        cls=DefaultRnnModel,
        rnn_type='gru',
        use_dist=True,
        bidirectional=True,
        hidden_size=F('../hidden_size', lambda x: x // 2),
        out_size=F('../plan_size'),
        model_inputs=F('../state_names'),
        model_output='plan_dist',
    ),

    action_decoder=d(
        exp_name='_{rnn_type}-ps{policy_size}{?use_policy_dist:-pd{policy_num_mix}}',
        cls=RNNActionDecoder,
        input_names=F(['../state_names', '../goal_names'], lambda s, g: s + g + ['plan']),
        action_names=['action'],
        rnn_type='lstm',
        use_policy_dist=F('../use_policy_dist'),
        policy_num_mix=1,
        use_policy_dist_mean=True,
        policy_sig_min=1e-04,
        policy_sig_max=10.,
        use_tanh_out=True,
        policy_sample_cat=False,
        dropout_p=0,
        hidden_size=F('../hidden_size'),
        policy_size=0,
        rnn_depth=2,
        flush_horizon=F('../replan_horizon')
    ),
    loss_fn=F('use_policy_dist',
              lambda x: get_default_nll_loss_fn(['action'], policy_out_norm_names=[], vel_act=True)
              if x else
              get_default_mae_action_loss_fn(['action'], max_grab=None,
                                             err_fn=mse_err_fn, vel_act=True,
                                             policy_out_norm_names=[])
              ),
    plan_dist_fn=get_plan_dist_fn('plan'),
    plan_sample_fn=lambda out: d(plan=out["plan_dist"].rsample()),
)
