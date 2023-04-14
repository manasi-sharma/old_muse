from configs.fields import Field as F
from muse.models.bc.action_decoders import RNNActionDecoder
from muse.models.bc.gcbc import BaseGCBC
from muse.models.model import Model
from muse.utils.loss_utils import get_default_mae_action_loss_fn, mse_err_fn, get_default_nll_loss_fn
from attrdict import AttrDict as d

export = d(
    exp_name='_bc-l2',
    cls=BaseGCBC,
    use_goal=False,
    use_last_state_goal=False,

    normalize_states=False,
    save_action_normalization=False,
    use_policy_dist=False,
    horizon=10,

    # names
    goal_names=['object'],
    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    extra_names=[],

    # encoders
    state_encoder_order=['proprio_encoder'],
    proprio_encoder=d(
        # normalizes and replaces these keys
        cls=Model,
        normalize_inputs=F('../normalize_states'),
        normalization_inputs=F('../state_names'),
    ),

    model_order=['proprio_encoder', 'action_decoder'],
    action_decoder=d(
        exp_name='_{rnn_type}-hs{hidden_size}-ps{policy_size}{?use_policy_dist:-pd{policy_num_mix}}',
        cls=RNNActionDecoder,
        input_names=F(['../state_names', '../extra_names'], lambda s, e: s + e),
        action_names=['action'],
        rnn_type='lstm',
        use_policy_dist=F('../use_policy_dist'),
        policy_num_mix=1,
        use_policy_dist_mean=True,
        policy_sig_min=1e-03,
        policy_sig_max=10.0,
        use_tanh_out=True,
        policy_sample_cat=False,
        dropout_p=0,
        hidden_size=400,
        policy_size=0,
        rnn_depth=2,
        flush_horizon=F('../horizon'),
    ),
    loss_fn=F('use_policy_dist',
              lambda x: get_default_nll_loss_fn(['action'], policy_out_norm_names=[], vel_act=True)
              if x else
              get_default_mae_action_loss_fn(['action'], max_grab=None,
                                             err_fn=mse_err_fn, vel_act=True,
                                             policy_out_norm_names=[])
              )
)
