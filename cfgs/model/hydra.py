from configs.fields import Field as F
from muse.models.bc.hydra.hydra_decoder import HydraRNNActionDecoder
from muse.models.bc.hydra.hydra_gcbc import HydraGCBC
from muse.models.model import Model
from muse.utils.loss_utils import get_default_mae_action_loss_fn, mae_err_fn
from attrdict import AttrDict as d

export = d(
    exp_name='_hydra2',
    cls=HydraGCBC,
    use_goal=False,
    use_last_state_goal=False,

    normalize_states=False,
    save_action_normalization=True,

    # hydra model/loss specific
    gamma=0.5,
    mode_beta=0.01,
    label_smoothing=0.0,
    use_smooth_mode=False,
    head_size=200,

    action_names=['action'],
    sparse_action_names=['target/position', 'target/orientation'],

    # mode loss functions (normalized sparse)
    mode0_loss_fn=F('sparse_action_names', lambda x: get_default_mae_action_loss_fn(policy_out_names=x,
                                                                                    vel_act=False,
                                                                                    policy_out_norm_names=x)),
    mode1_loss_fn=F('action_names', lambda x: get_default_mae_action_loss_fn(policy_out_names=x, vel_act=True,
                                                                             policy_out_norm_names=[])),

    # names
    goal_names=['object'],
    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],

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
        exp_name='_{rnn_type}-hs{hidden_size}-ps{action_head_size}-ms{mode_head_size}{?use_policy_dist:-pd}'
                 '_sms{sparse_mlp_size}',
        cls=HydraRNNActionDecoder,
        input_names=F('../state_names'),
        action_names=F('../action_names'),
        sparse_action_names=F('../sparse_action_names'),
        mode_key='mode',
        rnn_type='lstm',
        use_policy_dist=False,
        policy_num_mix=1,
        use_policy_dist_mean=False,
        policy_sig_min=1e-05,
        policy_sig_max=1000.0,
        use_tanh_out=True,
        dropout_p=0,
        hidden_size=400,
        policy_size=0,
        action_head_size=F('../head_size'),
        mode_head_size=F('../head_size'),
        sparse_mlp_size=F('hidden_size'),
        decoder_inter_size=F('hidden_size'),
        rnn_depth=2,
    ),
)
