from configs.fields import Field as F
from muse.models.bc.hydra.hydra_decoder import HydraRNNActionDecoder
from muse.models.bc.hydra.hydra_gcbc import HydraGCBC
from muse.models.model import Model
from muse.utils.loss_utils import get_default_mae_action_loss_fn, mse_err_fn, get_default_nll_loss_fn, mae_err_fn
from attrdict import AttrDict as d

export = d(
    exp_name='_hydra{?use_mode_predictor:-mp}-l2{?sparse_l1:-sl1}'
             '_g{gamma}_mb{mode_beta}{?label_smoothing:_ls{label_smoothing}}',
    cls=HydraGCBC,
    use_goal=False,
    use_last_state_goal=False,

    normalize_states=False,
    save_action_normalization=True,

    # macros
    use_policy_dist=False,
    use_mode_predictor=False,
    sparse_l1=False,

    # hydra model/loss specific
    gamma=0.5,
    mode_beta=0.01,
    label_smoothing=0.0,
    use_smooth_mode=False,
    head_size=200,
    # fill this in to override mode_head_size (which defaults to head_size)
    mode_predictor_size=0,

    action_names=['action'],
    sparse_action_names=['target/position', 'target/orientation'],

    # mode loss functions (normalized sparse)
    mode0_loss_fn=F(['sparse_l1', 'sparse_action_names'],
                    lambda l1, x: get_default_mae_action_loss_fn(policy_out_names=x,
                                                                 vel_act=False,
                                                                 err_fn=mae_err_fn if l1 else mse_err_fn,
                                                                 policy_out_norm_names=x)),
    mode1_loss_fn=F(['use_policy_dist', 'action_names'],
                    lambda pd, x: get_default_nll_loss_fn(x, policy_dist_name="action_decoder/decoder/policy_raw",
                                                          policy_out_norm_names=[], vel_act=True)
                    if pd else
                    get_default_mae_action_loss_fn(x, max_grab=None,
                                                   err_fn=mse_err_fn, vel_act=True,
                                                   policy_out_norm_names=[])
                    ),

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
        exp_name='_{rnn_type}-hs{hidden_size}{?policy_size:-aps{policy_size}}-ps{action_head_size}-ms{mode_head_size}'
                 '{?use_policy_dist:-pd{policy_num_mix}{?use_tight_sigma:t}}'
                 '_sms{sparse_mlp_size}',
        cls=HydraRNNActionDecoder,
        use_mode_predictor=F('../use_mode_predictor'),
        input_names=F(['../state_names', '../extra_names'], lambda s, e: s + e),
        action_names=F('../action_names'),
        sparse_action_names=F('../sparse_action_names'),
        mode_key='mode',
        rnn_type='lstm',
        mp_rnn_type='lstm',
        use_policy_dist=F('../use_policy_dist'),
        policy_num_mix=1,
        use_policy_dist_mean=True,
        use_tight_sigma=False,
        policy_sig_min=F('use_tight_sigma', lambda x: 1e-2 if x else 1e-03),
        policy_sig_max=F('use_tight_sigma', lambda x: 2. if x else 10.0),
        use_tanh_out=True,
        dropout_p=0,
        hidden_size=400,
        policy_size=0,
        action_head_size=F('../head_size'),
        mode_head_size=F(['../head_size', '../mode_predictor_size'], lambda x, m: x if m == 0 else m),
        sparse_mlp_size=F('../head_size'),
        decoder_inter_size=F('hidden_size'),
        rnn_depth=2,
    ),
)
