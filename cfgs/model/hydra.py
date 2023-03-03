from muse.models.bc.hydra.dynas_gcbc_model import RNN_DAS_GCBC
from muse.utils.loss_utils import get_default_mae_action_loss_fn
from attrdict import AttrDict as d

export = d(
    cls=RNN_DAS_GCBC,
    use_goal=False,
    use_final_goal=False,
    normalize_states=False,
    normalize_actions=False,
    use_vision_encoder=False,
    encoder_call_jointly=False,
    default_img_embed_size=64,
    default_use_spatial_softmax=False,
    default_use_crop_randomizer=False,
    default_use_color_randomizer=False,
    default_use_erasing_randomizer=False,
    default_downsample_frac=1.0,
    crop_frac=0.9,
    encoder_use_shared_params=False,
    use_policy_dist=False,
    policy_num_mix=1,
    use_policy_dist_mean=False,
    policy_sig_min=1e-05,
    policy_sig_max=1000.0,
    use_tanh_out=True,
    dropout_p=0,
    hidden_size=400,
    policy_size=0,
    rnn_depth=2,
    rnn_type='lstm',
    sparse_normalize_states=False,
    sparse_normalize_actions=True,
    gamma=0.5,
    mode_beta=0.01,
    balance_mode_loss=False,
    balance_cross_entropy=False,
    label_smoothing=0.0,
    use_smooth_mode=False,
    sparse_use_policy_dist=False,
    sparse_policy_num_mix=1,
    sparse_use_policy_dist_mean=False,
    sparse_policy_sig_min=1e-05,
    sparse_policy_sig_max=1000.0,
    sparse_use_tanh_out=False,
    sparse_dropout_p=0,
    sparse_mlp_size=200,
    sparse_mlp_depth=3,
    device='cuda',
    state_names=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    goal_names=['object'],
    action_names=['action'],
    sparse_action_names=['target/position', 'target/orientation'],
    raw_out_name='policy_raw',
    inner_postproc_fn=None,
    image_keys=[],
    mode0_loss_fn=get_default_mae_action_loss_fn(policy_out_names=['target/position', 'target/orientation'],
                                                 vel_act=False,
                                                 policy_out_norm_names=['target/position',
                                                                        'target/orientation']),
    mode1_loss_fn=get_default_mae_action_loss_fn(policy_out_names=['action'], vel_act=True,
                                                 policy_out_norm_names=[]),
    default_normalize_sigma=1.0,
    split_head_layers=2,
    mode_head_size=200,
    action_head_size=200,
    inner_hidden_size=400,
    use_mode_predictor=False,
    sample_cat=False,
    sparse_sample_cat=False,
)