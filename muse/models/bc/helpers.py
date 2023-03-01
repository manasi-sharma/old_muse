from muse.utils.config_utils import bool_cond_add_to_exp_name
from configs.utils import hr_name
from muse.utils.loss_utils import get_default_nll_loss_fn, mae_err_fn, pose_mae_err_fn, mse_err_fn
from attrdict.utils import get_with_default


def gcbc_declare_naming_arguments(parser):
    # things that all GCBC runs will share
    parser.add_argument("--include_goal_proprio", action='store_true')
    parser.add_argument("--no_objects", action='store_true')
    parser.add_argument("--use_quat", action='store_true', help='end effector quat, else eul')
    parser.add_argument("--single_grab", action='store_true')
    parser.add_argument("--do_grab_norm", action='store_true')
    parser.add_argument("--norm_two_sigma", action='store_true')
    parser.add_argument("--relative_actions", action='store_true')
    parser.add_argument("--include_block_sizes", action='store_true')
    parser.add_argument("--exclude_velocities", action='store_true')

    parser.add_argument("--use_image_obs", action='store_true')
    parser.add_argument("--use_ego_image_obs", action='store_true')

    parser.add_argument("--use_real_inputs", action='store_true',
                        help="uses the inputs available in the real world")
    parser.add_argument("--no_single_head", action='store_true')
    parser.add_argument("--do_mse_err", action='store_true')
    parser.add_argument("--do_pose_err", action='store_true')
    parser.add_argument("--ignore_existing_goal", action='store_true')
    
    parser.add_argument("--use_targ_quat", action='store_true', help='target end effector quat, else eul')

def dyn_gcbc_declare_naming_arguments(parser):
    gcbc_declare_naming_arguments(parser)
    parser.add_argument("--sparse_do_mse_err", action='store_true')
    parser.add_argument("--sparse_do_pose_err", action='store_true')

def gcbc_wrap_exp_name(group_name, exp_name_fn, dyn=False):
    """ Wrap high level model spec things, like inputs / outputs / loss"""

    def get_exp_name(common_params, short_name=False):
        prms = common_params >> group_name
        pref = "_dyn" if dyn else ""
        NAME = exp_name_fn(common_params) + pref + ("_gcbc" if prms['use_goal'] else "_bc")

        NAME = bool_cond_add_to_exp_name(NAME, prms, [("use_image_obs", "im" if short_name else "imobs"),
                                                      ("use_ego_image_obs", "eim" if short_name else "eimobs"),
                                                      ("ignore_existing_goal", "nogl" if short_name else "noexgoal")],
                                         sep='-')

        NAME = bool_cond_add_to_exp_name(NAME, prms, [("no_objects", "noobj"),
                                                      ("include_goal_proprio", "goalproprio"),
                                                      ("use_quat", "qt"),
                                                      ("relative_actions", "relac"),
                                                      ("single_grab", "singlegrab"),
                                                      ("do_grab_norm", "ngril" if short_name else "normgrabil"),
                                                      ("norm_two_sigma", "n2s"),
                                                      ("include_block_sizes", "bsz"),
                                                      ("exclude_velocities", "no-vel"),
                                                      ("use_real_inputs", "realin")])

        if prms["do_pose_err"]:
            NAME += "_perr"
        elif prms["do_mse_err"]:
            NAME += "_l2" if short_name else "_l2err"

        if dyn:
            if prms["sparse_do_pose_err"]:
                NAME += "_sp-perr"
            elif prms["sparse_do_mse_err"]:
                NAME += "_sp-l2" if short_name else "_sp-l2err"

        return NAME

    return get_exp_name


def base_gcbc_specific_args_to_name(NAME, prms, skip_vision=False, short_name=False):
    """ Specific to BaseGCBC class & architecture"""
    NAME = bool_cond_add_to_exp_name(NAME, prms, [
        ("use_final_goal", "goalfinal"),
        ("use_tanh_out", "tanh"),
        ("normalize_actions", "normac"),
        ("normalize_states", "normst"),
    ] + ([] if skip_vision else [("default_use_crop_randomizer", "cr" if short_name else "imcrop"),
                                 ("default_use_color_randomizer", "clr" if short_name else "imclr"),
                                 ("default_use_erasing_randomizer", "er" if short_name else "imerase"),
                                 ("default_use_spatial_softmax", "sp" if short_name else "spatmax"),
                                 ("encoder_use_shared_params", "sharedenc"),
                                 ]
         )
                                     )

    if not skip_vision:
        if (prms['use_image_obs'] or prms['use_ego_image_obs']) and prms["default_img_embed_size"] != 64:
            NAME += f"_iems{prms['default_img_embed_size']}"

        if prms['default_use_crop_randomizer'] and prms['crop_frac'] != 0.9:
            NAME += f"_cr{hr_name(prms.crop_frac)}"

        if prms['default_downsample_frac'] < 1.0:
            NAME += f"_df{hr_name(prms.default_downsample_frac)}"

    if prms["use_policy_dist"]:
        pref = 'pd' if short_name else 'pd-sig'
        NAME += f"_{pref}{hr_name(prms['policy_sig_min'])}-{hr_name(prms['policy_sig_max'])}"

        if (prms["policy_num_mix"]) > 1:
            NAME += f"-gmm{prms.policy_num_mix}"

        if prms['use_policy_dist_mean']:
            NAME += "-mn" if short_name else f"-mean"

    return NAME


def base_dyn_gcbc_specific_args_to_name(NAME, prms, skip_vision=False, short_name=False):
    """ Specific to DynamicActionBaseGCBC class & architecture"""
    NAME = base_gcbc_specific_args_to_name(NAME, prms, skip_vision=skip_vision, short_name=short_name)

    NAME = bool_cond_add_to_exp_name(NAME, prms, [
        ("sparse_normalize_states", "snst" if short_name else "sp-normst"),
        ("sparse_normalize_actions", "snac" if short_name else "sp-normac"),
        ("balance_mode_loss", "bmd" if short_name else "bal-mode"),
        ("balance_cross_entropy", "bxe" if short_name else "bal-xe"),
        ("use_smooth_mode", "sm"),
    ])

    if prms["sparse_use_policy_dist"]:
        pref = 'sp-pd' if short_name else 'sparse-pd-sig'
        NAME += f"_{pref}{hr_name(prms['sparse_policy_sig_min'])}-{hr_name(prms['sparse_policy_sig_max'])}"

        if (prms["sparse_policy_num_mix"]) > 1:
            NAME += f"-gmm{prms.sparse_policy_num_mix}"

        if prms['sparse_use_policy_dist_mean']:
            NAME += f"-mn" if short_name else f"-mean"

    NAME += f"_gma{hr_name(prms['gamma'])}_mb{hr_name(prms['mode_beta'])}"

    if prms['label_smoothing'] > 0:
        NAME += f"_lsm{hr_name(prms['label_smoothing'])}"

    return NAME


def get_gcbc_names(common_params, prms):
    utils = common_params['utils']
    env_params = common_params['env_train']
    env_spec_params = common_params['env_spec/params']

    USE_IMG = prms['use_image_obs']
    USE_EGO_IMG = prms['use_ego_image_obs']
    # no object state if manually specified or using image observations.
    NOOBJ = (USE_IMG or USE_EGO_IMG) or ("noobj" in (common_params["dataset"]) or prms['no_objects'])

    # get all the names for the model to use (env specific)
    lmp_names_and_sizes = utils.get_default_lmp_names_and_sizes(env_spec_params, "plan", 32,
                                                                prms["include_goal_proprio"], 
                                                                prms["single_grab"],
                                                                prms["do_grab_norm"],
                                                                VEL_ACT=common_params["velact"],
                                                                ENCODE_ACTIONS=False,
                                                                ENCODE_OBJECTS=not NOOBJ,
                                                                INCLUDE_BLOCK_SIZES=prms["include_block_sizes"],
                                                                USE_DRAWER=get_with_default(env_params, "use_drawer",
                                                                                            False),
                                                                NO_OBJECTS=NOOBJ,
                                                                REAL_INPUTS=prms["use_real_inputs"],
                                                                EXCLUDE_VEL=prms["exclude_velocities"],
                                                                OBS_USE_QUAT=prms["use_quat"],
                                                                TARG_USE_QUAT=prms << 'use_targ_quat')

    assert not prms.disable_only_act_norm and not prms.disable_only_obs_norm

    # determine the inputs to the policy
    nsld = env_spec_params["names_shapes_limits_dtypes"]
    STATE_NAMES = lmp_names_and_sizes["POLICY_NAMES"]
    STATE_NAMES.remove("plan")  # there is no plan in GC BC or BC

    POLICY_GOAL_STATE_NAMES = lmp_names_and_sizes["POLICY_GOAL_STATE_NAMES"]

    policy_out_names = lmp_names_and_sizes["policy_out_names"]

    image_keys = ['image'] if USE_IMG else []
    if USE_EGO_IMG:
        image_keys += ['ego_image']

    inner_postproc_fn = None
    if prms['do_pose_err']:
        # will extract policy names from the raw model output
        inner_postproc_fn = utils.get_default_policy_postproc_fn(nsld, policy_out_names, raw_out_name="policy_raw",
                                                                 use_policy_dist=prms["use_policy_dist"],
                                                                 use_policy_dist_mean=prms["no_sample"],
                                                                 relative=prms["relative_actions"],
                                                                 do_orn_norm=True)

    return lmp_names_and_sizes, STATE_NAMES, policy_out_names, POLICY_GOAL_STATE_NAMES, image_keys, inner_postproc_fn


def get_dyn_gcbc_names(common_params, prms):
    # get all the names for the model to use (env specific)
    utils = common_params['utils']
    lmp_names_and_sizes, STATE_NAMES, policy_out_names, POLICY_GOAL_STATE_NAMES, image_keys, inner_postproc_fn = get_gcbc_names(
        common_params, prms)

    env_params = common_params['env_train']
    env_spec_params = common_params['env_spec/params']

    USE_IMG = prms['use_image_obs']
    USE_EGO_IMG = prms['use_ego_image_obs']
    # no object state if manually specified or using image observations.
    NOOBJ = (USE_IMG or USE_EGO_IMG) or ("noobj" in (common_params["dataset"]) or prms['no_objects'])

    # get all the names for the model to use (env specific)
    lmp_names_and_sizes_m0 = utils.get_default_lmp_names_and_sizes(env_spec_params, "plan", 32,
                                                                   prms["include_goal_proprio"],
                                                                   prms["single_grab"],
                                                                   prms["do_grab_norm"],
                                                                   VEL_ACT=False,
                                                                   ENCODE_ACTIONS=False,
                                                                   ENCODE_OBJECTS=not NOOBJ,
                                                                   INCLUDE_BLOCK_SIZES=prms["include_block_sizes"],
                                                                   USE_DRAWER=get_with_default(env_params, "use_drawer",
                                                                                               False),
                                                                   NO_OBJECTS=NOOBJ,
                                                                   REAL_INPUTS=prms["use_real_inputs"],
                                                                   EXCLUDE_VEL=prms["exclude_velocities"],
                                                                   OBS_USE_QUAT=prms["use_quat"],
                                                                   TARG_USE_QUAT=prms << 'use_targ_quat')

    sparse_policy_out_names = lmp_names_and_sizes_m0['policy_out_names']
    lmp_names_and_sizes.sparse_policy_out_names = sparse_policy_out_names
    return lmp_names_and_sizes, STATE_NAMES, policy_out_names, sparse_policy_out_names, \
        POLICY_GOAL_STATE_NAMES, image_keys, None


def get_utils_loss_fn(utils, policy_out_names, use_policy_dist, normalize_actions, policy_num_mix,
                      do_pose_err, do_mse_err, do_grab_norm, single_grab, velact, policy_raw_name="policy_raw"):
    """ Getting a loss function """

    if use_policy_dist:
        # outputting an action distribution, requires NLL loss
        policy_out_norm_names = [] if not normalize_actions else list(policy_out_names)
        policy_loss_fn = get_default_nll_loss_fn(policy_out_names,
                                                 policy_dist_name=policy_raw_name,
                                                 relative=False,
                                                 policy_out_norm_names=policy_out_norm_names,
                                                 vel_act=velact)
    else:
        # outputting deterministic action, loss is either (1) mae (2) mse or (3) pose
        assert (policy_num_mix) <= 1, "Cannot use GMM if policy is deterministic"
        err_fn = pose_mae_err_fn if do_pose_err else mae_err_fn
        err_fn = mse_err_fn if do_mse_err else err_fn
        assert not do_mse_err or not do_pose_err, "mse+pose at once not supported"
        policy_out_norm_names = [] if not normalize_actions else list(policy_out_names)

        # pose error does its own normalization for orientations.
        if normalize_actions and do_pose_err:
            if "target/orientation_eul" in policy_out_norm_names:
                policy_out_norm_names.remove("target/orientation_eul")
            elif "target/ee_orientation_eul" in policy_out_norm_names:
                policy_out_norm_names.remove("target/ee_orientation_eul")
            else:
                raise NotImplementedError(policy_out_norm_names)

        policy_loss_fn = utils.get_action_loss_fn(policy_out_names, single_grab,
                                                  do_grab_norm, relative=False,
                                                  err_fn=err_fn, vel_act=velact,
                                                  policy_out_norm_names=policy_out_norm_names)

    return policy_loss_fn, policy_out_norm_names


def gcbc_get_loss_fn(common_params, prms, lmp_names_and_sizes):
    utils = common_params['utils']
    policy_loss_fn, policy_out_norm_names = get_utils_loss_fn(utils, lmp_names_and_sizes["policy_out_names"], *prms.get_keys_required([
        "use_policy_dist", "normalize_actions", "policy_num_mix",
        "do_pose_err", "do_mse_err", "do_grab_norm", "single_grab"
    ]), common_params['velact'])

    # e.g. for policy to use
    lmp_names_and_sizes.policy_out_norm_names = policy_out_norm_names

    return policy_loss_fn


def gcbc_update_batch_names(common_params, prms, lmp_names_and_sizes):
    """
    Update the batch_names for dataset to know what to load.
    """
    env_spec_params = common_params["env_spec/params"]

    SAVE_NORMALIZATION_NAMES = lmp_names_and_sizes["SAVE_NORMALIZATION_NAMES"]
    extra_action_names = lmp_names_and_sizes < ["action_names", "waypoint_names"]
    extra_action_names = list(v for x in extra_action_names.leaf_values() for v in x)

    # setting common_params.batch_names_to_get, which a dataset can use to know what limited set of keys to extract.
    common_params["batch_names_to_get"] = list(
        set(SAVE_NORMALIZATION_NAMES + ["policy_type", "action", "policy_switch"] + extra_action_names))

    if prms["use_final_goal"]:
        common_params["batch_names_to_get"] = common_params["batch_names_to_get"] + env_spec_params.final_names

        if prms["ignore_existing_goal"] and common_params.has_leaf_key("dataset_train/load_ignore_prefixes"):
            common_params["dataset_train/load_ignore_prefixes"].append("goal/")

    if "policy_name" in common_params["batch_names_to_get"]:
        common_params["batch_names_to_get"].remove("policy_name")  # this should not be retrieved every time

    if "image" in env_spec_params.observation_names:
        common_params["batch_names_to_get"].append("image")  # careful with this

    if "ego_image" in env_spec_params.observation_names:
        common_params["batch_names_to_get"].append("ego_image")

    if "mode" in env_spec_params.observation_names:
        common_params['batch_names_to_get'].append("mode")

    if "real" in env_spec_params.param_names:
        common_params['batch_names_to_get'].append("real")

    # target is like long term action
    target_names = [tn for tn in env_spec_params.action_names if tn.startswith('target/')]

    common_params['batch_names_to_get'].extend(target_names)
