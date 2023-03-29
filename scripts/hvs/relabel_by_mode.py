"""
For datasets with actions of different modalities, process the raw teleop format into the training format.

This entails...
1. Parsing into chunks of type (mode 0, mode 1) from binary "click" signals in the data.
2. For each state, labeling its true action for both modalities: "action" (e.g., velocity) & "target/" (e.g. position)
    a. we keep the old target/ in raw/target/, and then relabel action according to the env dynamics.
3. Creating a corresponding version of the dataset using knowledge of underlying robot dynamics
        for the fixed mode data, along with some extra metadata denoting which are augmented and which are "real".


MODE = 0: SPARSE (waypoint)
MODE = 1: DENSE (action)

REAL = False:
"""

import os

import numpy as np
from attrdict import AttrDict

from configs.helpers import get_script_parser, load_base_config
from muse.envs.mode_param_spec import BiModalParamEnvSpec
from muse.envs.param_spec import ParamEnvSpec
from muse.experiments import logger
from muse.experiments.file_manager import ExperimentFileManager, FileManager
from muse.utils.file_utils import file_path_with_default_dir
from muse.utils.input_utils import query_string_from_set
from muse.utils.general_utils import exit_on_ctrl_c
from muse.utils.torch_utils import combine_then_concatenate


def postproc_clicks(clicks):
    """
    Post-processes clicks that came from raw user data.
    clicks is (N,)
    """

    # check for two clicks, this should be treated as a single click
    p2_clicks = np.concatenate([[False, False], click_state])[:-2]
    p1_clicks = np.concatenate([[False], click_state])[:-1]
    n1_clicks = np.concatenate([click_state, [False]])[1:]  #

    # select for [0, 1, *1*, 0] pattern, replace second 1 with 0
    clicks = np.where(~p2_clicks & p1_clicks & clicks & ~n1_clicks, False, clicks)

    # TODO other relevant post-proc

    return clicks.astype(bool)


if __name__ == '__main__':

    parser = get_script_parser()
    parser.add_argument('config', type=str)
    parser.add_argument('--file', type=str, nargs="+", required=True, help="1 or more input files")
    parser.add_argument('--output_file', type=str, required=True)
    # batch_size within episode
    # if horizon > 1, eval len will be different than data len, pad eval with zeros
    parser.add_argument('--click_state_key', type=str, default='click_state')
    parser.add_argument('--dataset_in_group', type=str, default='dataset_in')
    parser.add_argument('--dataset_out_group', type=str, default='dataset_out')
    parser.add_argument('--populate_mode0_actions', action='store_true')
    parser.add_argument('--populate_mode0_gripper', action='store_true')
    parser.add_argument('--relabel_states', action='store_true')
    parser.add_argument('--no_change', action='store_true', help='just relabels mode, optionally populates actions')
    parser.add_argument('--no_mode1_change', action='store_true', help='skips mode1 action relabel.')
    parser.add_argument('--use_dense_goal', action='store_true',
                        help='during dense period, mode0 action will be the goal, not the next state')
    parser.add_argument('--fill_n_mode0_action_dims', type=int, default=0)
    parser.add_argument('--n_intermediate_wp', type=int, default=0)
    parser.add_argument('--mode0_action_skip', type=int, default=1,
                        help='how many states in the future is the default waypoint?')
    local_args, unknown = parser.parse_known_args()

    """ Parsing command line inputs """

    # load the config
    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    # allow for a default experiment if the current one does not exist.
    if os.path.exists(os.path.join(FileManager.base_dir, 'experiments', exp_name)):
        file_manager = ExperimentFileManager(exp_name, is_continue=True)
    else:
        logger.warn(f"Experiment: 'experiments/{exp_name}' does not exist, using 'experiments/test' instead")
        file_manager = ExperimentFileManager('test', is_continue=True)

    exit_on_ctrl_c()
    mode_key = 'mode'  # 0 if velact, 1 for posact
    real_key = 'real'  # 0 if fake data, 1 if real data
    click_state_key = local_args.click_state_key
    fill_n_ac_dim = local_args.fill_n_mode0_action_dims

    assert not (local_args.no_change and local_args.relabel_states), "No change conflicts w/ relabeling of states!"
    if local_args.no_change:
        logger.warn("[NO CHANGE] Will not change any actions!")

    if local_args.no_mode1_change:
        logger.warn("[NO MODE1 CHANGE] Will not change any mode=1 actions!")

    assert not local_args.no_mode1_change or not local_args.no_change, \
        "no_change is mutually exclusive from no_mode1_change. choose one."

    assert issubclass(params.env_spec.cls, BiModalParamEnvSpec), "Can only relabel by modes with bimodal env spec!"

    """ Computing env spec params """

    if click_state_key not in (params.env_spec.observation_names + params.env_spec.action_names):
        # add click state key as an action
        logger.debug(f"Adding click key {click_state_key} to env_spec.observation_names!")
        params.env_spec.observation_names.append(click_state_key)

    # remove mode0 keys and
    for k in [mode_key] + params.env_spec.mode0_action_names:
        if k in params.env_spec.observation_names:
            # remove mode key
            logger.debug(f"Removing key {k} from env_spec.observation_names!")
            params.env_spec.observation_names.remove(k)
        elif k in params.env_spec.action_names:
            # remove mode key
            logger.debug(f"Removing key {k} from env_spec.action_names!")
            params.env_spec.action_names.remove(k)

    # create a param env spec, since data we are loading will not have certain keys
    env_spec = ParamEnvSpec(params.env_spec.leaf_copy())

    logger.debug(f"mode0 names: {params.env_spec.mode0_action_names}")
    logger.debug(f"mode1 names: {params.env_spec.mode1_action_names}")

    """ Computing dataset params (input and output) """

    params[local_args.dataset_in_group].file = local_args.file
    dataset_input = params[local_args.dataset_in_group].cls(params[local_args.dataset_in_group], env_spec, file_manager)
    all_names = env_spec.all_names + ['done', 'rollout_timestep']
    logger.debug(f"ALL NAMES: {all_names}")

    # Compute the spec to save (output spec)
    env_spec_out_prms = params.env_spec.leaf_copy()
    env_spec_out_prms.observation_names.extend([mode_key, 'real'])
    if local_args.populate_mode0_actions:
        not_present = list(set(params.env_spec.mode0_action_names).difference(env_spec_out_prms.action_names))
        env_spec_out_prms.action_names.extend(not_present)
        logger.debug(f'Will use base mode0 action skip of {local_args.mode0_action_skip}')

    if local_args.populate_mode0_gripper and 'target/gripper' not in env_spec_out_prms.action_names:
        env_spec_out_prms.action_names.append('target/gripper')
    env_spec_out_prms.action_names.extend(
        [f'raw/{nm}' for nm in (params.env_spec.mode0_action_names + params.env_spec.mode1_action_names)]
    )
    env_spec_out = BiModalParamEnvSpec(env_spec_out_prms)
    output_file = file_path_with_default_dir(local_args.output_file, file_manager.data_dir, expand_user=True)
    params[local_args.dataset_out_group].file = 'none.npz'  # no load
    params[local_args.dataset_out_group].output_file = output_file
    params[local_args.dataset_out_group].frozen = False
    dataset_out = params[local_args.dataset_out_group].cls(params[local_args.dataset_out_group], env_spec_out, file_manager)

    """ Go through each episode, and relabel (either with state rollout, or just action relabel) """

    eval_datadict = []

    if local_args.relabel_states:
        logger.debug("Will relabel states with imagined rollouts...")

    for i in range(dataset_input.get_num_episodes()):
        # (N x ...) where N is episode length
        inputs, outputs = dataset_input.get_episode(i, all_names, split=True)
        length = len(outputs["done"])

        """ Parse clicks into waypoints (single click) and dense periods (sustained clicks) """
        click_state = inputs[click_state_key]  # N x 1

        click_state = click_state.reshape(length) > 0

        # check that there are clicks labeled.
        if not np.any(click_state) and not local_args.no_change:
            logger.warn(f"Episode {i} has all zero click state! Skipping")
            continue

        click_state = postproc_clicks(click_state)

        # always mark the last state as a click (either will be waypoint or already true if part of dense)
        click_state[-1] = True

        prev_click_state = np.concatenate([[False], click_state])[:-1]
        next_click_state = np.concatenate([click_state, [False]])[1:]  #
        # next_next_click_state = np.concatenate([click_state, [False, False]])[2:]  #

        is_dense = click_state & (prev_click_state | next_click_state)  # partially or fully surrounded click
        is_isolated = ~prev_click_state & click_state & ~next_click_state
        first_is_dense = click_state & ~prev_click_state & next_click_state
        # next_is_dense = ~click_state & next_click_state & next_next_click_state
        is_last_dense = click_state & prev_click_state & ~next_click_state
        predense_or_iso_click = is_isolated | first_is_dense | is_last_dense  # first_is_dense used to be next_is_dense.

        # prestart_or_last_click = (click_state & ~next_click_state) | (click_state & ~prev_click_state)
        relabel_target_idxs = predense_or_iso_click.nonzero()[0]  # points to extract target states from

        state = combine_then_concatenate(inputs, env_spec_out.dynamics_state_names, dim=1)

        if local_args.relabel_states:
            """
            Run rollouts for each waypoint chunk (non dense), which will likely change the sequence length.
            """

            assert not local_args.n_intermediate_wp > 0

            inputs_chunked = inputs.leaf_apply(lambda arr: np.split(arr, relabel_target_idxs[:-1] + 1))
            outputs_chunked = outputs.leaf_apply(lambda arr: np.split(arr, relabel_target_idxs[:-1] + 1))

            is_dense_chunked = np.split(is_dense, relabel_target_idxs[:-1] + 1)
            chunk_is_dense = np.array(
                [idc[0] for idc in is_dense_chunked])  # True or False for each chunk, if its dense or not

            for c in range(len(chunk_is_dense)):
                if not chunk_is_dense[c]:
                    old_inputs_c = inputs_chunked.leaf_apply(lambda ls: ls[c])
                    first_in = old_inputs_c.leaf_apply(lambda arr: arr[0])  # these will be repeated til the end
                    last_in = old_inputs_c.leaf_apply(lambda arr: arr[-1])  # these will be the target action
                    first_out = outputs_chunked.leaf_apply(lambda ls: ls[c][0])  # these will be repeated til the end
                    last_out = outputs_chunked.leaf_apply(lambda ls: ls[c][-1])  # these will be repeated til the end

                    # relabel new states using rolled out dynamics
                    start = combine_then_concatenate(first_in, env_spec_out.dynamics_state_names, dim=0)
                    end_mode0p1_dc = last_in > (env_spec_out.mode0_action_names + env_spec_out.mode1_action_names)  # last
                    end = combine_then_concatenate(last_in, env_spec_out.dynamics_state_names, dim=0)
                    default_mode0_dc = env_spec_out.parse_view_from_concatenated_flat(end, env_spec_out.mode0_action_names)

                    reached = np.array([False])
                    last_reached = False
                    all_ins = []
                    all_outs = []
                    state = start
                    while not last_reached:  # one extra step
                        last_reached = reached.item()
                        # gets us the transition and also the mode 0 action
                        mode1_ac, next_state, reached = env_spec_out.wp_dynamics_fn(state, end)
                        mode1_ac_dc = env_spec_out.parse_view_from_concatenated_flat(mode1_ac, env_spec_out.mode1_action_names)
                        state_dc = env_spec_out.parse_view_from_concatenated_flat(state, env_spec_out.dynamics_state_names)
                        all_ins.append(first_in & state_dc)
                        if last_reached:
                            all_ins[-1].combine(end_mode0p1_dc)  # last one gets the correct next action(s).
                        else:
                            all_ins[-1].combine(default_mode0_dc & mode1_ac_dc)
                        all_outs.append(first_out)
                        state = next_state

                    new_inputs_c = AttrDict.leaf_combine_and_apply(all_ins, lambda vs: np.stack(vs, axis=0))
                    new_outputs_c = AttrDict.leaf_combine_and_apply(all_outs, lambda vs: np.stack(vs, axis=0))

                    # all done should be set to False
                    new_outputs_c["done"][:] = False
                    new_outputs_c["done"][-1] = last_out["done"]

                    # is_dense assignment (this segment is not dense)
                    is_dense_chunked[c] = np.zeros(len(all_ins), dtype=bool)

                    # assignment to c
                    for key in inputs_chunked.leaf_keys():
                        inputs_chunked[key][c] = new_inputs_c[key]  # put it in the list
                    for key in outputs_chunked.leaf_keys():
                        outputs_chunked[key][c] = new_outputs_c[key]  # put it in the list

            # TO be added to dataset
            inputs = inputs_chunked.leaf_apply(lambda vs: np.concatenate(vs, axis=0))
            outputs = outputs_chunked.leaf_apply(lambda vs: np.concatenate(vs, axis=0))

            # duplicate, these aren't really the raw actions b/c the raw actions don't make sense anymore
            inputs.raw = inputs > (env_spec_out.mode0_action_names + env_spec_out.mode1_action_names)

            is_dense = np.concatenate(is_dense_chunked, axis=0)

        else:
            """
            Just relabel the action(s) to be correct, without changing the state. 
            This violates some history assumptions.
            """
            if local_args.n_intermediate_wp > 0:
                # we are gonna label sparse periods with extra segments
                new_relabel = []
                rlts = np.concatenate([[0], relabel_target_idxs])
                for i, idx, next_idx in zip(range(len(rlts) - 1), rlts[:-1], rlts[1:]):
                    # sparse segment
                    if not is_dense[idx]:
                        # [5 , 9], 1 -> [5, 7, 10], spacing = (9 - 5 + 1) / 2
                        spacing = (next_idx - idx + 1) / (local_args.n_intermediate_wp + 1)
                        # not divisible enough for this segment... set it to every 1.
                        n_inter = local_args.n_intermediate_wp
                        if spacing < 1:
                            n_inter = next_idx - idx
                            spacing = 1
                        for j in range(n_inter):
                            new_relabel.append(idx + int((j + 1) * spacing))
                    new_relabel.append(next_idx)

                relabel_target_idxs = np.asarray(new_relabel)

            is_dense_idxs = is_dense.nonzero()[0]
            repeats = np.diff(relabel_target_idxs, prepend=0)  # used to be prepend=-1
            # print(is_dense_idxs)

            if local_args.populate_mode0_actions:
                # populate with the "next" state
                assert local_args.mode0_action_skip >= 1
                next_state = np.concatenate(
                    [state[local_args.mode0_action_skip:]] + [state[-1:]] * local_args.mode0_action_skip, axis=0)
                # only used to populate the dense regions.
                mode0_action = next_state.copy()
                inputs.combine(env_spec_out.parse_view_from_concatenated_flat(mode0_action, env_spec_out.mode0_action_names))
            else:
                # mode1_action = combine_then_concatenate(inputs, mode1_action_keys, dim=1)
                mode0_action = combine_then_concatenate(inputs, env_spec_out.mode0_action_names, dim=1)

            if local_args.populate_mode0_gripper:
                assert not inputs.has_leaf_key('target/gripper'), inputs.list_leaf_keys()
                mode1_action = combine_then_concatenate(inputs, env_spec_out.mode1_action_names, dim=1)
                inputs['target/gripper'] = mode1_action[..., -1:]

            # old actions become raw/..., new ones get copied in
            inputs.raw = inputs > (env_spec_out.mode0_action_names + env_spec_out.mode1_action_names)

            # this is the part that relabels mode0 actions.
            if not local_args.no_change:
                # relabel target/position for non-dense periods
                relabel_mode0_action = np.repeat(state[relabel_target_idxs], repeats, axis=0)

                # final state will not have an action, we just populate that with the last action.
                relabel_mode0_action = np.concatenate([relabel_mode0_action, relabel_mode0_action[-1:]], axis=0)
                assert len(relabel_mode0_action) == length, [relabel_mode0_action.shape, length]

                # otherwise, use the last state of dense region as the mode0 label during dense periods
                if not local_args.use_dense_goal:
                    # copy over dense actions back.
                    relabel_mode0_action[is_dense_idxs] = mode0_action[is_dense_idxs]

                inputs.combine(env_spec_out.parse_view_from_concatenated_flat(relabel_mode0_action,
                                                                          env_spec_out.mode0_action_names))

                if not local_args.no_mode1_change:
                    # zero-step dynamics to get mode0 actions (e.g. via P-control with clipping for nav2d)
                    # should be consistent for dense periods (provided dynamics are the same.
                    relabel_mode1_action, _, _ = env_spec_out.wp_dynamics_fn(state, relabel_mode0_action)

                    # copy over mode1 actions during previous portions.
                    mode1_action = combine_then_concatenate(inputs, env_spec_out.mode1_action_names, dim=1)
                    relabel_mode1_action[is_dense_idxs] = mode1_action[is_dense_idxs]
                    # relabel_mode1_action[relabel_target_idxs] = mode1_action[relabel_target_idxs]

                    if fill_n_ac_dim > 0:
                        assert fill_n_ac_dim < mode1_action.shape[-1], [fill_n_ac_dim, mode1_action.shape]
                        assert mode1_action.shape[-1] == relabel_mode1_action.shape[-1]
                        relabel_mode1_action[..., fill_n_ac_dim:] = mode1_action[..., fill_n_ac_dim:]

                    inputs.combine(
                        env_spec_out.parse_view_from_concatenated_flat(relabel_mode1_action,
                                                                   env_spec_out.mode1_action_names)
                    )

        # add mode information
        inputs[mode_key] = is_dense[:, None].astype(np.uint8)
        inputs[real_key] = (not local_args.relabel_states) * np.ones_like(inputs[mode_key])

        dataset_out.add_episode(inputs, outputs)

    logger.debug(f"Old dataset length: {len(dataset_input)}, num eps: {dataset_input.get_num_episodes()}")
    logger.debug(f"New dataset length: {len(dataset_out)}, num eps: {dataset_out.get_num_episodes()}")

    do_save = query_string_from_set(f'Save to {dataset_out.save_dir}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        dataset_out.save()
