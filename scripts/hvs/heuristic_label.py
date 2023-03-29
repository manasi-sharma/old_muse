import os

import numpy as np

from configs.helpers import get_script_parser, load_base_config
from muse.experiments import logger
from muse.experiments.file_manager import FileManager, ExperimentFileManager
from muse.utils.general_utils import exit_on_ctrl_c
from muse.utils.input_utils import query_string_from_set


def velocity_heuristic(ep_idx, args, inputs, outputs) -> np.ndarray:
    ep_len = len(outputs.done)
    is_dense = np.ones(ep_len, dtype=bool)
    for vk, thresh in zip(args.velocity_keys, args.velocity_thresh):
        vel = inputs[vk]
        is_dense &= np.linalg.norm(vel, axis=-1) < thresh

    # print("is_dense: \n", is_dense, "\n--------")

    prev_is_dense = np.concatenate([[False], is_dense[:-1]])
    start_of_dense = (is_dense & ~prev_is_dense).nonzero()[0]
    end_of_dense_plus_one = (~is_dense & prev_is_dense).nonzero()[0]

    # special case where we end in a dense segment.
    if len(end_of_dense_plus_one) < len(start_of_dense):
        end_of_dense_plus_one = np.append(end_of_dense_plus_one, ep_len)

    # remove short segments
    long_enough = end_of_dense_plus_one - start_of_dense >= args.min_dense_len
    start_of_dense = start_of_dense[long_enough]
    end_of_dense_plus_one = end_of_dense_plus_one[long_enough]

    click_state = np.zeros(ep_len, dtype=bool)
    if len(start_of_dense) == 0:
        # no dense segments
        logger.warn(f'[Ep={ep_idx}] No dense segments found!')
        # label fixed width segments
        click_state[np.arange(0, ep_len, args.wp_horizon)] = True
        return click_state.reshape(-1, 1)

    last_end_plus_one = 0
    for i, s, e in zip(range(len(start_of_dense)), start_of_dense, end_of_dense_plus_one):
        assert s < e, [s, e, start_of_dense, end_of_dense_plus_one]
        click_state[s:e] = True

        if s > last_end_plus_one + args.wp_horizon:
            # sparse segment here, starting WP_HORIZON away from last end
            click_state[np.arange(last_end_plus_one + args.wp_horizon, s, args.wp_horizon)] = True

        last_end_plus_one = e

        # adding the last segment optionally, starting WP_HORIZON away from this end.
        if i == len(start_of_dense) - 1 and e + args.wp_horizon < ep_len:
            click_state[np.arange(e + args.wp_horizon, ep_len, args.wp_horizon)] = True

    # print("click_state: \n", click_state, "\n--------")

    # print('---------------------------------------------------------------------')
    return click_state.reshape(-1, 1)


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
    parser.add_argument('--velocity_keys', type=str, nargs='*', default=['robot0_eef_vel_lin', 'robot0_eef_vel_ang'])
    parser.add_argument('--velocity_thresh', type=float, nargs='*', default=[0.12, 0.22])
    parser.add_argument('--wp_horizon', type=int, default=20)
    parser.add_argument('--min_dense_len', type=int, default=5)
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
    click_state_key = local_args.click_state_key

    # env spec used for loading
    env_spec = params.env_spec.cls(params.env_spec.leaf_copy())

    if click_state_key not in (params.env_spec.observation_names + params.env_spec.action_names):
        # add click state key as an action
        logger.debug(f"Adding click key {click_state_key} to env_spec.observation_names!")
        params.env_spec.observation_names.append(click_state_key)

    # env spec used for saving (with added click key)
    env_spec_out = params.env_spec.cls(params.env_spec.leaf_copy())

    # dataset input
    params[local_args.dataset_in_group].file = local_args.file
    dataset_input = params[local_args.dataset_in_group].cls(params[local_args.dataset_in_group], env_spec, file_manager)

    # dataset output
    params[local_args.dataset_out_group].output_file = local_args.output_file
    params[local_args.dataset_out_group].frozen = False
    dataset_out = params[local_args.dataset_out_group].cls(params[local_args.dataset_out_group], env_spec_out, file_manager)

    for ep_i in range(dataset_input.get_num_episodes()):
        # (N x ...) where N is episode length
        inputs, outputs = dataset_input.get_episode(ep_i, None, split=True)
        length = len(outputs["done"])

        inputs[click_state_key] = velocity_heuristic(ep_i, local_args, inputs, outputs)

        dataset_out.add_episode(inputs, outputs)

    do_save = query_string_from_set(f'Save to {dataset_out.save_dir}? (y/n)', ['y', 'n']) == 'y'
    if do_save:
        dataset_out.save()
