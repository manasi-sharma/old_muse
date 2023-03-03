# things we can use from command line
import argparse

import torch
from attrdict import AttrDict

from configs.helpers import load_base_config, get_script_parser
from muse.experiments import logger
from muse.experiments.file_manager import FileManager
from muse.models.basic_model import BasicModel
from muse.utils.general_utils import timeit
from muse.utils.param_utils import SequentialParams, build_mlp_param_list
from muse.utils.torch_utils import concatenate, combine_after_dim

if __name__ == '__main__':
    parser = get_script_parser()
    parser.add_argument('config', type=str, help="Config with env_spec and dataset_train")
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default="dataset_train")
    parser.add_argument('--torch_device', type=str, default="cuda")
    parser.add_argument('--do_example_training', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--log_every_n', type=int, default=1000)

    local_args, unknown = parser.parse_known_args()
    # this defines the required command line groups, and the defaults
    # if in this list, we look for it

    params, _ = load_base_config(local_args.config, unknown)

    file_manager = FileManager()

    env_spec = params.env_spec.cls(params.env_spec)

    if local_args.file is not None:
        params[local_args.dataset_name].file = local_args.file
    dataset_train = params[local_args.dataset_name].cls(params[local_args.dataset_name], env_spec, file_manager)

    sampler = dataset_train.sampler

    logger.debug("Dataset length:", len(dataset_train))

    res = dataset_train.get_batch(torch_device=local_args.torch_device)
    init_ins, init_outs = res[:2]

    in_names = init_ins.list_leaf_keys()
    in_size = env_spec.dim(in_names)
    logger.info(f"Input Keys Loaded: {init_ins.list_leaf_keys()}, Outputs: {init_outs.list_leaf_keys()}")
    logger.debug(f"Input Shapes: {init_ins.leaf_apply(lambda arr: arr.shape).pprint(ret_string=True)}")

    if local_args.do_example_training:
        # example model that takes all inputs.
        model = BasicModel(AttrDict(
            normalize_inputs=True,
            normalization_inputs=in_names,
            model_inputs=in_names,
            model_output="sample_out",
            device=local_args.torch_device,
            network=SequentialParams(build_mlp_param_list(in_size, [400, 400, 200, in_size // 4, 200, 400, in_size])),
        ), env_spec, dataset_train)

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # very simple auto-encoder with a dataset.
    for j in range(10):
        timeit.reset()

        with timeit("loop"):
            for i in range(local_args.log_every_n):
                with timeit("get_batch"):
                    with torch.no_grad():
                        # idxs = np.random.choice(len(dataset_train._merged_datadict), dataset_train.batch_size * dataset_train.horizon)
                        # data = dataset_train._merged_datadict[idxs]
                        # batch = torch.from_numpy(data).to(device="cuda")
                        res = dataset_train.get_batch(indices=sampler.get_indices(), torch_device=local_args.torch_device)
                        inputs, outputs = res[:2]
                if local_args.do_example_training:
                    with timeit("train"):
                        model.train()
                        inputs_flat = inputs.leaf_apply(lambda arr: combine_after_dim(arr, 2))
                        model_outputs = model.forward(inputs_flat)
                        input_arr = concatenate(inputs_flat, in_names, dim=-1)
                        loss = ((model_outputs >> "sample_out") - input_arr).abs().mean()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

        logger.info(f"[{(j+1) * local_args.log_every_n}] timing...")
        print(timeit)

