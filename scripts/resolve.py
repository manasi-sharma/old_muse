import argparse

from configs.helpers import load_base_config, get_script_parser

if __name__ == '__main__':
    # things we can use from command line

    parser = get_script_parser()
    parser.add_argument('config', type=str, help="common params for all modules.")
    local_args, unknown = parser.parse_known_args()

    params, root = load_base_config(local_args.config, unknown)
    exp_name = root.get_exp_name()

    print("----------------------- EXPERIMENT -----------------------")
    print(f"exp_name = {exp_name}")
    # print("common_params = " + params.pprint(ret_string=True, str_max_len=None))
    print("params=" + params.pprint(ret_string=True))
