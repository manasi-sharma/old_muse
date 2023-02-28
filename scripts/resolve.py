import argparse

from configs.helpers import load_base_config

if __name__ == '__main__':
    # things we can use from command line

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('config', type=str, help="common params for all modules.")
    local_args, unknown = parser.parse_known_args()

    params = load_base_config(local_args.config, unknown)

    print("----------------------- EXPERIMENT -----------------------")
    print(f"exp_name = TODO")
    # print("common_params = " + params.pprint(ret_string=True, str_max_len=None))
    print("params=" + params.pprint(ret_string=True))
