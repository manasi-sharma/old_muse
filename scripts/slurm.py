"""
Launching arbitrary command through sbatch. Does not support multiple jobs yet.

Usage:
    python scripts/slurm.py <slurm_args> --- <command_args>

Example:
    Single command, single job:
        python scripts/slurm.py -c iliad_gpu --job-name learn_gmm --- python scripts/tests/test_learn_gmm.py
    [TODO] sweep multiple at once on different machines (multiple commands, multiple jobs)

    Multiple commands, *same* job:
        python scripts/slurm.py -p -c iliad_gpu --job-name learn_gmm_and_smi --- 
               python scripts/tests/test_learn_gmm.py --- nvidia-smi

Example with sweep (parallel):
    python scripts/slurm.py -p -s -c iliad_gpu --job-name learn_gmm_and_smi --- 
           python scripts/tests/test_learn_gmm.py -w [[[ 0 0.1 0.2 ]]]

"""
import argparse
import subprocess
import sys
import tempfile
from argparse import ArgumentParser

from muse.experiments import logger
from muse.utils.slurm_utils import SLURM_ARGS, write_slurm_header, SLURM_CFGS, update_parser_with_cfg

BASE_CONDA_ENV = "sbrl"
DELIM = "---"
SWEEP_DELIM_START = "[[["
SWEEP_DELIM_END = "]]]"


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, choices=SLURM_CFGS.keys(), default=None)
    parser.add_argument("-e", "--cenv", type=str, default=BASE_CONDA_ENV, help="Conda environment")
    parser.add_argument("-p", "--parallel", action='store_true', 
                        help="if parallel, will read multiple delimiters as parallel jobs.")
    parser.add_argument("-s", "--sweep", action='store_true', help="if sweep, will look for SWEEP_DELIM")
    parser.add_argument("-d", "--delay", type=str, default=None, help="if sweep & parallel, will add a delay using sleep <this>")
    parser.add_argument("-v", "--vars", type=str, default=None,
                        help="Extra environment arguments for new shell, specified as 'arg:val,...'")
    # parser.add_argument("--jobs-per-instance", default=1, type=int, help="Number of jobs to run in one slurm sbatch.")

    # Add Slurm Arguments
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    all_args = list(sys.argv)
    mid_idx = None
    for i, a in enumerate(all_args):
        if a == DELIM:
            mid_idx = i
            break

    assert mid_idx is not None, f"Format is <slurm args> {DELIM} <command args>"
    assert mid_idx < len(all_args) - 1, f"Delimiter \"{DELIM}\" cannot be at the end! Must specify some command"

    slurm_arg_ls = all_args[1:mid_idx]
    cmd_arg_ls = all_args[mid_idx+1:]

    # parse just the slurm args.
    slurm_args = parser.parse_args(args=slurm_arg_ls)
    if slurm_args.cfg is not None:
        # updates the defaults, reparses
        update_parser_with_cfg(parser, slurm_args.cfg)
        slurm_args = parser.parse_args(args=slurm_arg_ls)

    # Call python subprocess to launch the slurm job.
    procs = []

    # sweeping specific variables, will duplicate strings separated by DELIM
    if slurm_args.sweep:
        assert DELIM not in cmd_arg_ls, "Delimiter is present in command args, but you are using sweep mode!!"
        all_commands = []
        all_sweep_points = []
        cmd_chunks = []
        last_end = 0
        # print(cmd_arg_ls)
        for i, arg in enumerate(cmd_arg_ls):
            if arg == SWEEP_DELIM_START:
                end = None
                for j, narg in enumerate(cmd_arg_ls[i + 1:]):
                    if narg == SWEEP_DELIM_END:
                        end = i + 1 + j
                        break
                assert end is not None and end > i + 1, "Sweep delimiter is not closed or is empty!"
                num_commands = (end - i - 1)
                to_sweep = cmd_arg_ls[i + 1:end]
                if num_commands == 1:
                    logger.warn(
                        f"Using sweep from arg ({i} -> {end}) but only 1 sweep value was presented ({to_sweep})")

                # num_commands, the exact commands, start_idx, end_idx
                all_sweep_points.append((num_commands, to_sweep, i, end))
                cmd_chunks.append(cmd_arg_ls[last_end:i])
                last_end = end + 1  # for next chunk
        # final chunk is what remains
        cmd_chunks.append(cmd_arg_ls[last_end:])

        assert len(all_sweep_points) > 0, "Must provide at least one sweep since --sweep is enabled!"
        assert len(set(
            a[0] for a in all_sweep_points)) == 1, "Multiple sweep points specified but all have different lengths!!"

        num_commands = all_sweep_points[0][0]
        for cmd_idx in range(num_commands):
            this_cmd = cmd_chunks[0]
            for (_, cmds, _, _), next_chunk in zip(all_sweep_points, cmd_chunks[1:]):
                this_cmd = this_cmd + [cmds[cmd_idx]] + next_chunk  # add to list
            if cmd_idx < num_commands - 1:
                this_cmd = this_cmd + [DELIM]  # add delim between commands, for all but the last one.
            all_commands.extend(this_cmd)

        cmd_arg_ls = all_commands  # replace with unrolled commands

    # this is the command string we will use.
    all_command_str = ' '.join(cmd_arg_ls)

    # stack multiple commands (in parallel, same job) by looking for inner delimiters
    if slurm_args.parallel:
        all_commands = [s.strip() for s in all_command_str.split(DELIM)]
        for i in range(len(all_commands)):
            all_commands[i] = all_commands[i] + " &\n"  # run in background

        # add delay in between commands
        if slurm_args.delay is not None:
            new_commands = []
            for c in all_commands[:-1]:
                new_commands.append(c)
                new_commands.append(f"sleep {slurm_args.delay} \n")
            new_commands.append(all_commands[-1])
            all_commands = list(new_commands)

        all_command_str = ''.join(all_commands) + "wait"  # wait on all of them

    # stack commands by separating by process (different jobs)
    elif DELIM in all_command_str:
        all_command_str = [s.strip() for s in all_command_str.split(DELIM)]

    if not slurm_args.parallel:
        assert slurm_args.delay is None, "cannot add delay for non-parallel job"

    # each element gets its own job
    if isinstance(all_command_str, str):
        all_command_str = [all_command_str]

    init_slurm_args = slurm_args
    for jnum, command_str in enumerate(all_command_str):
        if len(all_command_str) > 1:
            slurm_args = argparse.Namespace(**vars(init_slurm_args))  # deep copy
            slurm_args.job_name = slurm_args.job_name + f"_job{jnum}"  # suffix to make it a unique name
        _, slurm_file = tempfile.mkstemp(text=True, prefix=f'job[{slurm_args.job_name}]', suffix='.sh')
        logger.debug(f"Launching job {jnum} with slurm file: {slurm_file}")

        with open(slurm_file, 'w+') as f:
            # will write the header based on slurm_args
            write_slurm_header(f, slurm_args, env=slurm_args.cenv)
            if slurm_args.vars is not None:
                all_vars = slurm_args.vars.split(',')
                for av in all_vars:
                    a, v = av.split(":")  # will throw an error if not arg:val
                    f.write(f"export {a}={v}\n")

            # Now that we have written the header we can launch the jobs.
            command_str = command_str + ' \n'
            f.write(command_str)

            # proc = subprocess.Popen(['cat', slurm_file])
            proc = subprocess.Popen(['sbatch', slurm_file])
            procs.append(proc)

    exit_codes = [p.wait() for p in procs]
