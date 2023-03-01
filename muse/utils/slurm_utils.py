"""
Utils for slurm job launching. See scripts/slurm.py for example usage of some of these fns.

A lot of the structure here taken from here: https://github.com/jhejna/research-lightning/blob/cede0315458cc5fbf54150e4dd82af5fc336b6d3/tools/utils.py
"""

import os
from argparse import ArgumentParser

from muse.experiments.file_manager import FileManager
from attrdict import AttrDict

SLURM_LOG_DEFAULT = FileManager.logs_dir
SLURM_ARGS = {
    "account": {"type": str, "default": 'iliad'},
    "partition": {"type": str, "default": 'iliad'},
    "time": {"type": str, "default": "7-0"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "default": 2},
    "gpus": {"type": str, "default": "0"},
    "mem": {"type": str, "default": "10G"},
    "output": {"type": str, "default": SLURM_LOG_DEFAULT},
    "error": {"type": str, "default": SLURM_LOG_DEFAULT},
    "job-name": {"type": str, "required": True},
    "exclude": {"type": str, "required": False, "default": None},
    "nodelist": {"type": str, "required": False, "default": None}
}

SLURM_NAME_OVERRIDES = {
    "gpus": "gres",
    "cpus": "cpus-per-task"
}


def update_parser_with_cfg(parser: ArgumentParser, cfg: str):
    # sets the defaults for parser
    config = SLURM_CFGS[cfg]
    parser.set_defaults(**config.as_dict())


def write_slurm_header(f, args, env=None):

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(args.error):
        os.makedirs(args.error)

    args.output = os.path.join(args.output, args.job_name + "_%A.out")
    args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus) if args.gpus is not None else args.gpus

    NL = '\n'
    f.write("#!/bin/bash" + NL)
    f.write(NL)
    for arg_name in SLURM_ARGS.keys():
        arg_value = vars(args)[arg_name.replace('-', '_')]
        if arg_name in SLURM_NAME_OVERRIDES:
            arg_name = SLURM_NAME_OVERRIDES[arg_name]
        if arg_value is not None:
            f.write("#SBATCH --" + arg_name + "=" + str(arg_value) + NL)

    f.write(NL)
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES' + NL)
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR' + NL)
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR' + NL)
    f.write(NL)
    f.write(". ~/.bashrc" + NL)
    f.write(f"cd {FileManager.base_dir}" + NL)
    if env is not None:
        f.write(f"conda activate {env}")
    f.write(NL)


iliad = AttrDict(partition='iliad')

iliad_min = iliad & AttrDict(gpus=0, cpus=1, mem="10G")

iliad_tbrd = iliad_min & AttrDict(time="21-0")

iliad_cpu32 = iliad & AttrDict(gpus=0, cpus=32, mem="20G")

iliad_cpu64 = iliad & AttrDict(gpus=0, cpus=64, mem="40G")

iliad_gpu = iliad & AttrDict(gpus=1, cpus=2, mem="20G")

iliad_gpu56 = iliad_gpu & AttrDict(exclude="iliad[1-4]")

# base configurations for SLURM, to then customize upon
SLURM_CFGS = {
    'iliad': iliad,
    'iliad_min': iliad_min,
    'iliad_tbrd': iliad_tbrd,
    'iliad_gpu': iliad_gpu,
    'iliad_gpu56': iliad_gpu56,
    'iliad_cpu32': iliad_cpu32,
    'iliad_cpu64': iliad_cpu64,
}
