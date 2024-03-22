# slurm job runner script
# adapted from: https://gist.github.com/willwhitney/e1509c86522896c6930d2fe9ea49a522

import os
import sys
import itertools
import argparse
from pathlib import Path
import copy
import importlib
import click

argparser = argparse.ArgumentParser(
    description="Generate and optionally submit slurm jobs. runner_config.py is the configuration file.")
argparser.add_argument("--job_name_base", default="meta-word",
                       help="The base name of jobs. All job names will start with this base name.")
argparser.add_argument("--config", default="runner_config/config.py",
                       help="The config module to import. To make command line inputs easier, I allow .py suffix.")
argparser.add_argument("--script_dir", type=Path, default=Path("script"),
                       help="The directory of scripts.")
argparser.add_argument("--log_dir", type=Path, default=Path("ckpt"),
                       help="The directory of logs. Slurm logs will be {log_dir}/{run_name}/slurm.{out,err}. Set to the checkpoint path so logs will be with the checkpoints.")
argparser.add_argument("--run_name_flag", default="name",
                       help="Flag used to pass run/experiment name to the main file.")
argparser.add_argument("--time", default="48:00:00",
                       help="The time limit of the jobs.")
argparser.add_argument("--mem", default="32GB",
                       help="The memory limit of the jobs.")
argparser.add_argument("--mail-type", default="END,FAIL",
                       help="What types of mails to send to the mail user.")
argparser.add_argument("--mail-user", default="ww2135@nyu.edu",
                       help="The mail user to send mails to.")
argparser.add_argument("--python", default="python",
                       help="The python to run with; e.g., python3.")
argparser.add_argument("--sbatch", action="store_true",
                       help="Jobs will be immediately submitted.")
argparser.add_argument("--auto-flag", action="store_true",
                       help="Automatically find varying flags and display them in job names; if not set, use designated ordered list of flags.")
args = argparser.parse_args()

# create slurm directories
args.log_dir.mkdir(parents=True, exist_ok=True)
args.script_dir.mkdir(parents=True, exist_ok=True)

# config
py_suffix = '.py'
args.config = args.config.removesuffix(py_suffix)
args.config = args.config.replace("/", ".")
config = importlib.import_module(args.config)
grids, flags = config.grids, config.flags

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

print(f"Creating {len(jobs)} jobs:")

all_keys = set().union(*[g.keys() for g in grids])
merged = {k: set() for k in all_keys}
for grid in grids:
    for key in all_keys:
        grid_key_value = grid.get(key, [])
        merged[key] = merged[key].union(grid_key_value)
varying_keys = {key for key in merged if len(merged[key]) > 1}

if args.auto_flag:
    # display all varying keys in job name
    flags = list(varying_keys)
else:  # use flags
    # check whether there are flags that are varying but omitted in flags
    omitted_flags = [key for key in varying_keys if key not in flags]
    if omitted_flags:
        print(
            f"ERROR: {', '.join(omitted_flags)} are varying but omitted in flags")
        sys.exit()


excluded_flags = {'main_file'}

for job in jobs:
    # construct the job name
    job_name = args.job_name_base
    for flag in flags:
        value = job[flag]
        if isinstance(value, str):
            value = value.replace("/", ":")
        job_name += f"_{flag}_{value}"

    # construct the string of arguments to be passed to the script
    flagstring = ""

    # use the order of flags first, then all other flags at the last
    # this order is actually unimportant and simply for elegency
    flags_order = copy.copy(flags)
    for flag in job:
        if (flag not in flags_order) and (flag not in excluded_flags):
            flags_order.append(flag)

    for flag in flags_order:
        value = job[flag]
        if isinstance(value, bool):
            if value:
                flagstring += f" --{flag}"
            else:
                print("WARNING: Excluding 'False' flag " + flag)
        else:
            flagstring += f" --{flag} {value}"

    flagstring += f" --{args.run_name_flag} {job_name}"

    # create slurm script and slurm log dirs
    slurm_script_path = args.script_dir / (job_name + '.slurm')
    slurm_script_path.parent.mkdir(parents=True, exist_ok=True)

    slurm_log_dir = args.log_dir / job_name
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    # specify job command and create slurm file
    jobcommand = f"{args.python} {job['main_file']}{py_suffix}{flagstring}"

    job_start_command = f"sbatch {slurm_script_path}"

    print(jobcommand)
    with slurm_script_path.open('w') as slurmfile:
        slurmfile.write(
f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --open-mode=append
#SBATCH --output={slurm_log_dir / 'slurm.out'}
#SBATCH --error={slurm_log_dir / 'slurm.err'}
#SBATCH --export=ALL
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=pascal|turing|volta
#SBATCH --mail-type={args.mail_type}
#SBATCH --mail-user={args.mail_user}

srun {jobcommand}
""")

    if args.sbatch and click.confirm("Submit job through sbatch?", default=True):
        os.system(job_start_command + " &")
