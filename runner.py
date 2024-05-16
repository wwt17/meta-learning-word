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
argparser.add_argument("--header", type=Path, default=Path("runner_config/header.slurm"),
                       help="The header of scripts.")
argparser.add_argument("--script_dir", type=Path,
                       help="The directory of scripts. Default to the directory of logs.")
argparser.add_argument("--log_dir", type=Path, default=Path("ckpt"),
                       help="The directory of logs. Slurm logs will be {log_dir}/{run_name}/slurm.{out,err}. Set to the checkpoint path so logs will be with the checkpoints.")
argparser.add_argument("--log_dir_flag",
                       help="Flag for the log_dir. If set, overrides --log_dir.")
argparser.add_argument("--run_name_flag",
                       help="Flag used to pass run/experiment name to the main file.")
argparser.add_argument("--program", default="python",
                       help="The program to run with; e.g., python.")
argparser.add_argument("--submit", action="store_true",
                       help="Jobs will be submitted.")
argparser.add_argument("--no-confirm", action="store_true",
                       help="No confirmation.")
argparser.add_argument("--auto-flag", action="store_true",
                       help="Automatically find varying flags and display them in job names; if not set, use designated ordered list of flags.")
args = argparser.parse_args()

# read header
with open(args.header, "r") as header_f:
    header = header_f.read()

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
    job_name_for_log = args.job_name_base
    for flag in flags:
        value = job[flag]
        if isinstance(value, str):
            value = value.replace("/", ":")
        s = f"_{flag}_{value}"
        job_name += s
        if flag != args.log_dir_flag:
            job_name_for_log += s
    log_dir = args.log_dir if args.log_dir_flag is None else Path(job[args.log_dir_flag])

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
                flagstring += f' --{flag}'
            else:
                print("WARNING: Excluding 'False' flag " + flag)
        else:
            flagstring += f' --{flag}'
            if isinstance(value, (list, tuple)):
                for v in value:
                    flagstring += f' "{v}"'
            else:
                flagstring += f' "{value}"'

    if args.run_name_flag:
        flagstring += f' --{args.run_name_flag} "{job_name}"'

    # create slurm script and slurm log dirs
    slurm_log_dir = log_dir / job_name_for_log
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    if args.script_dir is None:
        slurm_script_path = slurm_log_dir / 'script.slurm'
    else:
        slurm_script_path = args.script_dir / (job_name + '.slurm')
        slurm_script_path.parent.mkdir(parents=True, exist_ok=True)

    # specify job command and create slurm file
    jobcommand = f"{args.program} {job['main_file']}{flagstring}"

    job_start_command = f"sbatch {slurm_script_path}"

    print(jobcommand)
    with slurm_script_path.open('w') as slurmfile:
        slurmfile.write(header +
f"""
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_log_dir / 'slurm.out'}
#SBATCH --error={slurm_log_dir / 'slurm.err'}

srun {jobcommand}
""")

    try:
        submitting = args.submit and (args.no_confirm or click.confirm("Submit job?", default=True))
    except click.exceptions.Abort:
        print()
        break
    else:
        if submitting:
            os.system(job_start_command + " &")
