import argparse
import json
from pathlib import Path
import numpy as np
import scipy as sp


def mean_std_repr(elements, prec=1):
    mean, std, n = np.mean(elements), np.std(elements), len(elements)
    return f"{mean:.{prec}%}({std:.{prec}%};{n})"


# Define the tasks
tasks = [
    {
        'name': 'babylm',
        'cmd_template': 'python compare_generation.py babylm {model1} {model2} --word_example_prompt_name "llama-3 baseline" --judges gpt-4o --result_file {result_file} --skip_judged'
    },
    {
        'name': 'chimera',
        'cmd_template': 'python compare_generation.py chimera {model1} {model2} --word_example_prompt_name "llama-3 baseline" --judges gpt-4o --result_file {result_file} --skip_judged'
    },
    {
        'name': 'def_task',
        'cmd_template': 'python compare_generation.py def_task {model1} {model2} --mode definition --word_target GT --judges gpt-4o --result_file {result_file} --skip_judged'
    }
]

# Model pairs
model_pairs = [
    ("llama-3", "llama-3 baseline"),
    ("llama-3-instruct", "llama-3-instruct baseline"),
    ("llama-2", "college"),
]


def get_winner_ids(results, judge='gpt-4o', output_field='greedy outputs'):
    winner_ids = [
        winner_ids[judge]
        for example_results in results
        for winner_ids in example_results[output_field]
        if judge in winner_ids
    ]
    winner_ids = np.array(winner_ids)
    return winner_ids


def collect_results(res_dir):
    for output_field in [
        "greedy outputs",
        "sample with top-p=0.92 outputs",
    ]:
        print(f'{output_field=}')

        for base_model, comparison_model in model_pairs:
            print(f'  compare {base_model} to {comparison_model}:')

            for task in tasks:
                task_name = task['name']
                print(f'    {task_name=}')

                counts_list = {-1: [], 0: [], 1: []}

                for seed in range(3):
                    # Construct model names with seed
                    model1 = f"{base_model} seed {seed}"
                    model2 = comparison_model

                    # Create safe names for filenames (replace spaces with underscores)
                    model1_safe = model1.replace(' ', '_')
                    model2_safe = model2.replace(' ', '_')

                    # Construct the job name based on task and model names
                    job_name = f"compare_{task_name}_{model1_safe}_to_{model2_safe}"

                    result_file = res_dir/f"{job_name}.json"
                    with open(result_file, "r") as result_f:
                        results = json.load(result_f)
                    winner_ids = get_winner_ids(results, output_field=output_field)

                    unique, counts = np.unique(winner_ids, return_counts=True)
                    count_dict = dict(zip(unique, counts))
                    for i in range(-1, 2):
                        counts_list[i].append(count_dict.get(i, 0)/len(winner_ids))

                counts_array = {key: np.array(value) for key, value in counts_list.items()}
                for i in range(1, -1, -1):
                    print('      ' + mean_std_repr(counts_array[i]))
                pvalue = sp.stats.ttest_rel(counts_array[0], counts_array[1]).pvalue
                print('      ' + f'p: {pvalue:.6f}')


def create_scripts(res_dir):
    # Create directory for all SLURM jobs and results
    res_dir.mkdir(exist_ok=True)

    all_scripts = []

    # loops: task -> model_pair -> seed
    for task in tasks:
        task_name = task['name']

        for base_model, comparison_model in model_pairs:
            for seed in range(3):
                # Construct model names with seed
                model1 = f"{base_model} seed {seed}"
                model2 = comparison_model

                # Create safe names for filenames (replace spaces with underscores)
                model1_safe = model1.replace(' ', '_')
                model2_safe = model2.replace(' ', '_')

                # Construct the job name based on task and model names
                job_name = f"compare_{task_name}_{model1_safe}_to_{model2_safe}"

                # Format the command
                cmd = task['cmd_template'].format(
                    model1=f'"{model1}"',  # Quote model names with spaces
                    model2=f'"{model2}"',
                    result_file=res_dir/f"{job_name}.json",
                )

                script_path = res_dir/f"{job_name}.slurm"

                # Create SLURM script content
                slurm_content = f"""#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww2135@nyu.edu
#SBATCH --job-name={job_name}
#SBATCH --output={script_path.with_suffix('.out')}
#SBATCH --error={script_path.with_suffix('.err')}

srun {cmd}
"""

                # Write SLURM script
                with open(script_path, 'w') as f:
                    f.write(slurm_content)

                all_scripts.append(script_path)
                print(f"Created: {script_path}")

    print(f"\nTotal jobs created: {len(all_scripts)}")
    print(f"All scripts in: {res_dir}")

    return all_scripts


def create_submit_all_script(submit_all_path: Path, all_scripts: list[Path]):
    """Create a master submission script"""

    submit_all_content = f"""#!/bin/bash
    # Submit all jobs from the root directory
    # Run this script from the root directory: {submit_all_path}

    echo "Submitting all jobs..."

    """

    # Group scripts by task for organized submission
    for task in tasks:
        task_name = task['name']
        submit_all_content += f"\n# Submit {task_name} jobs\n"
        for script in all_scripts:
            if script.stem.startswith(f"compare_{task_name}_"):
                submit_all_content += f'echo "Submitting {script}"\n'
                submit_all_content += f'sbatch {script}\n'

    submit_all_content += '\necho "All jobs submitted!"'

    with open(submit_all_path, 'w') as f:
        f.write(submit_all_content)

    # Make the submission script executable
    submit_all_path.chmod(0o755)


def create_submit_sequential_script(submit_sequential_path: Path, all_scripts: list[Path]):
    """Create a sequential submission script with dependencies"""

    submit_sequential_content = f"""#!/bin/bash
# Submit all jobs sequentially with dependencies
# Run this script from the root directory: {submit_sequential_path}

echo "Submitting all jobs sequentially..."

# Submit first job and capture job ID
"""

    first_job = True
    for i, script in enumerate(all_scripts):
        if first_job:
            submit_sequential_content += f'first_job_id=$(sbatch --parsable {script})\n'
            submit_sequential_content += f'echo "Submitted {script} with job ID: $first_job_id"\n'
            submit_sequential_content += 'prev_job_id=$first_job_id\n\n'
            first_job = False
        else:
            submit_sequential_content += f'job_id=$(sbatch --parsable --dependency=afterok:$prev_job_id {script})\n'
            submit_sequential_content += f'echo "Submitted {script} with job ID: $job_id (depends on $prev_job_id)"\n'
            submit_sequential_content += 'prev_job_id=$job_id\n\n'

    submit_sequential_content += 'echo "All jobs submitted with sequential dependencies!"'

    with open(submit_sequential_path, 'w') as f:
        f.write(submit_sequential_content)

    # Make the sequential submission script executable
    submit_sequential_path.chmod(0o755)


def create_combined_job_script(combined_job_path: Path):
    """Create a single combined job script"""

    combined_job_content = f"""#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww2135@nyu.edu
#SBATCH --job-name=all_comparisons_sequential
#SBATCH --output={combined_job_path.with_suffix('.out')}
#SBATCH --error={combined_job_path.with_suffix('.err')}

echo "Starting sequential execution of all comparison tasks..."
echo "=================================================="

"""

    # Add all commands to the combined job
    for task in tasks:
        task_name = task['name']
        combined_job_content += f'\n# {task_name.upper()} TASKS\n'
        combined_job_content += f'echo "\\nStarting {task_name} tasks..."\n\n'

        for base_model, comparison_model in model_pairs:
            for seed in range(3):
                model1 = f"{base_model} seed {seed}"
                model2 = comparison_model
                model1_safe = model1.replace(' ', '_')
                model2_safe = model2.replace(' ', '_')
                job_name = f"compare_{task_name}_{model1_safe}_to_{model2_safe}"

                cmd = task['cmd_template'].format(
                    model1=f'"{model1}"',
                    model2=f'"{model2}"',
                    result_file=res_dir/f"{job_name}.json",
                )

                combined_job_content += f'echo "Running: {job_name}"\n'
                combined_job_content += f'{cmd}\n'
                combined_job_content += f'echo "Completed: {job_name}"\n'
                combined_job_content += f'echo "Sleeping for 2 seconds to avoid rate limits..."\n'
                combined_job_content += f'sleep 2\n\n'

    combined_job_content += '\necho "All comparisons completed!"'

    with open(combined_job_path, 'w') as f:
        f.write(combined_job_content)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["create", "collect"])
    argparser.add_argument("--res_dir", type=Path, default=Path("compare_gens_res"))
    args = argparser.parse_args()

    res_dir = args.res_dir

    if args.mode == "create":
        all_scripts = create_scripts(res_dir)

        submit_all_path = res_dir/"submit_all.sh"
        create_submit_all_script(submit_all_path, all_scripts)

        submit_sequential_path = res_dir/"submit_sequential.sh"
        create_submit_sequential_script(submit_sequential_path, all_scripts)

        combined_job_path = res_dir/"run_all_sequential.slurm"
        create_combined_job_script(combined_job_path)

        print(f"\nCreated master submission script: {submit_all_path}")
        print(f"Created sequential submission script: {submit_sequential_path}")
        print(f"Created combined sequential job: {combined_job_path}")
        print("\nTo submit all jobs in PARALLEL:")
        print(f"  ./{submit_all_path}")
        print("\nTo submit all jobs SEQUENTIALLY (with dependencies):")
        print(f"  ./{submit_sequential_path}")
        print("\nTo run all jobs in a SINGLE SLURM job (recommended for rate limits):")
        print(f"  sbatch {combined_job_path}")
        print("\nTo submit individual jobs:")
        print(f"  sbatch {res_dir}/<script_name>.slurm")
    
    elif args.mode == "collect":
        collect_results(res_dir)