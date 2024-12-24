from collections.abc import Mapping, Sequence
from collections import defaultdict
import os
import argparse
from pathlib import Path
import json
import re
import numpy as np
from utils import map_structure


def get_stat(*elements):
    if isinstance(elements[0], (int, float)):
        return np.mean(elements), np.std(elements), len(elements)
    elif all(e == elements[0] for e in elements):
        return elements[0]
    else:
        return None


def mean_std_repr(mean_std, prec=1):
    mean, std, n = mean_std
    return f"{mean:.{prec}%}({std:.{prec}%};{n})"


try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = '.'
pretrained_models = {
    'Llama-3-8B': llama_path+r'/Meta-Llama-3-8B',
    'Llama-3-8B-Instruct': llama_path+r'/Meta-Llama-3-8B-Instruct',
    'FLAN-Base-DefInstr': r'ltg/flan-t5-definition-en-base',
    'FLAN-Large-DefInstr': r'ltg/flan-t5-definition-en-large',
    'FLAN-XL-DefInstr': r'ltg/flan-t5-definition-en-xl',
}


llama_finetuned_path_formats = {
    'Llama-3-8B-finetuned': 'ckpt/meta-word_pretrained_model_{pretrained_model}_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_{seed}_eval_step_1000/best/meta-word-eval_data_dir_{eval_data}_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
    'Llama-3-8B-Instruct-finetuned': 'ckpt/meta-word_pretrained_model_{pretrained_model}_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_{seed}_eval_step_1000/best/meta-word-eval_data_dir_{eval_data}_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
}
llama_finetuned_paths = {
    name: {
        f'{name}_seed_{seed}': path_format.replace('{pretrained_model}', pretrained_models[name.removesuffix('-finetuned')].replace('/', ':')).replace('{seed}', str(seed))
        for seed in [0, 1, 2]
    }
    for name, path_format in llama_finetuned_path_formats.items()
}
paths = {
    **llama_finetuned_paths['Llama-3-8B-finetuned'],
    **llama_finetuned_paths['Llama-3-8B-Instruct-finetuned'],
    **{
        name: 'ckpt/meta-word-eval_data_dir_{eval_data}_pretrained_model_{pretrained_model}_prompt__n_examples_{n_examples}_max_new_tokens_100/slurm.out'.replace('{pretrained_model}', pretrained_model.replace('/', ':'))
        for name, pretrained_model in pretrained_models.items()
    },
    **{
        f'{name}-ori': 'ckpt/meta-word-eval_data_dir_{eval_data}_pretrained_model_{pretrained_model}_prompt__no_new_token_True_n_examples_{n_examples}_max_new_tokens_100/slurm.out'.replace('{pretrained_model}', pretrained_model.replace('/', ':'))
        for name, pretrained_model in pretrained_models.items()
    },
}


argparser = argparse.ArgumentParser()
argparser.add_argument("--collecting_results", action="store_true")
argparser.add_argument("--eval_data", choices=['def_task.json', 'oxford.json'], required=True)
argparser.add_argument("--job_name_base", default="evaluate-generation",
                    help="The base name of jobs. All job names will start with this base name.")
argparser.add_argument("--header", type=Path, default=Path("runner_config/header_4h.slurm"),
                    help="The header of scripts.")
argparser.add_argument("--submit", action="store_true",
                    help="Jobs will be submitted.")
argparser.add_argument("--no-confirm", action="store_true",
                    help="No confirmation.")
args = argparser.parse_args()
if args.submit and not args.no_confirm:
    import click

eval_data = args.eval_data
paths = {
    name: {
        n_shot: Path(p.format(eval_data=eval_data, n_examples=n_shot+1))
        for n_shot in ([1, 2, 3] if eval_data == 'def_task.json' and 'FLAN' not in name else [1])
    }
    for name, p in paths.items()
}

if not args.collecting_results:
    with open(args.header, "r") as header_f:
        header = header_f.read()

    for name, pdict in paths.items():
        sep = '' if 'FLAN' in name else None
        extract_definition_from_reference_generation = eval_data == 'def_task.json'
        extract_definition_from_generation = 'FLAN' not in name
        for n_shot, path in pdict.items():
            job_name = args.job_name_base+'_'+name+f'_{n_shot}-shot'
            jobcommand = f"python evaluate_generation.py '{path}'"
            if extract_definition_from_reference_generation:
                jobcommand += f" --extract_definition_from_reference_generation"
            if extract_definition_from_generation:
                jobcommand += f" --extract_definition_from_generation"
            if sep is not None:
                jobcommand += f" --sep '{sep}'"
            print(jobcommand)
            if not path.exists():
                print(f'{path} does not exist. Skipped.')
                continue
            wrapped_command = f"""
    srun {jobcommand}
    """
            slurm_script_path = path.with_name("evaluate_generation.slurm")
            with slurm_script_path.open('w') as slurmfile:
                slurmfile.write(header +
    f"""
    #SBATCH --job-name={job_name}
    #SBATCH --output={slurm_script_path.with_suffix('.out')}
    #SBATCH --error={slurm_script_path.with_suffix('.err')}
    """ + wrapped_command)

            job_start_command = f"sbatch {slurm_script_path}"

            try:
                submitting = args.submit and (args.no_confirm or click.confirm(f'Submit job {slurm_script_path}?', default=True))
            except click.exceptions.Abort:
                print()
                break
            else:
                if submitting:
                    os.system(job_start_command + " &")

else:
    results = {}
    for name, pdict in paths.items():
        for n_shot, path in pdict.items():
            try:
                with open(str(path)+'.eval_result.json', 'r') as f:
                    result = json.load(f)
            except FileNotFoundError:
                pass
            else:
                results[f'{name}-{n_shot}-shot'] = result
    agg_results = defaultdict(list)
    for name, result in results.items():
        agg_results[re.sub(r'_seed_\d+', r'', name)].append(result)
    agg_results = {
        name: map_structure(get_stat, *rlist, classinfo=(int, float, str))
        for name, rlist in agg_results.items()
    }
    for name, result in agg_results.items():
        assert isinstance(result, dict)
        rougeL = result['greedy outputs'][0]['rouge']['rougeL']
        bertscore_f1 = result['greedy outputs'][0]['bertscore']['f1']
        print(f'{name}: {mean_std_repr(bertscore_f1)} {mean_std_repr(rougeL)}')