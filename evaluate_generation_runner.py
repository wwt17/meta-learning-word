from collections.abc import Mapping, Sequence
from collections import defaultdict
import os
import argparse
from pathlib import Path
import json
import re
import numpy as np
import scipy as sp
from utils import map_structure


opposite_alternative = {
    'greater': 'less',
    'less': 'greater',
    'two-sided': 'two-sided',
}

def ttest(group0, group1, alternative='two-sided'):
    if len(group0) == 1:
        ttest_result = sp.stats.ttest_1samp(
            group1, group0[0],
            alternative=opposite_alternative[alternative])
    elif len(group1) == 1:
        ttest_result = sp.stats.ttest_1samp(
            group0, group1[0],
            alternative=alternative)
    else:
        ttest_result = sp.stats.ttest_ind(
            group0, group1, equal_var=False,
            alternative=alternative)
    return ttest_result


def tost(group1, group2, margin):
    """
    Two One-Sided Tests (TOST) for equivalence

    Tests if |μ1 - μ2| < margin (equivalence)

    Parameters:
    group1, group2: arrays of sample data
    margin: equivalence margin (δ)
    """

    group1 = np.array(group1)
    group2 = np.array(group2)

    # Test 1: H0: μ1 - μ2 ≤ -margin vs H1: μ1 - μ2 > -margin
    # Equivalent to testing: group1 > (group2 - margin)
    group2_adjusted1 = group2 - margin
    t1 = ttest(group1, group2_adjusted1, alternative='greater')

    # Test 2: H0: μ1 - μ2 ≥ margin vs H1: μ1 - μ2 < margin  
    # Equivalent to testing: group1 < (group2 + margin)
    group2_adjusted2 = group2 + margin
    t2 = ttest(group1, group2_adjusted2, alternative='less')

    # Overall TOST result
    # Both tests must reject H0 for equivalence
    p_tost = max(t1.pvalue, t2.pvalue)  # Conservative approach
    return p_tost


def get_collection(*elements):
    if isinstance(elements[0], (int, float)):
        return elements
    elif all(e == elements[0] for e in elements):
        return elements[0]
    else:
        return None


def mean_std_repr(elements, prec=1):
    mean, std, n = np.mean(elements), np.std(elements), len(elements)
    return f"{mean:.{prec}%}({std:.{prec}%};{n})"


try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = '.'
pretrained_models = {
    'Llama-2-7B': r'Llama-2-7b-hf',
    'Llama-3-8B': llama_path+r'/Meta-Llama-3-8B',
    'Llama-3-8B-Instruct': llama_path+r'/Meta-Llama-3-8B-Instruct',
    'FLAN-Base-DefInstr': r'ltg/flan-t5-definition-en-base',
    'FLAN-Large-DefInstr': r'ltg/flan-t5-definition-en-large',
    'FLAN-XL-DefInstr': r'ltg/flan-t5-definition-en-xl',
}


llama_finetuned_path_formats = {
    'Llama-3-8B-finetuned': 'ckpt/meta-word_pretrained_model_{pretrained_model}_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_{seed}_eval_step_1000/best/meta-word-eval_data_dir_{eval_data}_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
    'Llama-3-8B-Instruct-finetuned': 'ckpt/meta-word_pretrained_model_{pretrained_model}_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_{seed}_eval_step_1000/best/meta-word-eval_data_dir_{eval_data}_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
    'Llama-2-7B-finetuned': 'ckpt/meta-word_pretrained_model_{pretrained_model}_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_5_train_max_length_160_batch_size_16_lr_0.003_seed_{seed}/best/meta-word-eval_data_dir_{eval_data}_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
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
    **llama_finetuned_paths['Llama-2-7B-finetuned'],
    **{
        name: 'ckpt/meta-word-eval_data_dir_{eval_data}_pretrained_model_{pretrained_model}_prompt__n_examples_{n_examples}_max_new_tokens_100/slurm.out'.replace('{pretrained_model}', pretrained_model.replace('/', ':'))
        for name, pretrained_model in pretrained_models.items()
    },
    **{
        f'{name}-ori': 'ckpt/meta-word-eval_data_dir_{eval_data}_pretrained_model_{pretrained_model}_prompt__no_new_token_True_n_examples_{n_examples}_max_new_tokens_100/slurm.out'.replace('{pretrained_model}', pretrained_model.replace('/', ':'))
        for name, pretrained_model in pretrained_models.items()
    },
    'CoLLEGe': 'ckpt/meta-word-eval_data_dir_{eval_data}_pretrained_model_Llama-2-7b-hf_emb_gen_model_type_college_n_examples_{n_examples}_max_new_tokens_100/slurm.out',
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
        name: map_structure(get_collection, *rlist, classinfo=(int, float, str))
        for name, rlist in agg_results.items()
    }

    def get_scores(result):
        rougeL = result['greedy outputs'][0]['rouge']['rougeL']
        bertscore_f1 = result['greedy outputs'][0]['bertscore']['f1']
        return bertscore_f1, rougeL

    name_max_length = max(map(len, agg_results.keys()))
    for name, result in agg_results.items():
        assert isinstance(result, dict)
        scores = get_scores(result)
        print(f'{name:{name_max_length}}: {" ".join(map(mean_std_repr, scores))}')
    
    comparisons = [
        ('Llama-3-8B', 'Llama-3-8B-finetuned'),
        ('Llama-3-8B-Instruct', 'Llama-3-8B-Instruct-finetuned'),
        ('Llama-2-7B', 'Llama-2-7B-finetuned'),
        ('Llama-3-8B-finetuned', 'Llama-3-8B-Instruct-finetuned'),
        ('CoLLEGe', 'Llama-2-7B-finetuned'),
        ('Llama-3-8B-Instruct-finetuned', 'FLAN-XL-DefInstr'),
    ]
    alternative = 'less'
    margin = 0.005
    print(f'Comparing the latter to the former. {alternative=}')
    for n_shot in [1]:
        for comparison_prefixes in comparisons:
            assert len(comparison_prefixes) == 2
            comparison_names = tuple((prefix+f"-{n_shot}-shot" for prefix in comparison_prefixes))
            comparison_scores = tuple((get_scores(agg_results[name]) for name in comparison_names))
            print("Comparing:")
            for name, scores in zip(comparison_names, comparison_scores):
                print(f'{name:{name_max_length}}: {" ".join(map(mean_std_repr, scores))}')
            pvalues = []
            for score_pair in zip(*comparison_scores):
                ttest_result = ttest(*score_pair, alternative=alternative)
                pvalues.append(ttest_result.pvalue)
            print(f'{"pvalue":{name_max_length}}: {" ".join(f"{pvalue:.7e}" for pvalue in pvalues)}')
            tost_pvalues = []
            for score_pair in zip(*comparison_scores):
                p_tost = tost(*score_pair, margin=margin)
                tost_pvalues.append(p_tost)
            print(f'{"tost pvalue":{name_max_length}}: {" ".join(f"{pvalue:.7e}" for pvalue in tost_pvalues)}')