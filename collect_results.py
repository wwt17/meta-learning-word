import os
import argparse
import copy
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


plt.rcParams['axes.grid'] = True  # Enable grid lines
plt.rcParams['grid.color'] = 'gray'  # Set grid color
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style
plt.rcParams['grid.linewidth'] = 0.7  # Set grid line width
plt.rcParams["font.family"] = "serif"


try:
    llama_path = os.environ["SCRATCH"]
except KeyError:
    llama_path = '.'
pretrained_models = {
    'Llama-3 8B': llama_path+r'/Meta-Llama-3-8B',
    'Llama-3 8B Instruct': llama_path+r'/Meta-Llama-3-8B-Instruct',
}


syn_ctgs = ["n", "v", "adj", "adv"]
ctgs = ["ctg0", "ctg1"]
data_kinds = {
    'diff': '_different_{ctg}_{split}.txt'.format,
    'ident': '_identical_{ctg}_{split}.txt'.format,
}


def collect_results_from_file(file):
    results = {}

    for line in file:
        match = re.fullmatch(r"(?P<key>[\w-]+)=(|(\d+/\d+)=)(?P<value>[\d\.]*)%", line.strip())
        if match is None:
            continue
        results[match["key"]] = float(match["value"])

    if 'test_n_v_ctg0_diff_acc' in results:
        for split in ['dev', 'test']:
            for syn_ctg_pair in combinations(syn_ctgs, 2):
                for data_kind in data_kinds:
                    s = 0.
                    for ctg in ctgs:
                        name = f"{split}_{'_'.join(syn_ctg_pair)}_{ctg}_{data_kind}_acc"
                        s += results[name]
                    s /= len(ctgs)
                    results[f"{split}_{'_'.join(syn_ctg_pair)}_mean_{data_kind}_acc"] = s

    return results


def mean_std(a):
    return f"{np.mean(a):.1f}({np.std(a):.1f})"


def zipdicts(*ds):
    return {
        key: [d[key] for d in ds]
        for key in ds[0].keys()
        if all((key in d for d in ds))
    }


c = [
    {
        "data_dir": "word_use_data/childes/word",
        "n_examples": 5,
        "batch_size": 32,
        "lr": 3e-3,
        "train_max_length": 80,
    },
    {
        "data_dir": "word_use_data/childes/word",
        "n_examples": 10,
        "batch_size": 8,
        "lr": 3e-4,
        "train_max_length": 160,
    },
    {
        "data_dir": "word_use_data/babylm_data/babylm_10M/word",
        "n_examples": 5,
        "batch_size": 16,
        "lr": 1e-3,
        "train_max_length": 160,
    },
][2]
eval_name = [
    f"meta-word-eval_data_dir_{c['data_dir'].replace('/', ':')}_split_test_n_examples_{c['n_examples']}_max_new_tokens_100",
    "meta-word-eval_data_dir_syntactic",
][1]
trained_from_scratch_on_childes = [
    f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_5_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_{seed}/best/{eval_name}/slurm.out"
    for seed in [0, 1, 2]
]
trained_from_scratch_on_babylm_10m = [
    f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{c['n_examples']}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_{seed}/best/{eval_name}/slurm.out"
    for seed in [0, 1, 2]
]
init_name = 'Llama-3 8B'
pretrained_model = pretrained_models[init_name]
pretrained_model_option = f"_pretrained_model_{pretrained_model.replace('/', ':')}"
baseline = [
    f"ckpt/{eval_name}_pretrained_model_Meta-Llama-3-8B-hf_prompt__new_word_{new_word}/slurm.out"
    for new_word in [" dax", " wug", " blicket"]
]
baseline_2 = [
    f"ckpt/{eval_name}_pretrained_model_Llama-2-7b-hf_prompt__new_word_{new_word}/slurm.out"
    for new_word in [" dax", " wug", " blicket"]
]
college = [
    f"ckpt/{eval_name}_pretrained_model_Llama-2-7b-hf_emb_gen_model_type_college/slurm.out"
]
finetuned = [
    f"ckpt/meta-word{pretrained_model_option}_data_dir_{c['data_dir'].replace('/', ':')}_embedding_init_mean_train_params_new_word_sep_n_examples_{c['n_examples']}_train_max_length_{c['train_max_length']}_batch_size_{c['batch_size']}_lr_{c['lr']}_seed_{seed}_eval_step_1000/best/{eval_name}/slurm.out"
    for seed in [0, 1, 2]
]
method_name = "Minnow"
model_filenames = {
    f"{method_name} from scratch on CHILDES": trained_from_scratch_on_childes,
    f"{method_name} from scratch on BabyLM-10M": trained_from_scratch_on_babylm_10m,
    f"{init_name} baseline": baseline,
    f"{init_name} +{method_name} on BabyLM-10M": finetuned,
    f"Llama-2 7B baseline": baseline_2,
    f"CoLLEGe": college,
}


def plot_model_results(model_results, split='test', data_kind='diff'):
    attrs, attr_names = [], []
    for syn_ctg_pair in combinations(syn_ctgs, 2):
        attr = f"{split}_{'_'.join(syn_ctg_pair)}_mean_{data_kind}_acc"
        attrs.append(attr)
        attr_name = f"{syn_ctg_pair[0].capitalize()} vs. {syn_ctg_pair[1].capitalize()}"
        attr_names.append(attr_name)
    run_data = [
        (model_name, [res[attr] for attr in attrs])
        for model_name, results in model_results.items()
        for res in results
    ]
    run_models, run_acc = zip(*run_data)
    acc = pd.DataFrame(
        run_acc,
        index=run_models,
        columns=attr_names,
    )
    acc.insert(0, 'Mean', acc.mean(axis=1))
    print(acc)

    acc = acc.melt(var_name="Category Pair", value_name="Accuracy", ignore_index=False).reset_index(names="Model")

    plt.figure(figsize=(8, 3))
    sns.barplot(acc, x="Category Pair", y="Accuracy", hue="Model", hue_order=list(model_results.keys()))
    sns.despine()
    plt.legend(fontsize="small")
    plt.gca().axhline(50, c='black', ls="--")
    plt.ylim(0, 100)
    plt.tight_layout()
    #plt.show()
    plt.savefig("syntactic_classification_accuracy.pdf")


if __name__ == "__main__":
    model_results = {}
    for model_name, filenames in model_filenames.items():
        print(f"model: {model_name}")
        results = []
        for filename in filenames:
            with open(filename) as file:
                res = collect_results_from_file(file)
                results.append(res)
        model_results[model_name] = results
        results = zipdicts(*results)
        print(json.dumps(results, indent=2))
        results_mean_std = {
            key: mean_std(values)
            for key, values in results.items()
        }
        print(json.dumps(results_mean_std, indent=2))

    if eval_name == "meta-word-eval_data_dir_syntactic":
        plot_model_results(model_results)