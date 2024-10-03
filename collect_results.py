import argparse
import copy
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


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


if __name__ == "__main__":
    filenames = [
        #f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_{seed}/best/{eval_name}/slurm.out"
        f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_{seed}/best/{eval_name}/slurm.out"
        for n_examples in [5, 10][:1]
        for eval_name in [
            f"meta-word-eval_data_dir_word_use_data:childes:word_split_test_n_examples_{n_examples}",
            f"meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_n_examples_{n_examples}",
            "meta-word-eval_data_dir_syntactic",
        ][1:2]
        for seed in [0, 1, 2]
    ]
    results = []
    for filename in filenames:
        with open(filename) as file:
            res = collect_results_from_file(file)
            results.append(res)
            #print(res)
    res = zipdicts(*results)
    print(json.dumps(res, indent=2))
    res = {
        key: mean_std(values)
        for key, values in res.items()
    }
    print(json.dumps(res, indent=2))