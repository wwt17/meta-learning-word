import argparse
import copy
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


attr_patterns = {
    "#examples": r"_n_examples_(\d+)",
    "step": r"_step(\d+)",
}


def read_cls_acc(result_files, chance=False):
    records = []
    for result_file in result_files:
        config = {}
        for attr, pattern in attr_patterns.items():
            match = re.search(pattern, result_file)
            if match is None:
                value = None
            else:
                value = int(match[1])
            config[attr] = value
        try:
            with open(result_file, "r") as f:
                for line in f:
                    match = re.fullmatch(r"val_cls_(\d+)_acc=([0-9\.]*)%", line.strip())
                    if match is None:
                        break
                    record = copy.copy(config)
                    record.update({
                        "#classes": int(match[1]),
                        "Accuracy (%)": float(match[2]),
                    })
                    records.append(record)
        except FileNotFoundError:
            pass
    df = pd.DataFrame(records)
    if chance:
        df_chance = df.copy()
        df_chance["#examples"] = 1
        df_chance["Accuracy (%)"] = 100 / df_chance["#classes"]
        df_chance = df_chance.drop_duplicates()
        df = pd.concat([df_chance, df], ignore_index=True)
        df = df.reset_index(drop=True)
    return df


def plot_cls_acc(df, title, x="#classes", y="Accuracy (%)", hue="#examples", kind="line"):
    x_values, hue_values = np.sort(df[x].unique()), np.sort(df[hue].unique())
    palette = sns.color_palette("husl", len(hue_values))

    if kind == "line":
        ax = sns.lineplot(df, x=x, y=y, hue=hue, palette=palette, dashes=False)
        ax.set_xticks(x_values)
        if x == "#examples":
            ax.set_xlim(xmin=1)
        elif x == "step":
            ax.set_xlim(xmin=0)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(bottom=0, top=100)
        ax.set_title(title)
    else:
        grid = sns.catplot(
            df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            kind=kind, #type:ignore
            facet_kws=dict(
                ylim=(0, 100)
            )
        )
        grid.figure.subplots_adjust(top=0.9)
        grid.figure.suptitle(title)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    rc = {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
    }
    sns.set_theme(style="whitegrid", rc=rc)

    title = "Pythia-70M with simple format on CHILDES"
    out = title+".png"
    steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000))
    result_files = {
        "Meta-Trained GPT-2 on CHILDES": [
            f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_gpt2_concat_False_context_length_128_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_0.12_seed_0/best/meta-word-eval_data_dir_word_use_data:childes:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        ],
        "Meta-Trained GPT-NeoX on CHILDES": [
            f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_0/best/meta-word-eval_data_dir_word_use_data:childes:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        ],
        "Meta-Trained GPT-NeoX on BabyLM-10M": [
            f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_0/best/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        ],
        "Pretrained GPT-2 small with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_gpt2_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        ],
        "Pretrained Pythia-160M with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-160m_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        ],
        "Pretrained Pythia-160M with simple format on BabyLM-10M": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_pretrained_model_EleutherAI:pythia-160m_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        ],
        "Pretrained Llama-3-8B on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        ],
        "Pythia-70M on CHILDES": [
            f"ckpt/meta-word-eval__data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-70m_revision_step{step}_n_examples_10/slurm.out"
            for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000]
        ],
        "Pythia-70M with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-70m_revision_step{step}_n_examples_10/slurm.out"
            for step in steps
        ],
        "Pythia-70M-deduped with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-70m-deduped_revision_step{step}_n_examples_10/slurm.out"
            for step in steps
        ],
        "Pythia-160M with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-160m_revision_step{step}_n_examples_10/slurm.out"
            for step in steps
        ],
        "Pythia-410M with simple format on CHILDES": [
            f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-410m_revision_step{step}_n_examples_10/slurm.out"
            for step in steps
        ],
    }[title]

    if re.fullmatch(r"Pythia-\d+M(|-deduped)(| with simple format) on .*", title):
        x, hue = "step", "#classes"
    else:
        x, hue = "#examples", "#classes"
    chance = "#examples" in [x, hue]
    df = read_cls_acc(result_files, chance=chance)
    plot_cls_acc(df, title, x=x, hue=hue, kind="line")

    plt.savefig(out, transparent=True, dpi=1000)