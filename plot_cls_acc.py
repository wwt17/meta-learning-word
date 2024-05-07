import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cls_acc(result_files, title, x_axis="#classes", chance=True, kind="line"):
    n_classes = None
    accuracies = {}

    for n_examples, result_file in result_files.items():
        with open(result_file, "r") as f:
            n_classes_, accuracies_ = [], []
            for line in f:
                match = re.fullmatch(r"val_cls_(\d+)_acc=([0-9\.]*)%", line.strip())
                if match is None:
                    break
                n_cls, accuracy = int(match.group(1)), float(match.group(2))
                n_classes_.append(n_cls)
                accuracies_.append(accuracy)
        if n_classes is None:
            n_classes = n_classes_
        else:
            assert n_classes == n_classes_
        accuracies[n_examples] = accuracies_

    n_classes = np.array(n_classes)
    n_examples = list(result_files.keys())
    xticks, yticks = n_classes, n_examples
    df = pd.DataFrame(accuracies, index=n_classes)
    df.index.name = "#classes"
    df.columns.name = "#examples"
    if x_axis == "#examples":
        df = df.transpose()
        xticks, yticks = yticks, xticks

    palette = sns.color_palette("husl", len(df.columns))

    if chance:
        if x_axis == "#examples":
            df = pd.concat([pd.DataFrame([df.columns.map(lambda c: 100/c)], columns=df.columns, index=[1]), df], ignore_index=False)
            xticks = [1] + xticks
        else:
            df.insert(0, "chance", df.index.map(lambda c: 100/c))
            palette = [(0., 0., 0.)] + palette

    if kind == "line":
        dashes = {column: {"chance": (1, 1)}.get(column, "") for column in df.columns}
        ax = sns.lineplot(df, palette=palette, dashes=dashes)
        ax.set_xticks(xticks)
        if chance and x_axis == "#examples":
            ax.set_xlim(xmin=1)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(bottom=0, top=100)
        ax.set_title(title)
    else:
        grid = sns.catplot(
            df.melt(ignore_index=False, value_name="Accuracy (%)").reset_index(),
            x=x_axis,
            y="Accuracy (%)",
            hue=("#examples" if x_axis == "#classes" else "#classes"),
            palette=palette,
            kind=kind, #type:ignore
            facet_kws=dict(
                ylim=(0, 100)
            )
        )
        grid.fig.subplots_adjust(top=0.9)
        grid.fig.suptitle(title)



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

    title = "Meta-Trained GPT-2 on CHILDES"
    out = title+".png"
    result_files = {
        "Meta-Trained GPT-2 on CHILDES": {
            n_examples: f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_gpt2_concat_False_context_length_128_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0001_weight_decay_0.12_seed_0/best/meta-word-eval_data_dir_word_use_data:childes:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        },
        "Meta-Trained GPT-NeoX on CHILDES": {
            n_examples: f"ckpt/meta-word_data_dir_word_use_data:childes:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.07_seed_0/best/meta-word-eval_data_dir_word_use_data:childes:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        },
        "Meta-Trained GPT-NeoX on BabyLM-10M": {
            n_examples: f"ckpt/meta-word_data_dir_word_use_data:babylm_data:babylm_10M:word_config_model_config:pythia-160m_concat_False_no_new_token_False_n_examples_{n_examples}_max_sample_times_0_batch_size_8_lr_0.0003_weight_decay_0.15_seed_0/best/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_{n_examples}/slurm.out"
            for n_examples in range(4, 11)
        },
        "Pretrained GPT-2 small with simple format on CHILDES": {
            n_examples: f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_gpt2_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        },
        "Pretrained Pythia-160M with simple format on CHILDES": {
            n_examples: f"ckpt/meta-word-eval_data_dir_word_use_data:childes:word_pretrained_model_EleutherAI:pythia-160m_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        },
        "Pretrained Pythia-160M with simple format on BabyLM-10M": {
            n_examples: f"ckpt/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_pretrained_model_EleutherAI:pythia-160m_n_examples_{n_examples}/slurm.out"
            for n_examples in range(2, 11)
        },
    }[title]

    plot_cls_acc(result_files, title, x_axis="#examples", chance=True, kind="line")

    plt.savefig(out, transparent=True, dpi=1000)