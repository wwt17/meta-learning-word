from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from itertools import islice, chain
import random, re, json
from math import floor, ceil
import tqdm
import datasets
import tokenizers
import transformers
from transformers import PreTrainedTokenizerFast, set_seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import spacy
import seaborn as sns
from utils import frac_repr, cache, get_pos_tags, replace_at_offsets
from pos_tags import extend_pos
from plotting import palette
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS


nlp = spacy.load("en_core_web_sm", exclude=["parser", "attribute_ruler", "lemmatizer", "ner"])

def nlp_tokenizer(sentence):
    word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
    words, offsets = zip(*word_offsets)
    spaces = [
        (offsets[i][1] < (offsets[i+1][0] if i+1 < len(offsets) else len(sentence)))
        for i in range(len(offsets))
    ]
    doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces) # type: ignore
    assert str(doc) == sentence
    return doc

nlp.tokenizer = nlp_tokenizer


def count_tokens(
        pre_tokenized_sentences: Iterable[Iterable[tuple[str, Any]]]
) -> dict[str, list[tuple[int, Any]]]:
    counter = defaultdict(list)
    for i, word_offsets in enumerate(pre_tokenized_sentences):
        for word, offset in word_offsets:
            counter[word].append((i, offset))
    return counter


def count_token_pos(
        pre_tokenized_sentences: Iterable[Iterable[tuple[str, Any]]],
        pos_tags: Iterable[Iterable[str]]
) -> dict[tuple[str, str], list[tuple[int, Any]]]:
    counter = defaultdict(list)
    for i, (word_offsets, sentence_pos_tags) in enumerate(zip(pre_tokenized_sentences, pos_tags)):
        for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
            counter[(word, pos)].append((i, offset))
    return counter


KeyType = TypeVar('KeyType')
def sorted_counter_list(counter: Mapping[KeyType, Sized]) -> list[tuple[KeyType, Sized]]:
    return sorted(counter.items(), key=lambda item: (-len(item[1]), item[0]))

def sorted_counter_dict(counter: Mapping[KeyType, Sized]) -> Mapping[KeyType, Sized]:
    return dict(sorted_counter_list(counter))

PosType = TypeVar('PosType')
def max_freq_pos_counter_dict(counter: Mapping[tuple[KeyType, PosType], Sized]) -> Mapping[KeyType, tuple[PosType, Sized]]:
    items = sorted_counter_list(counter)
    ret = {}
    for (word, pos), occurrences in items:
        if word not in ret:
            ret[word] = (pos, occurrences)
    return ret


def build_vocab(
        pre_tokenized_sentences: Iterable[Iterable[tuple[str, Any]]],
        special_tokens: list[str] = SPECIAL_TOKENS,
        new_tokens: list[str] = NEW_TOKENS,
) -> dict[str, int]:
    counter = count_tokens(pre_tokenized_sentences)
    vocab = sorted_counter_dict(counter)
    vocab = (
        special_tokens +
        [word for word in vocab if word not in special_tokens] +
        new_tokens
    )
    vocab = {word: idx for idx, word in enumerate(vocab)}
    return vocab


def build_word_use_data(
        data: datasets.Dataset,
        used_vocab: Mapping[str, tuple[str, Any]],
        mode: str = "word",
        offsets_matching_pos: bool = False,
) -> Mapping[str, list[str]]:
    word_use_data = defaultdict(list)

    if mode == "sentence":
        for sentence, sentence_pos_tags in zip(data["sentence"], data["pos_tags"]):
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            potential_word_offsets = []
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word in used_vocab and pos == used_vocab[word][0]:
                    potential_word_offsets.append((word, offset))
            if potential_word_offsets:
                used_word, _ = random.choice(potential_word_offsets)
                offsets = [
                    offset
                    for word, offset in (potential_word_offsets if offsets_matching_pos else word_offsets)
                    if word == used_word
                ]
                word_use_data[used_word].append({"sentence": sentence, "offsets": offsets})

    elif mode == "word":
        remaining_vocab = defaultdict(list)
        for i, (sentence, sentence_pos_tags) in enumerate(zip(data["sentence"], data["pos_tags"])):
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word in used_vocab and pos == used_vocab[word][0]:
                    occurrences = remaining_vocab[word]
                    if not (occurrences and occurrences[-1] == i):
                        occurrences.append(i)
        example_used_by_word = np.full(len(data), None, dtype="object")

        n_round = 0
        while remaining_vocab:
            n_round += 1
            n_needed_examples = 2 if n_round <= 1 else 1

            # shuffle remaining_vocab
            shuffle_remaining_vocab = False
            if shuffle_remaining_vocab:
                remaining_vocab_items = list(remaining_vocab.items())
                random.shuffle(remaining_vocab_items)
                remaining_vocab = dict(remaining_vocab_items)

            for word, occurred_example_idxs in remaining_vocab.items():
                occurred_example_idxs = [i for i in occurred_example_idxs if example_used_by_word[i] is None]
                sampled_example_idxs = occurred_example_idxs[:n_needed_examples]
                if len(sampled_example_idxs) == n_needed_examples:
                    example_used_by_word[sampled_example_idxs] = word
                occurred_example_idxs = occurred_example_idxs[n_needed_examples:]
                remaining_vocab[word] = occurred_example_idxs
            remaining_vocab = {
                word: occurred_example_idxs
                for word, occurred_example_idxs in remaining_vocab.items()
                if occurred_example_idxs
            }

        print(f"left-out examples: {frac_repr((example_used_by_word == None).sum(), len(example_used_by_word))}")

        for sentence, sentence_pos_tags, word_ in zip(data["sentence"], data["pos_tags"], example_used_by_word):
            if not word_:
                continue
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            offsets = []
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word == word_ and (not offsets_matching_pos or pos == used_vocab[word][0]):
                    offsets.append(offset)
            word_use_data[word_].append({"sentence": sentence, "offsets": offsets})

    else:
        raise Exception(f"Unknown build_word_use_data mode {mode}")

    return word_use_data


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", default="childes",
        help="Dataset name on Hugging Face Hub, or path to the dataset."
    )
    argparser.add_argument(
        "--print_original_use_samples", action="store_true",
        help="Print samples of words in the vocabulary with their uses."
    )
    argparser.add_argument(
        "--plot_format", default="pdf",
        help="file format of the plots (e.g., pdf, png)."
    )
    argparser.add_argument(
        "--plot_word_frequency", action="store_true",
        help="Plot word frequency distribution."
    )
    argparser.add_argument(
        "--plot_pos", action="store_true",
        help="Plot POS tag distribution."
    )
    argparser.add_argument(
        "--tokenize_original_dataset", action="store_true",
        help="Tokenize the original dataset and plot length distribution."
    )
    argparser.add_argument(
        "--used_pos", nargs="+",
        default=[
            "NN", "NNS",  # nouns
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbs
            "JJ", "JJR", "JJS",  # adjectives
            "RB", "RBR", "RBS",  # adverbs
        ],
        help="Use words of these used POS tags to build dataset of word uses."
    )
    argparser.add_argument(
        "--punc_pos", nargs="+",
        default=[
            "SYM", ".", ":", "HYPH", "UH"
        ],
        help="POS tags for punctuations."
    )
    argparser.add_argument(
        "--build_word_use_data_mode", choices=["sentence", "word"],
        default="word"
    )
    argparser.add_argument(
        "--offsets_matching_pos", action="store_true",
    )
    argparser.add_argument(
        "--build_word_use_data_from_original_splits", action="store_true",
        help="Build the word use data from each original split independently "
             "and create the splits respectively. Not recommended since word "
             "overlaps between splits are large. "
             "If not set, will merge all original splits, build a single word "
             "use dataset, and then split by words."
    )
    argparser.add_argument(
        "--allow_duplicate_sents", action="store_true",
        help="Do not deduplicate sentences in the dataset."
    )
    argparser.add_argument(
        "--remove_sents_less_than_n_words", type=int, default=1,
        help="Remove sentences with less than this number of words (excluding "
             "punctuations)."
    )
    argparser.add_argument(
        "--split_ratio", type=float, nargs="+",
        help="Split ratio."
    )
    argparser.add_argument(
        "--word_use_data_dir", type=Path, default=Path("word_use_data"),
        help="Directory for word use data."
    )
    argparser.add_argument(
        "--seed", type=int,
        help="Random seed."
    )
    args = argparser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if Path(args.dataset).exists():
        dataset_cache_path = Path(args.dataset)
    else:
        dataset_cache_path = Path('dataset_cache', args.dataset)
    dataset: datasets.DatasetDict = datasets.load_dataset(args.dataset)  # type: ignore
    if "text" in dataset["train"].features:
        dataset = dataset.rename_column("text", "sentence")

    pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()  # type: ignore

    vocab = build_vocab(
        map(pre_tokenizer.pre_tokenize_str, dataset["train"]["sentence"])
    )
    print(f"vocab size: {len(vocab)}")

    pos_tag_dfs = []
    for split, data_split in dataset.items():
        print(f"{split} split:")
        sentences = data_split["sentence"]

        pos_tags = cache(dataset_cache_path/f'{split}.pos_tags.pkl')(get_pos_tags)(sentences, nlp)
        data_split = data_split.add_column("pos_tags", pos_tags)
        dataset[split] = data_split
        pos_tag_df = pd.DataFrame(list(chain.from_iterable(pos_tags)), columns=['pos'])
        pos_tag_df['split'] = split
        pos_tag_dfs.append(pos_tag_df)

        counter = count_tokens(
            map(pre_tokenizer.pre_tokenize_str, sentences)
        )
        counter = sorted_counter_dict(counter)
        freqs = np.array(list(map(len, counter.values())))
        total_n_tokens = freqs.sum().item()
        oov_words = set(counter.keys()) - set(vocab.keys())
        if oov_words:
            print(f"OOV words: {oov_words}")
        print(f"UNK rate: {frac_repr(len(counter[UNK_TOKEN]), total_n_tokens)}")

        if args.print_original_use_samples and split in ["train"]:
            counter_ = list(counter.items())
            for word_i in np.linspace(0, len(counter_)-1, 20, dtype=int):
                word, occurrences = counter_[word_i]
                if True:
                    print(f"word: {word}\tfrequency: {len(occurrences)}")
                max_n = 10
                if len(occurrences) > max_n:
                    occurrences = random.sample(occurrences, max_n)
                for example_i, span in occurrences:
                    sentence = sentences[example_i]
                    sentence = re.sub(r"\b"+word+r"\b", NEW_TOKEN, sentence)
                    print(f"example #{example_i}:", sentence)
                print()

        if args.plot_word_frequency and split in ["train"]:
            n = 100
            counter_ = dict(islice(counter.items(), n))
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.bar(counter_.keys(), list(map(len, counter_.values())), width=1, align="center")  #type: ignore
            ax.set_xlim(xmin=-.5, xmax=n-.5)
            plt.xticks(rotation=270, fontsize=5)
            ax.set_xlabel("Word")
            ax.set_ylabel("Frequency")
            fig.savefig(dataset_cache_path/f"word_frequency.{args.plot_format}", transparent=True) # type: ignore

            m = 100
            freq_counts = np.bincount(freqs)
            freq_counts_cum = np.cumsum(freq_counts)
            freq_counts_cum_ = np.concatenate(([0], freq_counts_cum[:m]))
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(freq_counts_cum_)
            ax.set_xlim(xmin=0, xmax=len(freq_counts_cum_)-1)
            ax.set_ylim(ymin=0, ymax=len(vocab))
            ax.set_xlabel("Frequency")
            ax.set_ylabel("#words")
            fig.savefig(dataset_cache_path/f"word_frequency_cumulative.{args.plot_format}", transparent=True) # type: ignore

    pos_tag_df = pd.concat(pos_tag_dfs)
    extend_pos(pos_tag_df)
    if args.plot_pos:
        for pos_field, figsize in {'POS tag': (20, 10), 'syntactic category': (6, 5)}.items():
            g = sns.catplot(kind='count', data=pos_tag_df, x='split', hue=pos_field, palette=palette, height=figsize[1], aspect=figsize[0]/figsize[1])
            plt.savefig(dataset_cache_path/f"{pos_field} distribution.{args.plot_format}", transparent=True)

    max_freq_pos_vocab = {}
    for split in ["train"]:
        data_split = dataset[split]
        token_pos_vocab = count_token_pos(
            map(pre_tokenizer.pre_tokenize_str, data_split["sentence"]),
            data_split["pos_tags"])
        max_freq_pos_vocab = max_freq_pos_counter_dict(token_pos_vocab)
        pos_vocab = defaultdict(dict)
        for word, (pos, occurrences) in max_freq_pos_vocab.items():
            pos_vocab[pos][word] = occurrences
        for pos, vocab_pos in pos_vocab.items():
            frequency_series = pd.Series({word: len(occurrences) for word, occurrences in vocab_pos.items()})
            df = pd.DataFrame({"frequency": frequency_series})
            path = Path("vocab") / f"{pos}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)

    # build tokenizer
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, UNK_TOKEN)  # type: ignore
    )
    tokenizer.pre_tokenizer = pre_tokenizer # type: ignore
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing( # type: ignore
        single=f"{SEP_TOKEN} $A {SEP_TOKEN}:1",
        pair=f"{SEP_TOKEN} $A {SEP_TOKEN}:1 $B:1 {SEP_TOKEN}:2",
        special_tokens=[(SEP_TOKEN, 2)],
    )
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN),
        pad_token=PAD_TOKEN,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=SEP_TOKEN,
        sep_token=SEP_TOKEN,
    )

    if args.tokenize_original_dataset:
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"]),
            batched=True
        )
        dataset = dataset.map(
            lambda example: {"length": len(example["input_ids"])},
            batched=False
        )
        n_rows, n_cols = len(dataset), 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=False)
        for (split, data_split), ax in zip(dataset.items(), axs):
            plot = sns.histplot(np.array(data_split["length"])-2, binwidth=1, ax=ax)
            ax.set_xlabel("sentence length")
            ax.set_xlim(xmin=0)
            ax.set_title(split)
        fig.savefig(dataset_cache_path/f"length_distribution.{args.plot_format}", transparent=True) # type: ignore

    # build dataset of word uses
    print("Build dataset of word uses:")
    used_vocab = {
        word: (pos, occurrences)
        for word, (pos, occurrences) in max_freq_pos_vocab.items()
        if pos in args.used_pos
    }

    original_dataset = dataset
    if not args.build_word_use_data_from_original_splits:
        # merge original splits
        merged_data = datasets.concatenate_datasets(list(dataset.values()))
        original_dataset = datasets.DatasetDict({"merged": merged_data})

    if not args.allow_duplicate_sents:
        # deduplicate sentences
        print("deduplicating sentences...")
        sentences_set = set()

        def _is_first_occur(example):
            sent = example["sentence"]
            if sent in sentences_set:
                return False
            sentences_set.add(sent)
            return True

        new_original_dataset = {}
        for split in ["train", "validation", "test", "merged"]:
            if split in original_dataset.keys():
                print(f"{split}:")
                data = original_dataset[split]
                new_data = data.filter(_is_first_occur)
                new_original_dataset[split] = new_data
                print(f"{frac_repr(len(new_data), len(data))} left")
        new_original_dataset = datasets.DatasetDict(new_original_dataset)
        original_dataset = new_original_dataset

    if args.remove_sents_less_than_n_words:
        # remove sentences <= N words (excluding punctuations)
        print(f"removing sentences <= {args.remove_sents_less_than_n_words} words (excluding punctuations)...")
        def _more_than_n_words(example):
            return sum(((pos not in args.punc_pos) for pos in example["pos_tags"])) > args.remove_sents_less_than_n_words
        new_original_dataset = original_dataset.filter(_more_than_n_words)
        for split in original_dataset.keys():
            data = original_dataset[split]
            new_data = new_original_dataset[split]
            print(f"{split}: {frac_repr(len(new_data), len(data))} left")
        original_dataset = new_original_dataset

    word_use_dataset = {}
    for split, data in original_dataset.items():
        print(f"building word use data from original {split} split...")
        word_use_data = build_word_use_data(
            data,
            used_vocab,
            mode=args.build_word_use_data_mode,
            offsets_matching_pos=args.offsets_matching_pos,
        )
        word_use_dataset[split] = word_use_data

    if not args.build_word_use_data_from_original_splits:
        if args.split_ratio is None:
            # get original split ratio
            ratio = {
                split: len(data) / len(merged_data)  # type: ignore
                for split, data in dataset.items()
            }
        else:
            sum_ratio = sum(args.split_ratio)
            ratio = {
                split: ratio / sum_ratio
                for split, ratio in zip(["train", "validation", "test"], args.split_ratio)
            }
        # split word use data by words
        word_use_data = word_use_dataset["merged"]
        word_use_data = list(word_use_data.items())
        indices = np.arange(len(word_use_data))  # use indices to keep original order
        np.random.shuffle(indices)
        # get number of examples in each split according to ratio
        splits = list(ratio.keys())
        total_n_examples = len(word_use_data)
        split_n_examples = np.array([
            floor(total_n_examples * ratio[split])
            for split in splits[:-1]
        ])
        split_points = np.cumsum(split_n_examples)
        # split word_use_data by numbers of examples in each split
        word_use_dataset = {}
        for split, split_indices in zip(splits, np.split(indices, split_points)):
            split_indices.sort()
            data = [word_use_data[idx] for idx in split_indices]
            data = dict(data)
            word_use_dataset[split] = data

    word_use_data_path = args.word_use_data_dir / args.dataset / args.build_word_use_data_mode
    word_use_data_path.mkdir(parents=True, exist_ok=True)

    print("word use data info:")
    for split, word_use_data in word_use_dataset.items():
        print(f"{split} split:")
        print(f"#words: {len(word_use_data)}")
        if split != "train":
            n_words_in_train_split = sum(word in word_use_dataset["train"] for word in word_use_data)
            print(f"{frac_repr(n_words_in_train_split, len(word_use_data))} words in train split")
        word_use_n_data = {word: len(uses) for word, uses in word_use_data.items()}
        print(f"#uses distribution: total={sum(word_use_n_data.values())}")
        dist = np.bincount(list(word_use_n_data.values())) # type: ignore
        print(dist)
        sns.displot(list(word_use_n_data.values()), discrete=True, binrange=(0, 50))
        title = f"word uses {split} split distribution"
        plt.title(title)
        plt.savefig(word_use_data_path/f"{title}.png", transparent=True)

    # save word use dataset
    for split, word_use_data in word_use_dataset.items():
        save_path = word_use_data_path / f"{split}.json"
        print(f"save word use data {split} split to {save_path}")
        with open(save_path, "w") as f:
            json.dump(word_use_data, f, indent='\t')