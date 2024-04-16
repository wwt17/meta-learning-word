from typing import Any, Optional, Union, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
from enum import IntEnum
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import random, re, json
from math import floor, ceil
import tqdm
import datasets
import tokenizers
import transformers
from transformers import set_seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import spacy
import seaborn as sns
from utils import frac_repr, normalize_dict, cache, get_pos_tags, plot_bar, replace_at_offsets
from pos_tags import extend_pos, pos_mappings
from plotting import palette
from text_configs import NEW_TOKEN


nlp = spacy.load("en_core_web_trf", exclude=["parser", "attribute_ruler", "lemmatizer", "ner"])

def nlp_tokenizer(sentence, keep_original_spaces=False):
    word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
    words, offsets = zip(*word_offsets) if word_offsets else ([], [])
    if keep_original_spaces:
        spaces = [
            (offsets[i][1] < (offsets[i+1][0] if i+1 < len(offsets) else len(sentence)))
            for i in range(len(offsets))
        ]
    else:
        spaces = [True] * len(words)
    doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces) # type: ignore
    return doc

nlp.tokenizer = nlp_tokenizer


def count_tokens(
        pre_tokenized_sentences: Iterable[Sequence[tuple[str, Any]]]
):
    counter = defaultdict(list)
    for i, word_offsets in enumerate(pre_tokenized_sentences):
        for word, offset in word_offsets:
            counter[word].append((i, offset))
    return counter


def count_token_pos(
        pre_tokenized_sentences: Iterable[Sequence[tuple[str, Any]]],
        tags: Iterable[Sequence[Any]],
):
    counter = defaultdict(list)
    tag_counter = defaultdict(lambda: defaultdict(int))
    for i, (word_offsets, tags_) in enumerate(zip(pre_tokenized_sentences, tags)):
        assert len(word_offsets) == len(tags_), f"{word_offsets=} {tags_=}"
        for (word, offset), tag in zip(word_offsets, tags_):
            counter[word].append((i, offset))
            tag_counter[word][tag] += 1
    return counter, tag_counter


KeyType = TypeVar('KeyType')
def sorted_counter_list(counter: Mapping[KeyType, Sized]) -> list[tuple[KeyType, Sized]]:
    return sorted(counter.items(), key=lambda item: (-len(item[1]), item[0]))

def sorted_counter_dict(counter: Mapping[KeyType, Sized]) -> dict[KeyType, Sized]:
    return dict(sorted_counter_list(counter))

TagType = TypeVar('TagType')
def get_max_freq_tag(tag_counts: Mapping[TagType, int]) -> tuple[TagType, int]:
    return min(tag_counts.items(), key=(lambda item: -item[1]))

def get_max_freq_tag_vocab(tag_counter: Mapping[KeyType, Mapping[TagType, int]]) -> Mapping[KeyType, tuple[TagType, int]]:
    return {
        word: get_max_freq_tag(tag_counts)
        for word, tag_counts in tag_counter.items()
    }


class MatchingPosLevel(IntEnum):
    none = 0
    assign = 1
    offsets = 2

    def __str__(self):
        return self.name


def build_word_use_data(
        data: datasets.Dataset,
        used_vocab_pos: Mapping[str, str],
        mode: str = "word",
        min_n_examples: int = 2,
        remaining_vocab_order: str = "sort",
        matching_pos_level: MatchingPosLevel = MatchingPosLevel.none,
) -> tuple[Mapping[str, list[dict]], datasets.Dataset, np.ndarray]:
    meta_data = defaultdict(list)
    example_used_by_word = np.full(len(data), None, dtype="object")

    if mode == "sentence":
        for i, (sentence, sentence_pos_tags) in enumerate(zip(data["sentence"], data["pos_tags"])):
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            potential_word_offsets = []
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word in used_vocab_pos and (matching_pos_level < MatchingPosLevel.assign or pos == used_vocab_pos[word]):
                    potential_word_offsets.append((word, offset))
            if potential_word_offsets:
                used_word, _ = random.choice(potential_word_offsets)
                example_used_by_word[i] = used_word
                offsets = [
                    offset
                    for word, offset in (potential_word_offsets if matching_pos_level >= MatchingPosLevel.offsets else word_offsets)
                    if word == used_word
                ]
                meta_data[used_word].append({"sentence": sentence, "offsets": offsets})

    elif mode == "word":
        remaining_vocab = defaultdict(list)
        for i, (sentence, sentence_pos_tags) in enumerate(zip(data["sentence"], data["pos_tags"])):
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word in used_vocab_pos and (matching_pos_level < MatchingPosLevel.assign or pos == used_vocab_pos[word]):
                    occurrences = remaining_vocab[word]
                    if not (occurrences and occurrences[-1] == i):
                        occurrences.append(i)

        def _filter_unused_idxs(idxs):
            return [i for i in idxs if example_used_by_word[i] is None]

        n_round = 0
        while remaining_vocab:
            n_round += 1
            n_needed_examples = min_n_examples if n_round <= 1 else 1

            if remaining_vocab_order == "shuffle":
                remaining_vocab_items = list(remaining_vocab.items())
                random.shuffle(remaining_vocab_items)
                remaining_vocab = dict(remaining_vocab_items)
            elif remaining_vocab_order == "sort":
                remaining_vocab_items = list(remaining_vocab.items())
                remaining_vocab_items.sort(key=lambda item: (len(item[1]), item[0]))
                remaining_vocab = dict(remaining_vocab_items)
            elif remaining_vocab_order == "none":
                pass
            else:
                raise Exception(f"Unknown remaining_vocab_order {remaining_vocab_order}")

            for word, occurred_example_idxs in remaining_vocab.items():
                occurred_example_idxs = _filter_unused_idxs(occurred_example_idxs)
                sampled_example_idxs = occurred_example_idxs[:n_needed_examples]
                if len(sampled_example_idxs) == n_needed_examples:
                    example_used_by_word[sampled_example_idxs] = word
                occurred_example_idxs = occurred_example_idxs[n_needed_examples:]
                remaining_vocab[word] = occurred_example_idxs
            remaining_vocab = {
                word: _filter_unused_idxs(occurred_example_idxs)
                for word, occurred_example_idxs in remaining_vocab.items()
                if occurred_example_idxs
            }

        for sentence, sentence_pos_tags, word_ in zip(data["sentence"], data["pos_tags"], example_used_by_word):
            if not word_:
                continue
            word_offsets = pre_tokenizer.pre_tokenize_str(sentence)
            offsets = []
            for (word, offset), pos in zip(word_offsets, sentence_pos_tags):
                if word == word_ and (matching_pos_level < MatchingPosLevel.offsets or pos == used_vocab_pos[word]):
                    offsets.append(offset)
            meta_data[word_].append({"sentence": sentence, "offsets": offsets})

    else:
        raise Exception(f"Unknown build_word_use_data mode {mode}")

    print(f"left-out examples: {frac_repr((example_used_by_word == None).sum(), len(example_used_by_word))}")
    leftout_data = data.filter(lambda example, idx: example_used_by_word[idx] == None, with_indices=True)

    return meta_data, leftout_data, example_used_by_word


def split_dataset(
        dataset,
        ratio: Mapping[Any, float],
        kind: str = "none",
        rng: Optional[np.random.Generator] = None,
):
    dataset_size = len(dataset)

    # get number of examples in each split according to ratio
    splits = list(ratio.keys())
    split_sizes = np.array([
        floor(dataset_size * ratio[split])
        for split in splits[:-1]
    ])
    split_points = np.cumsum(split_sizes)

    # assigning datapoint indices to splits
    if kind == "none":
        indices = np.arange(dataset_size)
    elif kind in ["shuffle", "random"]:
        assert rng is not None, "Must provide rng"
        indices = rng.permutation(dataset_size)
        if kind == "random":  # keep original order
            for split_indices in np.split(indices, split_points):
                split_indices.sort()
    else:
        raise Exception(f"Unknown split kind {kind}")

    # build dataset dict
    if isinstance(dataset, datasets.Dataset):
        dataset_dict = datasets.DatasetDict({
            split: dataset.select(indices=split_indices)
            for split, split_indices in zip(splits, np.split(indices, split_points))
        })
    else:
        _convert = type(dataset)
        if isinstance(dataset, dict):
            if isinstance(dataset, defaultdict):
                _convert = partial(defaultdict, dataset.default_factory)
            dataset = list(dataset.items())
        dataset_dict = {
            split: _convert((dataset[idx] for idx in split_indices))
            for split, split_indices in zip(splits, np.split(indices, split_points))
        }
    return dataset_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", default="childes",
        help="Dataset name on Hugging Face Hub, or path to the dataset."
    )
    argparser.add_argument(
        "--lower", action="store_true",
        help="Lowercase all text."
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
        "--used_pos", nargs="+",
        default=[
            "NN", "NNS", "NNP", "NNPS",  # nouns
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
        "--matching_pos_level", type=MatchingPosLevel.__getitem__, choices=list(MatchingPosLevel), default=MatchingPosLevel.none
    )
    argparser.add_argument(
        "--min_n_examples", type=int, default=5,
        help="Minimum number of examples per word for meta learning."
    )
    argparser.add_argument(
        "--max_freq", type=int,
        help="Maximum frequncy of word for meta learning. No limit if not "
             "specified."
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
        "--split_ratio", type=float, nargs="+", default=[8, 1, 1],
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
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(args.dataset)  # type: ignore
    dataset = dataset_dict["train"]

    # merge original splits
    merging_original_splits = True
    if merging_original_splits:
        dataset = datasets.concatenate_datasets(list(dataset_dict.values()))

    if "text" in dataset.features:
        dataset = dataset.rename_column("text", "sentence")
    raw_dataset = dataset
    if args.lower:
        dataset = dataset.map(
            lambda example: {"sentence": example["sentence"].lower()},
            desc="lowercase"
        )

    pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()  # type: ignore

    pos_tags = cache(dataset_cache_path/f'pos_tags.pkl')(get_pos_tags)(raw_dataset["sentence"], nlp)
    dataset = dataset.add_column("pos_tags", pos_tags)  # type: ignore

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

        new_dataset = dataset.filter(_is_first_occur)
        print(f"{frac_repr(len(new_dataset), len(dataset))} left")
        dataset = new_dataset

    if args.remove_sents_less_than_n_words:
        # remove sentences <= N words (excluding punctuations)
        print(f"removing sentences <= {args.remove_sents_less_than_n_words} words (excluding punctuations)...")
        def _more_than_n_words(example):
            return sum(((pos not in args.punc_pos) for pos in example["pos_tags"])) > args.remove_sents_less_than_n_words
        new_dataset = dataset.filter(_more_than_n_words)
        print(f"{frac_repr(len(new_dataset), len(dataset))} left")
        dataset = new_dataset

    pos_tags = dataset["pos_tags"]

    if args.plot_pos:
        pos_tag_df = pd.DataFrame(list(chain.from_iterable(pos_tags)), columns=['pos'])
        pos_tag_df['split'] = "merged"
        extend_pos(pos_tag_df)
        for pos_field, figsize in {'POS tag': (20, 10), 'syntactic category': (6, 5)}.items():
            g = sns.catplot(kind='count', data=pos_tag_df, x='split', hue=pos_field, palette=palette, height=figsize[1], aspect=figsize[0]/figsize[1])
            plt.savefig(dataset_cache_path/f"{pos_field} distribution.{args.plot_format}", transparent=True)

    pre_tokenized_sentences = map(
        pre_tokenizer.pre_tokenize_str,
        tqdm.tqdm(dataset["sentence"], desc="count tokens"))
    if pos_tags is not None:
        vocab, pos_counter = count_token_pos(pre_tokenized_sentences, pos_tags)
        max_freq_pos_vocab = get_max_freq_tag_vocab(pos_counter)
    else:
        vocab = count_tokens(pre_tokenized_sentences)
        pos_counter = {}
        max_freq_pos_vocab = {
            word: (".", len(occurrences))
            for word, occurrences in vocab.items()
        }
    vocab = sorted_counter_dict(vocab)
    freq_vocab = {key: len(value) for key, value in vocab.items()}
    total_n_tokens = sum(freq_vocab.values())
    print(f"n_types={len(vocab)} {total_n_tokens=}")
    vocab_dir = dataset_cache_path/f"vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    frequency_series = pd.Series({word: len(occurrences) for word, occurrences in vocab.items()})
    tag_series = pd.Series({word: max_freq_pos_vocab[word][0] for word in vocab})
    df = pd.DataFrame({"frequency": frequency_series, "tag": tag_series})
    df.to_csv(vocab_dir/"all.csv")

    if pos_tags is not None:
        pos_vocab = defaultdict(dict)
        for word, (pos, freq) in max_freq_pos_vocab.items():
            pos_vocab[pos][word] = freq
        for pos, vocab_pos in pos_vocab.items():
            frequency_series = pd.Series(vocab_pos)
            df = pd.DataFrame({"frequency": frequency_series})
            df.to_csv(vocab_dir/f"{pos}.csv")


    if args.print_original_use_samples:
        counter_ = list(vocab.items())
        for word_i in np.linspace(0, len(counter_)-1, 20, dtype=int):
            word, occurrences = counter_[word_i]
            if True:
                print(f"word: {word}\tfrequency: {len(occurrences)}")
            max_n = 10
            if len(occurrences) > max_n:
                occurrences = random.sample(occurrences, max_n)
            for example_i, span in occurrences:
                sentence = dataset[example_i]["sentence"]
                sentence = sentence.split()
                sentence = " ".join((NEW_TOKEN if word_ == word else word_ for word_ in sentence))
                print(f"example #{example_i}:", sentence)
            print()

    if args.plot_word_frequency:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        x = list(freq_vocab.keys())
        y = list(freq_vocab.values())
        hue = [pos_mappings['syntactic category'][max_freq_pos_vocab[word][0]] for word in freq_vocab.keys()]
        plot_bar(
            x, y,
            hue=hue, palette=palette,
            x_max=100,
            x_tick_step=1,
            legend=True,
            ax=ax,
        )
        plt.xticks(rotation=270, fontsize=5)
        plt.yticks(fontsize=5)
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency")
        fig.savefig(dataset_cache_path/f"word_frequency.top.{args.plot_format}", transparent=True) # type: ignore

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        plot_bar(
            np.arange(len(y)), y,
            hue=hue, palette=palette,
            x_tick_step=len(y)//20,
            y_max=y[len(y)//20],
            y_tick_step=5,
            legend=True,
            ax=ax,
        )
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency")
        fig.savefig(dataset_cache_path/f"word_frequency.all.{args.plot_format}", transparent=True) # type: ignore

    ### build dataset of word uses
    print("Build dataset of word uses:")
    if args.max_freq is not None:
        print(f"Filter only words with frequency <= {args.max_freq} for meta learning")
        used_vocab_pos = {
            word: pos
            for word, (pos, _) in max_freq_pos_vocab.items()
            if len(vocab[word]) <= args.max_freq
        }
        print(f"{frac_repr(len(used_vocab_pos), len(max_freq_pos_vocab))} left")
    else:
        used_vocab_pos = {
            word: pos
            for word, (pos, _) in max_freq_pos_vocab.items()
        }
    print("Filter only words in used pos for meta learning")
    used_vocab_pos_ = {
        word: pos
        for word, pos in used_vocab_pos.items()
        if pos in args.used_pos
    }
    print(f"{frac_repr(len(used_vocab_pos_), len(used_vocab_pos))} left")
    used_vocab_pos = used_vocab_pos_

    print(f"building word use dataset...")
    meta_data, lm_data, used_by_word = build_word_use_data(
        dataset,
        used_vocab_pos,  # type: ignore
        mode=args.build_word_use_data_mode,
        min_n_examples=args.min_n_examples,
        remaining_vocab_order="sort",
        matching_pos_level=args.matching_pos_level,
    )

    ratio = normalize_dict(dict(zip(["train", "validation", "test"], args.split_ratio)))
    meta_dataset = split_dataset(
        meta_data, ratio, kind="random", rng=np.random.default_rng(args.seed))
    lm_dataset = split_dataset(
        lm_data, ratio, kind="random", rng=np.random.default_rng(args.seed))

    word_use_data_path = args.word_use_data_dir / args.dataset / args.build_word_use_data_mode
    word_use_data_path.mkdir(parents=True, exist_ok=True)

    print("word use data info:")
    print("meta learning data:")
    for split, meta_data in meta_dataset.items():
        print(f"{split} split:")
        print(f"#words: {len(meta_data)}")
        word_use_n_data = {word: len(uses) for word, uses in meta_data.items()}
        print(f"#uses distribution: total={sum(word_use_n_data.values())}")
        dist = np.bincount(list(word_use_n_data.values())) # type: ignore
        print(dist)
        sns.displot(list(word_use_n_data.values()), discrete=True, binrange=(0, 50))
        title = f"meta learning word n_uses {split} distribution"
        plt.title(title)
        plt.savefig(word_use_data_path/f"{title}.png", transparent=True)
    print("language modeling data:")
    for split, lm_data in lm_dataset.items():
        print(f"{split} split:")
        print(f"#sentences: {len(lm_data)}")

    # save word use dataset
    for split, meta_data in meta_dataset.items():
        save_path = word_use_data_path / f"meta.{split}.json"
        print(f"save meta learning data {split} split to {save_path}")
        with open(save_path, "w") as f:
            json.dump(meta_data, f, indent='\t')
    for split, lm_data in lm_dataset.items():
        save_path = word_use_data_path / f"lm.{split}.txt"
        print(f"save language modeling data {split} split to {save_path}")
        with open(save_path, "w") as f:
            for sentence in lm_data["sentence"]:
                print(sentence, file=f)