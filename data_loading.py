from typing import Union, Optional, Any, Generator, TextIO
from collections.abc import Iterable, Sequence, Mapping, Sized
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import islice, chain
from functools import partial
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils import frac_repr, zipdict, batchify, cache, count_tokens, sorted_counter_dict, example_str, clean_up_tokenization_spaces_for_example, prepend_to_example
from in_context_format import InContextFormat, add_format_arguments
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS


def get_meta_data_sentences(
        data: datasets.Dataset,
        t: Optional[str],
):
    return (
        example_str(example, t)
        for examples in data["examples"]
        for example in examples
    )


def get_lm_data_sentences(
        data: datasets.Dataset,
):
    return data["sentence"]


def build_vocab(
        pre_tokenized_sentences: Iterable[Sequence[tuple[str, Any]]],
        freq_cutoff: int = 0,
        exclude_tokens: Iterable[str] = set(),
        special_tokens: list[str] = SPECIAL_TOKENS,
        new_tokens: list[str] = NEW_TOKENS,
) -> dict[str, int]:
    vocab = sorted_counter_dict(count_tokens(pre_tokenized_sentences))
    exclude_tokens = set(exclude_tokens) | set(special_tokens)
    vocab = [
        token
        for token, occs in vocab.items()
        if len(occs) > freq_cutoff and token not in exclude_tokens
    ]
    vocab = special_tokens + vocab + new_tokens
    vocab = {token: idx for idx, token in enumerate(vocab)}
    return vocab


def get_tokenizer(
        sentences: Iterable[str],
        freq_cutoff: int = 0,
        exclude_tokens: Iterable[str] = set(),
) -> PreTrainedTokenizerFast:
    pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()  # type: ignore
    vocab = build_vocab(
        map(pre_tokenizer.pre_tokenize_str, sentences),
        freq_cutoff=freq_cutoff,
        exclude_tokens=exclude_tokens,
    )
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, UNK_TOKEN)  # type: ignore
    )
    tokenizer.pre_tokenizer = pre_tokenizer # type: ignore
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
        bos_token=SEP_TOKEN,
        eos_token=SEP_TOKEN,
    )
    return tokenizer


def is_data_tokenizer(tokenizer) -> bool:
    return tokenizer.eos_token == SEP_TOKEN


def read_examples(
        word_pattern: re.Pattern,
        file: Optional[TextIO] = None,
        word_must_occur_in_sentence: bool = False,
) -> Generator[dict, None, None]:
    n_examples = 0
    while True:
        sentence = file.readline().strip() if file is not None else input(f"sentence #{n_examples}: ")
        if not sentence:
            break
        offsets = [match.span() for match in word_pattern.finditer(sentence)]
        if file is None:
            print(f"{offsets=}")
        if word_must_occur_in_sentence and not offsets:
            if file is None:
                print(f"Cannot find occurrence of word pattern '{word_pattern.pattern}'.")
            continue
        example = {"sentence": sentence, "offsets": offsets}
        yield example
        n_examples += 1


def read_and_preprocess_examples(
        word_pattern: re.Pattern,
        file: Optional[TextIO] = None,
        word_must_occur_in_sentence: bool = False,
        clean_up_tokenization_spaces: bool = False,
        prepend: str = "",
) -> Generator[dict, None, None]:
    examples = read_examples(
        word_pattern,
        file=file,
        word_must_occur_in_sentence=word_must_occur_in_sentence,
    )
    if clean_up_tokenization_spaces:
        examples = map(clean_up_tokenization_spaces_for_example, examples)
    if prepend:
        examples = map(partial(prepend_to_example, prepend), examples)
    yield from examples


def read_meta_episode(
        file: Optional[TextIO] = None,
        word_must_occur_in_sentence: bool = False,
        clean_up_tokenization_spaces: bool = False,
        prepend: str = "",
) -> tuple[str, list[dict]]:
    if file is not None:
        word = ''
        while not word:
            line = file.readline()
            if not line:
                raise EOFError
            word = line.strip()
    else:
        word = input("word: ")
    word_pattern = re.compile(re.escape(word))
    examples = read_and_preprocess_examples(
        word_pattern,
        file=file,
        word_must_occur_in_sentence=word_must_occur_in_sentence,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        prepend=prepend,
    )
    examples = list(examples)
    return word, examples


def read_meta_episodes(
        file: Optional[TextIO] = None,
        clean_up_tokenization_spaces: bool = False,
        prepend: str = "",
) -> Generator[dict, None, None]:
    while True:
        try:
            word, examples = read_meta_episode(
                file = file,
                clean_up_tokenization_spaces = clean_up_tokenization_spaces,
                prepend = prepend,
            )
            yield {"word": word, "examples": examples}
        except EOFError:
            break


def sample_examples(
        data: datasets.Dataset,
        n_examples: int,
        max_sample_times: Optional[int] = None,
        rng: Any = np.random,
        column_name: str = "examples",
):
    """For each word (row) in data, sample at most max_sample_times times, each consist n_examples samples.
    Args:
        data: a Dataset containing a column with column_name, which contains list of examples.
        n_examples: Number of examples in each sample.
        rng: Random number generator. If is None, get the first several samples in the same order.
        max_sample_times: Max sample times for each word. If None, sample as many times as allowed.
        column_name: The column name for sampled examples.
    """
    def _get_samples(batch):
        ret = defaultdict(list)
        for row in zipdict(batch):
            examples = row[column_name]
            sample_times = len(examples) // n_examples
            if max_sample_times:
                sample_times = min(sample_times, max_sample_times)
            sample_size = sample_times * n_examples
            sample_examples = (
                examples[:sample_size]
                if rng is None else
                rng.choice(examples, size=sample_size, replace=False)
            )
            for i in range(sample_times):
                for key, value in row.items():
                    if key != column_name:
                        ret[key].append(value)
                ret[column_name].append(list(sample_examples[i*n_examples:(i+1)*n_examples]))
        return ret

    return data.map(_get_samples, batched=True)


def sample_lm_seq(
        data: datasets.Dataset,
        n: int,
        in_context_format: InContextFormat,
        drop_last: bool = True,
        source_column_name: str = "sentence",
        target_column_name: str = "examples",
):
    """Concatenate every n sentences (rows) into one sequence (row) for 
    language modeling.
    Args:
        data: a Dataset containing a column with column_name, which contains 
        a sentence per row.
        n: Number of sentences to be grouped into a batch for a sequence.
        drop_last: Whether to drop the last batch smaller than n.
    """
    def _concat_sentence_batch(batch):
        return {target_column_name: [in_context_format.concat_strs(batch[source_column_name])]}

    return data.map(
        _concat_sentence_batch,
        batched=True, batch_size=n, drop_last_batch=drop_last,
        remove_columns=source_column_name,
    )


def load_meta_dataset(
        data_path,
        clean_up_tokenization_spaces: bool = False,
        prepend: str = "",
) -> datasets.Dataset:
    with open(data_path, "r") as f:
        data = json.load(f)
    if clean_up_tokenization_spaces:
        data = {
            word: list(map(clean_up_tokenization_spaces_for_example, examples))
            for word, examples in data.items()
        }
    if prepend:
        data = {
            word: list(map(partial(prepend_to_example, prepend), examples))
            for word, examples in data.items()
        }
    data = [{"word": word, "examples": examples} for word, examples in data.items()]
    data = datasets.Dataset.from_list(data)
    return data


def load_meta_datasets(
        data_paths: Iterable[Path],
        splits: list[str],
        kwargs: Mapping,
) -> Generator[tuple[str, datasets.Dataset], None, None]:
    for path in data_paths:
        assert path.exists(), f"Data does not exist: {path}"
        if path.is_file():
            yield path.stem, load_meta_dataset(path, **kwargs)
        elif path.is_dir():
            dataset_dict, _ = load_dataset(
                path,
                lm=False,
                splits=splits,
                meta_dataset_kwargs=kwargs,
            )
            yield from dataset_dict.items()
        else:
            assert False, f"Cannot handle data: {path}"


def load_dataset(
        dataset_dir,
        lm: bool = True,
        splits = ["train", "validation", "test"],
        meta_dataset_kwargs = dict(),
):
    meta_dataset = datasets.DatasetDict({
        split: load_meta_dataset(Path(dataset_dir, f"meta.{split}.json"), **meta_dataset_kwargs)
        for split in splits
    })
    if lm:
        lm_dataset: datasets.DatasetDict = datasets.load_dataset(
            str(dataset_dir),
            data_files={split: f"lm.{split}.txt" for split in splits}
        )  # type: ignore
        if "text" in lm_dataset["train"].features:
            lm_dataset = lm_dataset.rename_column("text", "sentence")
    else:
        lm_dataset = None  # type: ignore
    return meta_dataset, lm_dataset


def load_data_tokenizer(
        meta_dataset: Optional[datasets.DatasetDict] = None,
        lm_dataset: Optional[datasets.DatasetDict] = None,
        use_split: str = "train",
        freq_cutoff: int = 0,
        exclude_meta_words: bool = True,
        exclude_tokens: Iterable[str] = ["xxx"],
        cache_dir: Optional[os.PathLike] = None,
) -> PreTrainedTokenizerFast:
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cache_dir)  # type: ignore
    except OSError:
        if meta_dataset is None:
            raise Exception("Must provide meta_dataset for building the data tokenizer.")
        sentences = get_meta_data_sentences(meta_dataset[use_split], t=None)
        if lm_dataset is not None:
            sentences = chain(sentences, get_lm_data_sentences(lm_dataset[use_split]))
        if exclude_meta_words:
            exclude_tokens = chain(
                exclude_tokens,
                chain.from_iterable(
                    (data["word"] for split, data in meta_dataset.items())
                )
            )
        tokenizer = get_tokenizer(
            sentences,
            freq_cutoff=freq_cutoff,
            exclude_tokens=exclude_tokens,
        )
        if cache_dir is not None:
            tokenizer.save_pretrained(cache_dir)
    return tokenizer


def is_single_token_in_vocab(tokenizer, s):
    token_ids = tokenizer(s)['input_ids']
    return len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id


def load_tokenizer(
        name_or_path,
        revision=None,
        data_tokenizer_kwargs: dict = {},
        new_tokens=None,
        info_file=sys.stderr,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            revision=revision,
        )
    except OSError:
        if name_or_path is not None:  # load data tokenizer
            if Path(name_or_path).name == "tokenizer":
                pass
            else:
                m = re.search(r"_data_dir_(.+)_config_", str(name_or_path))
                if m is not None:
                    pretraining_data_dir = Path(*m[1].split(":"))
                    print(f"pretraining data_dir extracted from pretrained_model: {pretraining_data_dir}", file=sys.stderr)
                    name_or_path = Path(pretraining_data_dir, "tokenizer")
                else:
                    raise Exception(f"Cannot extract pretraining data_dir (used to find tokenizer) from pretrained_model {name_or_path}")
            tokenizer = load_data_tokenizer(
                cache_dir=name_or_path,
                **data_tokenizer_kwargs
            )
        else:
            raise Exception(f"Must provide name_or_path for the tokenizer")
    if tokenizer.pad_token_id is None:
        for pad_token in [
            "<|padding|>",
        ]:
            if pad_token in tokenizer.vocab:  # type: ignore
                break
        else:
            print("Warning: use eos token as pad token", file=info_file)
            pad_token = tokenizer.eos_token
        tokenizer.pad_token = pad_token
        print(f"set tokenizer pad token to {tokenizer.pad_token} with id {tokenizer.pad_token_id}", file=info_file)
    print(f"tokenizer: {name_or_path}", file=info_file)
    print(f"tokenizer size: {len(tokenizer)}", file=info_file)
    if new_tokens is not None:
        _new_tokens = list(filter(lambda token: not is_single_token_in_vocab(tokenizer, token), new_tokens))
        n_added_tokens = tokenizer.add_tokens(_new_tokens)
        print(f"added {n_added_tokens} tokens in {new_tokens}", file=info_file)
        print(f"tokenizer size: {len(tokenizer)}", file=info_file)
    else:
        n_added_tokens = 0
    return tokenizer, n_added_tokens


def set_and_get_format(tokenizer, args):
    t = None if args.no_new_token else args.new_word
    if tokenizer is None:
        sep = "\n" + args.sep
    elif not is_data_tokenizer(tokenizer):  # pretrained tokenizer
        sep = "\n" + args.sep
        tokenizer.pad_token = tokenizer.eos_token
    else:  # my own tokenizer
        # must not provide token_type_ids to the model
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        if t is not None:
            t = " " + t  # since we always replace a word in text with leading spaces
        sep = " " + SEP_TOKEN
    in_context_format = InContextFormat(
        t = t,
        sep = sep,
        prompt = args.prompt,
    )
    return in_context_format


def displot(data, discrete=True, binrange=None, y_grid=True, plot_mean=True, mean_color='r', height=4, aspect=2, **kwargs):
    sns.displot(data, discrete=discrete, binrange=binrange, height=height, aspect=aspect, **kwargs)
    if plot_mean:
        mean_value = np.mean(data)
        plt.axvline(mean_value, color=mean_color, linestyle='--', label='mean')
        plt.text(mean_value, plt.gca().get_ylim()[0], f'{mean_value:.2f}', ha='center', va='top', color=mean_color)
        plt.legend()
    if binrange is not None:
        plt.xlim(binrange[0]-0.5, binrange[1]+0.5)
    if y_grid:
        plt.grid(axis="y")


def sentence_stats(sentences: Iterable[str], tokenizer, path, title: str, length_range=None, plot_format="png"):
    lengths = []
    total_n_unks = 0
    for sentence in sentences:
        encoding = tokenizer(sentence)
        input_ids = encoding["input_ids"]
        length = len(input_ids)
        n_unks = sum(idx == tokenizer.unk_token_id for idx in input_ids)
        lengths.append(length)
        total_n_unks += n_unks
    lengths = np.array(lengths)
    n_sentences = len(lengths)
    print(f"{n_sentences=}")
    total_n_tokens = np.sum(lengths)
    print(f"{total_n_tokens=}")
    print(f"length_mean={total_n_tokens/n_sentences:.2f}")
    print(f"unk_rate={frac_repr(total_n_unks, total_n_tokens)}")
    length_dist = np.bincount(lengths)
    print(f"length distribution:\n{length_dist}")
    displot(lengths, binrange=length_range)
    plt.title(title)
    plt.savefig(path/f"{title}.{plot_format}", transparent=True)


def uses_stats(data: datasets.Dataset, path, title: str, n_uses_range=None, plot_format="png"):
    n_words = len(data)
    print(f"{n_words=}")
    n_uses = np.array([len(examples) for examples in data["examples"]])
    total_n_uses = np.sum(n_uses)
    print(f"{total_n_uses=}")
    print(f"n_uses_mean={total_n_uses/n_words:.2f}")
    n_uses_dist = np.bincount(n_uses)
    print(f"n_uses distribution:\n{n_uses_dist}")
    displot(n_uses, binrange=n_uses_range)
    plt.title(title)
    plt.savefig(path/f"{title}.{plot_format}", transparent=True)


def dataset_stats(
        meta_dataset: datasets.DatasetDict,
        lm_dataset: datasets.DatasetDict,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        path,
        t,
        length_range=None,
        n_uses_range=None,
):
    print(f"vocab size: {len(tokenizer)}")
    print("meta data:")
    for split, data in meta_dataset.items():
        print(f"{split} split:")
        uses_stats(data, path, f"meta learning word n_uses {split} distribution", n_uses_range=n_uses_range)
        sentence_stats(get_meta_data_sentences(data, t=t), tokenizer, path, f"meta learning sentence length {split} distribution", length_range=length_range)

    print("lm data:")
    for split, data in lm_dataset.items():
        print(f"{split} split:")
        sentence_stats(get_lm_data_sentences(data), tokenizer, path, f"lm sentence length {split} distribution", length_range=length_range)


def tokenized_text(tokenizer, text, skip_special_tokens=False, **kwargs):  # TODO: skip other special tokens but retain the new word
    encoding = tokenizer(text)
    return tokenizer.decode(
        encoding["input_ids"],
        skip_special_tokens=skip_special_tokens,
        **kwargs
    )


def print_data(
        data: datasets.Dataset,
        n_examples: int,
        max_sample_times: int,
        in_context_format: InContextFormat,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        clean_up_tokenization_spaces: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = False,
        pause: bool = False,
):
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    n_episodes = 0
    try:
        if shuffle:
            data = data.shuffle(generator=rng)
        for item in sample_examples(data, n_examples, max_sample_times=max_sample_times, rng=rng):
            n_episodes += 1
            print(f"Episode #{n_episodes}:")
            word, examples = item["word"], item["examples"]  # type: ignore
            text = in_context_format.concat_examples(
                examples,
                start_with_sep=True,
                end_with_sep=False,
                start_index=1,
            )
            if tokenizer is not None:
                text = tokenized_text(
                    tokenizer,
                    text,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
            print(f"{word}:" + text)

            if pause:
                input()

    except EOFError:
        print()


def interactive_classification(
        data: datasets.Dataset,
        n_class: int,
        n_study_examples: int,
        max_sample_times: int,
        in_context_format: InContextFormat,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        clean_up_tokenization_spaces: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = False,
):
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    n_episodes = 0
    while True:
        try:
            if shuffle:
                data = data.shuffle(generator=rng)
            for word_examples_batch in batchify(sample_examples(data, n_study_examples+1, max_sample_times=max_sample_times, rng=rng), batch_size=n_class):
                n_episodes += 1
                print(f"Episode #{n_episodes}:")
                batch_size = len(word_examples_batch)
                for idx, item in enumerate(word_examples_batch):
                    word, examples = item["word"], item["examples"]
                    text = in_context_format.concat_examples(
                        examples[:-1],
                        start_with_sep=True,
                        end_with_sep=False,
                        start_index=1,
                    )
                    if tokenizer is not None:
                        text = tokenized_text(
                            tokenizer,
                            text,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    print(f"word #{idx+1}:" + text)
                indices = list(range(batch_size))
                np.random.shuffle(indices) # type: ignore
                print("predict:")
                for i, idx in enumerate(indices):
                    examples = word_examples_batch[idx]["examples"]
                    text = in_context_format.concat_examples(
                        examples[-1:],
                        start_with_sep=False,
                        end_with_sep=False,
                        start_index=n_study_examples+1,
                    )
                    if tokenizer is not None:
                        text = tokenized_text(
                            tokenizer,
                            text,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    print(f"{i+1}. " + text)

                while True:
                    try:
                        input_str = input("predicted word indices: ")
                        predicted_indices = [int(x)-1 for x in input_str.strip().split()]
                    except ValueError:
                        print("Invalid input: input are not numbers.")
                    else:
                        if sorted(predicted_indices) == list(range(batch_size)):
                            break
                        print(f"Invalid input: input is not a permutation from 1 to {batch_size}.")

                correct = predicted_indices == indices
                print("Correct!" if correct else "INCORRECT!")
                print("Words in original order:", ' '.join(word_examples_batch[idx]["word"] for idx in range(len(word_examples_batch))))
                print("Correct predicted words:", ' '.join(word_examples_batch[idx]["word"] for idx in indices))
                print("   Your predicted words:", ' '.join(word_examples_batch[idx]["word"] for idx in predicted_indices))
                #print()
                input()

        except EOFError:
            print()
            break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "mode", choices=["stat", "print", "class"],
    )
    argparser.add_argument(
        "--data_dir", type=Path,
        default=Path("word_use_data", "childes", "word"),
        help="Path to the dataset directory."
    )
    argparser.add_argument(
        "--split",
        default="train",
        help="Which split to use."
    )
    argparser.add_argument(
        "--tokenizer",
        help="Tokenizer to use. If it is 'build' (default in stat mode), build (but not cache) the data tokenizer."
    )
    argparser.add_argument(
        "--freq_cutoff", type=int, default=0,
        help="Remove words with frequency <= freq_cutoff from the vocabulary."
    )
    argparser.add_argument(
        "--include_meta_words", action="store_true",
        help="Include words for meta learning in the vocabulary."
    )
    argparser.add_argument(
        "--length_range", type=int, nargs=2, default=(0, 50),
        help="Plot length distribution range."
    )
    argparser.add_argument(
        "--n_uses_range", type=int, nargs=2, default=(5, 100),
        help="Plot n_uses distribution range."
    )
    argparser.add_argument(
        "--n_class", type=int, default=2,
        help="Number of words to classify, i.e., n-way classification."
    )
    argparser.add_argument(
        "--n_examples", type=int, default=5,
        help="Number of study examples for each new word."
    )
    argparser.add_argument(
        "--max_sample_times", type=int, default=1,
        help="Max sample times per word."
    )
    group = argparser.add_argument_group("In-context format")
    add_format_arguments(
        group,
        new_word = " " + NEW_TOKEN,
        sep = " *",
    )
    group.add_argument(
        "--enum_examples", action="store_true",
    )
    argparser.add_argument(
        "--clean_up_tokenization_spaces", action="store_true",
        help="Clean up tokenization spaces. Only useful and better for non-word-based tokenizer."
    )
    argparser.add_argument(
        "--seed", type=int,
        help="Random seed."
    )
    argparser.add_argument(
        "--shuffle", action="store_true",
    )
    argparser.add_argument(
        "--pause", action="store_true",
    )
    args = argparser.parse_args()
    if args.mode == "stat" and args.tokenizer is None:
        args.tokenizer = "build"

    meta_dataset, lm_dataset = load_dataset(
        args.data_dir,
        meta_dataset_kwargs = dict(
            clean_up_tokenization_spaces = args.clean_up_tokenization_spaces and args.tokenizer != "build",
            prepend = args.prepend,
        )
    )
    data_tokenizer_kwargs = dict(
        meta_dataset = meta_dataset,
        lm_dataset = lm_dataset,
        freq_cutoff = args.freq_cutoff,
        exclude_meta_words = not args.include_meta_words,
    )
    if args.tokenizer is None:
        tokenizer = None
    elif args.tokenizer == "build":
        tokenizer = load_data_tokenizer(
            **data_tokenizer_kwargs,  # type: ignore
        )
    else:
        tokenizer, n_added_tokens = load_tokenizer(
            args.tokenizer,
            data_tokenizer_kwargs=data_tokenizer_kwargs,
        )

    in_context_format = set_and_get_format(tokenizer, args)
    if args.enum_examples:
        in_context_format.sep_formatter = "\n    example #{}: ".format

    if args.mode == "stat":
        dataset_stats(
            meta_dataset,
            lm_dataset,
            tokenizer,  # type: ignore
            args.data_dir,
            in_context_format.t,
            length_range = args.length_range,
            n_uses_range = args.n_uses_range,
        )

    elif args.mode == "print":
        print_data(
            meta_dataset[args.split],
            args.n_examples,
            args.max_sample_times,
            in_context_format,
            tokenizer = tokenizer,
            clean_up_tokenization_spaces = args.clean_up_tokenization_spaces,
            seed = args.seed,
            shuffle = args.shuffle,
            pause = args.pause,
        )

    elif args.mode == "class":
        interactive_classification(
            meta_dataset[args.split],
            args.n_class,
            args.n_examples-1,
            args.max_sample_times,
            in_context_format,
            tokenizer = tokenizer,
            clean_up_tokenization_spaces = args.clean_up_tokenization_spaces,
            seed = args.seed,
            shuffle = args.shuffle,
        )

    else:
        raise Exception(f"Unknown mode {args.mode}")