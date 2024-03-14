from typing import Optional, Any
from collections.abc import Iterable, Sequence, Mapping, Sized
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import islice, chain
import json
import numpy as np
import datasets
from utils import zipdict, batchify, example_str


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


def load_dataset(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    data = [{"word": word, "examples": examples} for word, examples in data.items()]
    data = datasets.Dataset.from_list(data)
    return data


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data", type=Path,
        default=Path("word_use_data")/"childes"/"word"/"train.json",
        help="(json) data to load from."
    )
    argparser.add_argument(
        "--n_class", type=int, default=2,
        help="Number of words to classify, i.e., n-way classification."
    )
    argparser.add_argument(
        "--n_study_examples", type=int, default=2,
        help="Number of study examples for each new word."
    )
    argparser.add_argument(
        "--max_sample_times", type=int, default=1,
        help="Max sample times per word."
    )
    argparser.add_argument(
        "--seed", type=int,
        help="Random seed."
    )
    args = argparser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)

    data = load_dataset(args.data)

    n_episodes = 0
    while True:
        try:
            data = data.shuffle(generator=rng)
            for word_examples_batch in batchify(sample_examples(data, args.n_study_examples+1, max_sample_times=args.max_sample_times, rng=rng), batch_size=args.n_class):
                n_episodes += 1
                print(f"Episode #{n_episodes}:")
                batch_size = len(word_examples_batch)
                for idx, item in enumerate(word_examples_batch):
                    word, examples = item["word"], item["examples"]
                    print(f"word #{idx+1}:")
                    for j, example in enumerate(examples[:-1]):
                        sent = example_str(example)
                        print(4*" " + f"example #{j+1}: " + sent)
                indices = list(range(batch_size))
                np.random.shuffle(indices) # type: ignore
                print("predict:")
                for i, idx in enumerate(indices):
                    example = word_examples_batch[idx]["examples"][-1]
                    sent = example_str(example)
                    print(f"{i+1}. " + sent)

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