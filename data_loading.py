from collections.abc import Iterable, Sequence, Mapping, Sized
import argparse
from pathlib import Path
from itertools import islice, chain
import json
import numpy as np
from text_configs import NEW_TOKEN
from utils import batchify, replace_at_offsets


def example_str(example):
    return replace_at_offsets(example["sentence"], example["offsets"], NEW_TOKEN)


def build_eval_test_batches(data: Iterable[tuple[str, list]], n_class: int, n_study_examples: int, rng: np.random.Generator):
    k = n_study_examples + 1
    yield from batchify(
        map(
            lambda item: (item[0], list(rng.choice(item[1], size=k, replace=False))),
            filter(lambda item: len(item[1]) >= k, data)
        ),
        batch_size=n_class
    )


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
        "--seed", type=int,
        help="Random seed."
    )
    args = argparser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)

    with open(args.data, "r") as f:
        data = json.load(f)

    data = list(data.items())

    n_episodes = 0
    while True:
        try:
            rng.shuffle(data)
            for word_examples_batch in build_eval_test_batches(data, args.n_class, args.n_study_examples, rng):
                n_episodes += 1
                print(f"Episode #{n_episodes}:")
                batch_size = len(word_examples_batch)
                for idx, (word, examples) in enumerate(word_examples_batch):
                    print(f"word #{idx+1}:")
                    for j, example in enumerate(examples[:-1]):
                        sent = example_str(example)
                        print(4*" " + f"example #{j+1}: " + sent)
                indices = list(range(batch_size))
                np.random.shuffle(indices) # type: ignore
                print("predict:")
                for i, idx in enumerate(indices):
                    example = word_examples_batch[idx][1][-1]
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
                print("Words in original order:", ' '.join(word_examples_batch[idx][0] for idx in range(len(word_examples_batch))))
                print("Correct predicted words:", ' '.join(word_examples_batch[idx][0] for idx in indices))
                print("   Your predicted words:", ' '.join(word_examples_batch[idx][0] for idx in predicted_indices))
                #print()
                input()

        except EOFError:
            print()
            break