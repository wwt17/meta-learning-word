import argparse
from pathlib import Path
from itertools import islice, chain
import random
import json
from transformers import set_seed
from text_configs import NEW_TOKEN
from utils import batchify, replace_at_offsets


def example_str(example):
    return replace_at_offsets(example["sentence"], example["offsets"], NEW_TOKEN)


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

    if args.seed is not None:
        set_seed(args.seed)

    with open(args.data, "r") as f:
        data = json.load(f)

    k = args.n_study_examples + 1
    data = list(filter(lambda item: len(item[1]) >= k, data.items()))

    n_episodes = 0
    while True:
        try:
            random.shuffle(data)
            for word_uses_batch in batchify(data):
                n_episodes += 1
                print(f"Episode #{n_episodes}:")
                batch_size = len(word_uses_batch)
                word_examples_batch = tuple(((word, random.sample(uses, k=k)) for word, uses in word_uses_batch))
                for idx, (word, examples) in enumerate(word_examples_batch):
                    print(f"word #{idx+1}:")
                    for j, example in enumerate(examples[:-1]):
                        sent = example_str(example)
                        print(4*" " + f"example #{j+1}: " + sent)
                indices = list(range(batch_size))
                random.shuffle(indices)
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