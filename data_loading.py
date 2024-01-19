import argparse
from pathlib import Path
from itertools import islice, chain
import random
import json
from transformers import set_seed


def paired(g, n=2):
    g = iter(g)
    while True:
        p = tuple(islice(g, n))
        if len(p) < n:
            break
        yield p


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data", type=Path,
        default=Path("word_use_data")/"ptb_text_only"/"word"/"train.json",
        help="(json) data to load from."
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
            for word_uses_pair in paired(data):
                n_episodes += 1
                print(f"Episode #{n_episodes}:")
                word_examples_pair = tuple(((word, random.sample(uses, k=k)) for word, uses in word_uses_pair))
                for idx, (word, examples) in enumerate(word_examples_pair):
                    print(f"word #{idx+1}:")
                    for j, example in enumerate(examples[:-1]):
                        print(4*" " + f"example #{j+1}: " + example)
                indices = list(range(len(word_examples_pair)))
                random.shuffle(indices)
                print("predict:")
                for i, idx in enumerate(indices):
                    example = word_examples_pair[idx][1][-1]
                    print(f"{i+1}. " + example)

                valid_input = False
                predicted_indices = [0, 1]
                while not valid_input:
                    prediction = input("prediction (0: keep order; 1: reverse): ").strip()
                    if prediction in ["0"]:
                        valid_input = True
                    elif prediction in ["1"]:
                        valid_input = True
                        predicted_indices = [1, 0]
                    else:
                        print("Invalid input. Try again.")

                correct = predicted_indices == indices
                print("Correct!" if correct else "INCORRECT!")
                print("Words in original order:", ' '.join(word_examples_pair[idx][0] for idx in range(len(word_examples_pair))))
                print("Correct predicted words:", ' '.join(word_examples_pair[idx][0] for idx in indices))
                #print()
                input()

        except EOFError:
            print()
            break