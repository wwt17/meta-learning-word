from collections.abc import Iterable, Mapping
import signal
import argparse
from pathlib import Path
from itertools import islice, combinations
import json
import numpy as np
from utils import frac_repr


use_outlines = False
if use_outlines:
    import outlines
    outlines.caching.disable_cache()  # type: ignore
else:
    from openai import OpenAI
    client = OpenAI()


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)  # type: ignore


input_fields = [
    "ground-truth word",
    "ground-truth prefix",
    "     decoded prefix",
    "ground-truth suffix",
]
output_fields = [
    "greedy outputs",
    "sample with top-p=0.92 outputs",
    "beam search outputs",
][:-1]
fields = input_fields + output_fields


def parse_example_with_generations(
        lines: Iterable[str],
        fields=fields,
        cont_prefix_pattern="cont. {}: ".format,
):
    example = {}
    last_lines = []
    for line in lines:
        field, sep, rest = line.partition(":")
        if sep and field in fields:
            last_lines = [rest]
            example[field.lstrip()] = last_lines
        else:
            last_lines.append(line)
    for field, value_lines in example.items():
        if field.endswith("outputs"):
            assert len(value_lines) >= 1
            assert value_lines[0] == "\n"
            value_lines.pop(0)
            value = []
            n_cont = 0
            cont_prefix = cont_prefix_pattern(n_cont)
            while value_lines:
                assert value_lines[0].startswith(cont_prefix)
                cont = value_lines.pop(0).removeprefix(cont_prefix)
                n_cont += 1
                cont_prefix = cont_prefix_pattern(n_cont)
                while value_lines and not value_lines[0].startswith(cont_prefix):
                    cont += value_lines.pop(0)
                cont = cont.removesuffix("\n")
                value.append(cont)
        else:
            value = "".join(value_lines)
            value = value.removeprefix(" ")
            value = value.removesuffix("\n")
        example[field] = value
    return example


def read_generations(file, example_first_line_pattern="Example #{}:\n".format):
    n_example = 0
    example_first_line = example_first_line_pattern(n_example)
    line = file.readline()
    while True:
        while line and line != example_first_line:
            line = file.readline()
        if not line:  # EOF
            break
        line = file.readline()
        n_example += 1
        example_first_line = example_first_line_pattern(n_example)
        lines = []
        while line and line != example_first_line:
            lines.append(line)
            line = file.readline()
        if lines and lines[-1] == "\n":
            lines.pop()
        yield parse_example_with_generations(lines)


def compare(
        prefix,
        gens,
        judge = "human",
        rng: np.random.Generator = np,  # type: ignore
        prompt_qa_format = "Please answer in a single uppercase letter: Which of the following is a better next example for the word '{word}', or they tie?\nA){}\nB){}\nC){}\nAnswer Choice:".format,
        word = " wug",
        unpermuted_answer_id = None,
):
    indices = rng.permutation(len(gens) + 1) - 1
    permuted_choices = [
        gens[index] if index >= 0 else " Tie"
        for index in indices
    ]
    prompt_qa = prompt_qa_format(*permuted_choices, word=word.strip())
    prompt = prefix + prompt_qa
    if unpermuted_answer_id is not None:
        answer_id = np.where(indices == unpermuted_answer_id)[0][0]
        answer = chr(ord("A") + answer_id)
        print(f"{prompt} {answer}")
    else:
        if isinstance(judge, str) and judge.startswith("gpt"):
            # openai
            completion = client.chat.completions.create(
                model=judge,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ranks models by the quality of their answers."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer: str = completion.choices[0].message.content  # type: ignore
            print(f"{judge} returns:", answer)
            answer = answer.strip()[:1].upper()
        elif isinstance(judge, str):
            # human
            answer = None  # type: ignore
            while answer not in ['A', 'B', 'C']:
                if answer is not None:
                    print("Invalid input. Please re-enter.")
                answer = input(prompt)
                answer = answer.strip()[:1].upper()
        else:
            # outline
            answer = judge.generate_choice(prompt, ['A', 'B', 'C'])
        answer_id = ord(answer) - ord("A")
        unpermuted_answer_id = int(indices[answer_id])
    return unpermuted_answer_id


def get_case_name(winner_id: int, id_to_name: Mapping[int, str]):
    try:
        return f"{id_to_name[winner_id]} wins"
    except KeyError:
        return "tie"


def print_statistics(results, id_to_name: Mapping[int, str], judges):
    for judge in judges:
        print(f"{judge}:")
        for output_field in output_fields:
            print(f"{output_field}:")
            winner_ids = [
                winner_ids[judge]
                for example_results in results
                for winner_ids in example_results[output_field]
            ]
            winner_ids = np.array(winner_ids)
            cases, counts = np.unique(winner_ids, return_counts=True)
            for case, count in zip(cases, counts):
                case_name = get_case_name(case, id_to_name)
                print(f"Case {case_name}: {count}")
    for judge_pair in combinations(judges, 2):
        cnt_eq, cnt_oppo, tot = 0, 0, 0
        for example_results in results:
            for output_field in output_fields:
                for winner_ids in example_results[output_field]:
                    try:
                        winner_id_pair = tuple((winner_ids[judge] for judge in judge_pair))
                    except KeyError:
                        continue
                    tot += 1
                    cnt_eq += winner_id_pair[0] == winner_id_pair[1]
                    cnt_oppo += winner_id_pair[0] >= 0 and winner_id_pair[1] >= 0 and winner_id_pair[0] != winner_id_pair[1]
        print(f"{judge_pair[0]} and {judge_pair[1]} agree on {frac_repr(cnt_eq, tot)} judgments and have {frac_repr(cnt_oppo, tot)} opposite judgments")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "files", type=Path, nargs=2,
        help="The 2 files of evaluation outputs for comparison."
             " The first one should be from the finetuned model;"
             " the second one should be from the original model."
    )
    argparser.add_argument(
        "--word", nargs=2,
        default=["<|reserved_special_token_0|>", " wug"],
        help="Strings representing the new word to be learned,"
             " in the same order of their corresponding files."
    )
    argparser.add_argument(
        "--judges", nargs="+", default=["human"],
        help="Judges. Can be human or OpenAI models."
    )
    argparser.add_argument(
        "--result_file", type=Path, required=True,
        help="JSON file for saving and restoring comparison results."
    )
    argparser.add_argument(
        "--example_order", choices=["original", "shuffle"], default="original",
        help="In which order to present the examples."
    )
    argparser.add_argument(
        "--skip_judged", action="store_true",
        help="Skip judged generations."
    )
    argparser.add_argument(
        "--save_every_n_examples", type=int, default=1,
        help="Save the comparison results every this number of examples."
             " If set to 0, do not save when iterating over examples."
    )
    argparser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed."
    )
    argparser.add_argument(
        "--pause", choices=["none", "example", "judgment"], default="none",
        help="Pause after each judgment, example, or none."
    )
    args = argparser.parse_args()

    rng = np.random.default_rng(args.seed)

    judges = {
        judge: outlines.models.openai(judge) if use_outlines and judge.startswith("gpt") else judge  # type: ignore
        for judge in args.judges
    }

    files = [path.open() for path in args.files]

    try:
        with args.result_file.open() as result_f:
            results = json.load(result_f)
    except FileNotFoundError:
        results = []

    try:
        ft_id = args.word.index("<|reserved_special_token_0|>")
    except ValueError:
        assert False, "Cannot find the output file of a finetuned model"
    raw_id = 1 - ft_id
    id_to_name = {ft_id: "finetuned model", raw_id: "pretrained model"}

    ordered_examples = enumerate(zip(*map(read_generations, files)))
    if args.example_order != "original":
        ordered_examples = list(ordered_examples)
    if args.example_order == "shuffle":
        rng.shuffle(ordered_examples)  # type: ignore

    try:
        for n_example, (i_example, examples) in enumerate(ordered_examples):
            print(f"[{n_example}] Example #{i_example}:")
            for field in ["ground-truth word"]:
                values = [example[field] for example in examples]
                assert len(set(values)) == 1, f"{field} differs: {values}"
            for file_id, (example, word) in enumerate(zip(examples, args.word)):
                assert word in example["ground-truth prefix"], f"Cannot find word '{word}' in file {file_id}"
            prefix = examples[raw_id]["ground-truth prefix"]
            prefix = prefix.removesuffix(" *")

            while len(results) < i_example + 1:
                results.append({})
            example_results = results[i_example]

            for field in output_fields:
                if field not in example_results:
                    example_results[field] = []
                field_results = example_results[field]
                values = [example[field] for example in examples]
                for n_gen, gens in islice(enumerate(zip(*values)), 1):
                    gens = list(gens)
                    while len(field_results) < n_gen + 1:
                        field_results.append({})
                    gens[ft_id] = gens[ft_id].replace(args.word[ft_id], args.word[raw_id])
                    winner_ids = field_results[n_gen]
                    for judge_name, judge_obj in judges.items():
                        winner_id = winner_ids.get(judge_name, None)
                        if args.skip_judged and winner_id is not None:
                            continue
                        winner_id = compare(
                            prefix,
                            gens,
                            judge=judge_obj,
                            rng=rng,
                            word=args.word[raw_id],
                            unpermuted_answer_id=winner_id,
                        )
                        winner_ids[judge_name] = winner_id
                        print(f"{judge_name} prediction: " + get_case_name(winner_id, id_to_name) + "!")
                        if args.pause == "judgment":
                            input()

            if args.save_every_n_examples > 0 and (n_example + 1) % args.save_every_n_examples == 0:
                with DelayedKeyboardInterrupt(), args.result_file.open("w") as result_file:
                    json.dump(results, result_file, indent=2)

            if args.pause == "example":
                input()

    except EOFError:
        pass

    with args.result_file.open("w") as result_file:
        json.dump(results, result_file, indent=2)

    print_statistics(results, id_to_name, args.judges)

    for file in files:
        file.close()