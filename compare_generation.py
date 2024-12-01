from collections.abc import Iterable, Mapping, Sequence, Callable
import signal
import argparse
from pathlib import Path
from itertools import islice, combinations
import json
import numpy as np
from utils import frac_repr
from read_generations import input_fields, output_fields, fields, read_generations, extract_definition_from_generation


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


used_output_fields = output_fields[:-1]

prompt_qa_format_of_mode = {
    "example": "Please answer in a single uppercase letter: Which of the following is a better next example for the word '{word}', or they tie?\nA){}\nB){}\nC){}\nAnswer Choice:".format,
    "definition": "Please answer in a single uppercase letter: Which of the following is a better definition for the word '{word}', or they tie?\nA){}\nB){}\nC){}\nAnswer Choice:".format,
}


def compare(
        word_example_prompt: str,
        word: str,
        gens: Sequence[str],
        prompt_qa_format: Callable[..., str],
        judge = "human",
        unpermuted_answer_id = None,
        rng: np.random.Generator = np,  # type: ignore
):
    indices = rng.permutation(len(gens) + 1) - 1
    permuted_choices = [
        gens[index] if index >= 0 else " Tie"
        for index in indices
    ]
    prompt_qa = prompt_qa_format(*permuted_choices, word=word.strip())
    prompt = word_example_prompt + prompt_qa
    if unpermuted_answer_id is not None:
        answer_id = np.where(indices == unpermuted_answer_id)[0][0]
        answer = chr(ord("A") + answer_id)
        print(f"{prompt} {answer}")
    else:
        if isinstance(judge, str) and judge.startswith("gpt"):
            # openai
            print(f"prompt:\n{prompt}")
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
            print(f"prompt:\n{prompt}")
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
        for output_field in used_output_fields:
            print(f"{output_field}:")
            winner_ids = [
                winner_ids[judge]
                for example_results in results
                for winner_ids in example_results[output_field]
                if judge in winner_ids
            ]
            winner_ids = np.array(winner_ids)
            cases, counts = np.unique(winner_ids, return_counts=True)
            for case, count in zip(cases, counts):
                case_name = get_case_name(case, id_to_name)
                print(f"Case {case_name}: {frac_repr(count, len(winner_ids))}")
    for judge_pair in combinations(judges, 2):
        cnt_eq, cnt_oppo, tot = 0, 0, 0
        cnt_table = np.zeros((3, 3), dtype=int)
        for example_results in results:
            for output_field in used_output_fields:
                for winner_ids in example_results[output_field]:
                    try:
                        winner_id_pair = tuple((winner_ids[judge] for judge in judge_pair))
                    except KeyError:
                        continue
                    tot += 1
                    cnt_eq += winner_id_pair[0] == winner_id_pair[1]
                    cnt_oppo += winner_id_pair[0] >= 0 and winner_id_pair[1] >= 0 and winner_id_pair[0] != winner_id_pair[1]
                    cnt_table[winner_id_pair[0], winner_id_pair[1]] += 1
        print(f"{judge_pair[0]} and {judge_pair[1]} agree on {frac_repr(cnt_eq, tot)} judgments and have {frac_repr(cnt_oppo, tot)} opposite judgments")
        print("table:")
        for x in [0, 1, -1]:
            for y in [0, 1, -1]:
                print(frac_repr(cnt_table[x, y], tot), end=('\n' if y == -1 else ' '))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "files", type=Path, nargs=2,
        help="The 2 files of evaluation outputs for comparison."
             " The first one should be from the finetuned model;"
             " the second one should be from the original model."
    )
    argparser.add_argument(
        "--mode", choices=list(prompt_qa_format_of_mode.keys()),
        default="example",
        help="Generation mode. Either example or definition."
    )
    argparser.add_argument(
        "--use_gt_word_only", action="store_true",
        help="Show the model only the ground-truth word form in the question,"
             " without in-context learning examples."
    )
    argparser.add_argument(
        "--word", nargs=2,
        default=["<|reserved_special_token_0|>", " wug"],
        help="Strings representing the new word to be learned,"
             " in the same order of their corresponding files."
    )
    argparser.add_argument(
        "--sep", default="\n *",
        help="The sep string."
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

            gt_word = examples[0]['ground-truth word']
            prefix = examples[raw_id]["ground-truth prefix"]
            if args.use_gt_word_only:
                print(f"prefix: {prefix}")
                word_example_prompt = ""
                word = gt_word
            else:
                next_example_start_index = prefix.rfind(args.sep)
                assert next_example_start_index >= 0, "Cannot find the start of the next example (sep) in the prefix:\n{prefix}"
                word_example_prompt = prefix[ : next_example_start_index + 1]
                word = args.word[raw_id]

            while len(results) < i_example + 1:
                results.append({})
            example_results = results[i_example]

            for field in used_output_fields:
                if field not in example_results:
                    example_results[field] = []
                field_results = example_results[field]
                values = [example[field] for example in examples]
                for n_gen, gens in islice(enumerate(zip(*values)), 1):
                    gens = list(gens)
                    while len(field_results) < n_gen + 1:
                        field_results.append({})
                    gens[ft_id] = gens[ft_id].replace(args.word[ft_id], args.word[raw_id])
                    if args.mode == "definition":
                        gens = list(map(
                            extract_definition_from_generation,
                            gens
                        ))
                    winner_ids = field_results[n_gen]
                    for judge_name, judge_obj in judges.items():
                        winner_id = winner_ids.get(judge_name, None)
                        if args.skip_judged and winner_id is not None:
                            continue
                        winner_id = compare(
                            word_example_prompt,
                            word,
                            gens,
                            prompt_qa_format_of_mode[args.mode],
                            judge=judge_obj,
                            unpermuted_answer_id=winner_id,
                            rng=rng,
                        )
                        winner_ids[judge_name] = winner_id
                        print(f"{judge_name} prediction: " + get_case_name(winner_id, id_to_name) + "!")
                        if args.pause == "judgment":
                            input()

            if not args.use_gt_word_only:
                print(f"ground-truth word: {gt_word}")

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