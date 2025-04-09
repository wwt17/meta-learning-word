from collections.abc import Iterable, Mapping, Sequence, Callable
from collections import namedtuple
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
            while not answer[0].isalpha():
                answer = answer[1:]
            answer = answer[0].upper()
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


OutFile = namedtuple("OutFile", ["path", "word", "sep"])
named_files = {
    "llama-3 baseline on BabyLM-10M": OutFile(
        Path("ckpt/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_n_examples_5_max_new_tokens_100/slurm.out"),
        " dax", "\n *"
    ),
    "llama-3 finetuned on BabyLM-10M": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_n_examples_5_max_new_tokens_100/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-3-instruct baseline on BabyLM-10M": OutFile(
        Path("ckpt/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_n_examples_5_max_new_tokens_100/slurm.out"),
        " dax", "\n *"
    ),
    "llama-3-instruct finetuned on BabyLM-10M": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_n_examples_5_max_new_tokens_100/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-2 finetuned on BabyLM-10M": OutFile(
        Path("ckpt/meta-word_pretrained_model_Llama-2-7b-hf_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_5_train_max_length_160_batch_size_16_lr_0.003_seed_0/best/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_n_examples_5_max_new_tokens_100/slurm.out"),
        "<|new_word|>", "\n<|sep|>"
    ),
    "college on BabyLM-10M": OutFile(
        Path("ckpt/meta-word-eval_data_dir_word_use_data:babylm_data:babylm_10M:word_split_test_pretrained_model_Llama-2-7b-hf_emb_gen_model_type_college_n_examples_5_max_new_tokens_100/slurm.out"),
        "<nonce>", "\n"
    ),
    "llama-3 baseline on Chimera": OutFile(
        Path("ckpt/meta-word-eval_data_dir_chimeras.json_data_order_original_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_prompt__n_examples_5_longer/slurm.out"),
        " wug", "\n *"
    ),
    "llama-3 finetuned on Chimera": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_chimeras.json_data_order_original_n_examples_5_longer/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-3-instruct baseline on Chimera": OutFile(
        Path("ckpt/meta-word-eval_data_dir_chimeras.json_data_order_original_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_n_examples_5/slurm.out"),
        " wug", "\n *"
    ),
    "llama-3-instruct finetuned on Chimera": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_chimeras.json_data_order_original_n_examples_5_max_new_tokens_100/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-2 finetuned on Chimera": OutFile(
        Path("ckpt/meta-word_pretrained_model_Llama-2-7b-hf_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_5_train_max_length_160_batch_size_16_lr_0.003_seed_0/best/meta-word-eval_data_dir_chimeras.json_data_order_original_n_examples_5_max_new_tokens_100/slurm.out"),
        "<|new_word|>", "\n<|sep|>"
    ),
    "college on Chimera": OutFile(
        Path("ckpt/meta-word-eval_data_dir_chimeras.json_data_order_original_pretrained_model_Llama-2-7b-hf_emb_gen_model_type_college_n_examples_5_max_new_tokens_100/slurm.out"),
        "<nonce>", "\n"
    ),
    "llama-3 baseline on def_task": OutFile(
        Path("ckpt/meta-word-eval_data_dir_def_task.json_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_prompt__n_examples_4_max_new_tokens_100/slurm.out"),
        " wug", "\n *"
    ),
    "llama-3 finetuned on def_task": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_def_task.json_n_examples_4_max_new_tokens_100/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-3-instruct baseline on def_task": OutFile(
        Path("ckpt/meta-word-eval_data_dir_def_task.json_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_prompt__n_examples_4_max_new_tokens_100/slurm.out"),
        " wug", "\n *"
    ),
    "llama-3-instruct finetuned on def_task": OutFile(
        Path("ckpt/meta-word_pretrained_model_:scratch:ww2135:Meta-Llama-3-8B-Instruct_data_dir_word_use_data:babylm_data:babylm_10M:word_embedding_init_mean_prompt__train_params_new_word_sep_n_examples_5_train_max_length_160_batch_size_16_lr_0.001_seed_0_eval_step_1000/best/meta-word-eval_data_dir_def_task.json_n_examples_4_max_new_tokens_100/slurm.out"),
        "<|reserved_special_token_0|>", "\n<|reserved_special_token_1|>"
    ),
    "llama-2 finetuned on def_task": OutFile(
        Path("ckpt/meta-word_pretrained_model_Llama-2-7b-hf_data_dir_word_use_data:babylm_data:babylm_10M:word_n_examples_5_train_max_length_160_batch_size_16_lr_0.003_seed_0/best/meta-word-eval_data_dir_def_task.json_n_examples_4_max_new_tokens_100/slurm.out"),
        "<|new_word|>", "\n<|sep|>"
    ),
    "college on def_task": OutFile(
        Path("ckpt/meta-word-eval_data_dir_def_task.json_pretrained_model_Llama-2-7b-hf_emb_gen_model_type_college_n_examples_4_max_new_tokens_100/slurm.out"),
        "<nonce>", "\n"
    ),
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "names", nargs=2,
        help="Names for the 2 files for comparison."
    )
    argparser.add_argument(
        "--word_example_prompt_name",
        help="Use the ground-truth prefix from this name's file for word_example_prompt for in-context word learning."
    )
    argparser.add_argument(
        "--sep_target",
        help="The sep string in the target (prompt). Will replace sep_source with sep_target to construct the word_example_prompt."
    )
    argparser.add_argument(
        "--word_target",
        help="The word in the target (prompt). Will replace words with word_target to construct the word_eample_prompt and generations. If set to 'GT', use the ground-truth word form."
    )
    argparser.add_argument(
        "--mode", choices=list(prompt_qa_format_of_mode.keys()),
        default="example",
        help="Generation mode. Either example or definition."
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

    if args.word_example_prompt_name is not None:
        if args.word_example_prompt_name not in args.names:
            args.names.append(args.word_example_prompt_name)
    id_to_name = {i: name for i, name in enumerate(args.names)}
    files = [named_files[name].path.open() for name in args.names]

    try:
        with args.result_file.open() as result_f:
            results = json.load(result_f)
    except FileNotFoundError:
        results = []

    ordered_examples = enumerate(zip(*map(read_generations, files)))
    if args.example_order != "original":
        ordered_examples = list(ordered_examples)
    if args.example_order == "shuffle":
        rng.shuffle(ordered_examples)  # type: ignore

    try:
        for n_example, (i_example, examples) in enumerate(ordered_examples):
            examples = {name: example for name, example in zip(args.names, examples)}
            print(f"[{n_example}] Example #{i_example}:")
            field_unique_values = {}
            for field in ["ground-truth word"]:
                values = {name: example[field] for name, example in examples.items()}
                unique_values = set(values.values())
                assert len(unique_values) == 1, f"{field} differs: {values}"
                field_unique_values[field] = unique_values
            gt_word = list(field_unique_values['ground-truth word'])[0]
            for name, example in examples.items():
                word = named_files[name].word
                assert word in example["ground-truth prefix"], f"Cannot find word '{word}' in file {name}"

            if args.word_target is not None:
                if args.word_target == "GT":
                    word = gt_word
                else:
                    word = args.word_target
            if args.word_example_prompt_name is None:
                if args.word_target is None:
                    raise ValueError
                word_example_prompt = ""
            else:
                prefix = examples[args.word_example_prompt_name]["ground-truth prefix"]
                next_example_start_index = prefix.rfind(named_files[args.word_example_prompt_name].sep)
                assert next_example_start_index >= 0, "Cannot find the start of the next example (sep) in the prefix:\n{prefix}"
                word_example_prompt = prefix[ : next_example_start_index + 1]
                if args.word_target is None:
                    word = named_files[args.word_example_prompt_name].word
                else:
                    word_example_prompt = word_example_prompt.replace(
                        named_files[args.word_example_prompt_name].word,
                        word
                    )
                if args.sep_target is not None:
                    word_example_prompt = word_example_prompt.replace(
                        named_files[args.word_example_prompt_name].sep,
                        args.sep_target
                    )

            while len(results) < i_example + 1:
                results.append({})
            example_results = results[i_example]

            for field in used_output_fields:
                if field not in example_results:
                    example_results[field] = []
                field_results = example_results[field]
                values = [example[field] for example in islice(examples.values(), 2)]
                for n_gen, gens in islice(enumerate(zip(*values)), 1):
                    gens = list(gens)
                    while len(field_results) < n_gen + 1:
                        field_results.append({})
                    gens = [gen.replace(named_files[name].word, word) for name, gen in zip(args.names, gens)]
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

            if args.word_target != "GT":
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