from typing import Optional
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import islice
import numpy as np
import evaluate
import spacy
from read_generations import input_fields, output_fields, fields, read_generations, extract_definition_from_generation, extract_definition_from_reference_generation


used_output_fields = output_fields[:-1]


def get_mean_result(result: dict):
    return {
        key: np.mean(value).item() if isinstance(value, list) else value
        for key, value in result.items()
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "files", type=Path, nargs="+",
        help="The files of evaluation outputs."
    )
    argparser.add_argument(
        "--extract_definition_from_reference_generation", action="store_true",
    )
    argparser.add_argument(
        "--extract_definition_from_generation", action="store_true",
    )
    argparser.add_argument(
        "--word", nargs="+",
        default=[None],
        help="Strings representing the new word to be learned,"
             " in the same order of their corresponding files."
             " If only one is given, use it for all files."
             " If not given, keep words unreplaced."
    )
    argparser.add_argument(
        "--replace_word", nargs="+",
        default=[None],
        help="Strings used to replace the new word with,"
             " in the same order of the files or a same replace word for all files."
             " If only one is given, use it for all files."
             " If not given, keep words unreplaced."
    )
    argparser.add_argument(
        "--sep", nargs="+",
        default=[None],
        help="Strings representing the seperators between examples,"
             " in the same order of their corresponding files."
             " If only one is given, use it for all files."
             " If not given, find it out from the file."
    )
    argparser.add_argument(
        "--n_gens", type=int,
        help="Use the first n_gens generations."
             " If not given, use all."
    )
    argparser.add_argument(
        "--n_first", type=int,
        help="Use only the first n_first examples."
    )
    args = argparser.parse_args()

    for argname in ["word", "replace_word", "sep"]:
        if len(getattr(args, argname)) == 1:
            setattr(args, argname, getattr(args, argname) * len(args.files))
        assert len(getattr(args, argname)) == len(args.files), f"Must provide same number of elements in {argname} ({len(getattr(args, argname))}) as files ({len(args.files)}), or only one"

    rouge = evaluate.load("rouge", keep_in_memory=True)
    bertscore = evaluate.load("bertscore", keep_in_memory=True)
    nlp = spacy.load('en_core_web_sm')

    for file, word, replace_word, _sep in zip(args.files, args.word, args.replace_word, args.sep):
        if replace_word is not None:
            assert word is None, "Must provide word for replacement"
        print(f"Processing {file}:")
        sep = _sep

        with file.open() as f:
            examples = read_generations(f)
            if args.n_first:
                examples = islice(examples, args.n_first)

            records = []

            for example in examples:
                gt_suffix = example["ground-truth suffix"]
                if _sep is None:  # infer the sep
                    try:
                        newline_index = gt_suffix.rindex("\n")
                    except ValueError:
                        raise Exception(f"Cannot infer the separator from: {gt_suffix}")
                    else:
                        inferred_sep = gt_suffix[newline_index:]
                    if sep is None:
                        print(f"Inferred separator: {repr(inferred_sep)}")
                        sep = inferred_sep
                    else:
                        assert inferred_sep == sep, f"Find different separator ({inferred_sep}) vs. the previous one ({sep})"
                try:
                    sep_index = gt_suffix.rindex(sep)
                except ValueError:
                    raise Exception(f"Cannot find separator in {gt_suffix}")
                ref = gt_suffix[:sep_index]
                if args.extract_definition_from_reference_generation:
                    ref = extract_definition_from_reference_generation(ref, nlp)
                ref = ref.strip()

                gens = {}
                for field in used_output_fields:
                    value = example[field][:args.n_gens]
                    if args.extract_definition_from_generation:
                        value = list(map(
                            extract_definition_from_generation,
                            value
                        ))
                    if replace_word is not None:
                        value = [g.replace(word, replace_word) for g in value]
                    value = [g.strip() for g in value]
                    gens[field] = value

                records.append((gens, ref))

        all_generations, references = zip(*records)

        # evaluate
        results = defaultdict(list)
        for field in used_output_fields:
            print(f"Evaluating {field}:")
            for i, generations in enumerate(zip(*(gens[field] for gens in all_generations))):
                rouge_result: Optional[dict] = rouge.compute(predictions=generations, references=references, use_stemmer=True)
                assert rouge_result is not None, "rouge is not run"
                bertscore_result: Optional[dict] = bertscore.compute(predictions=generations, references=references, lang='en')
                assert bertscore_result is not None, "bertscore is not run"
                mean_bertscore_result = get_mean_result(bertscore_result)
                results[field].append({"rouge": rouge_result, "bertscore": mean_bertscore_result})

        # save results
        evaluate.save(file.with_name(file.name + ".eval_result.json"), **results)