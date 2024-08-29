from typing import Any, Iterable, TypeVar, Optional, Union
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import datasets
import transformers
from transformers import AutoModelForCausalLM, set_seed
from data_loading import load_meta_datasets, load_tokenizer, is_data_tokenizer, set_and_get_format, sample_examples
from in_context_format import InContextFormat, add_format_arguments
from evaluation_cls import cls_collate_fn, evaluate_cls
from utils import frac_repr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StopSubStringCriteria(transformers.StoppingCriteria):
    def __init__(self, tokenizer, stop_string: str, prefix_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.prefix_length = prefix_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        if self.prefix_length is not None:
            input_ids = input_ids[:, self.prefix_length:]  # type: ignore
        decoded_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        return torch.tensor([self.stop_string in text for text in decoded_text], dtype=torch.bool, device=input_ids.device)


def print_top_k_preds(logits, k: int, tokenizer):
    print(f"top-{k} predictions:")
    for logits_step in logits:
        probs = logits_step.softmax(dim=-1)
        topk_out = probs.topk(k, dim=-1)
        for prob, idx in zip(*topk_out):
            print(f"{prob.item():4.0%} {tokenizer.convert_ids_to_tokens(idx.item()):<10s}", end="")
        print()


def evaluate_generation(
        cls_dataset: datasets.Dataset,
        tokenizer,
        generation_config: transformers.generation.GenerationConfig,
        stopping_criteria = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: Union[bool, str] = False,
        num_return_sequences: int = 1,
        device = device,
        print_decoded_prefix: bool = False,
        print_gt_full: bool = False,
        print_top_k_pred: int = 0,
        skip_stop_string: Optional[str] = None,
        interactive: bool = False,
):
    for i, item in enumerate(cls_dataset):
        print(f"Example #{i}:")
        print(f"ground-truth word:", item["word"]) # type: ignore
        print(f"ground-truth prefix:", item["prefix"]) # type: ignore
        prefix_input = tokenizer(item["prefix"], return_tensors='pt').to(device) # type: ignore
        if print_decoded_prefix:
            print(f"     decoded prefix:", tokenizer.decode(prefix_input.input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        print(f"ground-truth suffix:", item["suffix"]) # type: ignore
        full_str = item["prefix"] + item["suffix"] # type: ignore
        full_input = tokenizer(full_str, return_tensors='pt').to(device)

        def _print_outputs(
                outputs,
                skip_eos_token: bool = False,
                skip_stop_string: Optional[str] = skip_stop_string,
                skip_special_tokens: bool = False,  # TODO: skip other special tokens but retain the new word
                clean_up_tokenization_spaces: bool = False,
                print_top_k_pred: int = print_top_k_pred,
                **kwargs
        ):
            assert isinstance(outputs, transformers.utils.ModelOutput), "must set return_dict_in_generate=True"
            for j, sequence in enumerate(outputs.sequences):  # type: ignore
                sequence_length = len(sequence)
                while sequence_length > 0 and sequence[sequence_length - 1].item() == tokenizer.pad_token_id:
                    sequence_length -= 1
                if skip_eos_token and sequence_length > 0 and sequence[sequence_length - 1].item() == generation_config.eos_token_id:
                    sequence_length -= 1
                output_string = tokenizer.decode(
                    sequence[len(prefix_input.input_ids[0]):sequence_length],
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    **kwargs
                )
                if skip_stop_string:  # type: ignore
                    stop_string_index = output_string.rfind(skip_stop_string)  # type: ignore
                    if stop_string_index >= 0:
                        output_string = output_string[:stop_string_index]
                print(f"cont. {j}:", output_string)
                if print_top_k_pred:
                    assert outputs.scores is not None, "must set output_scores=True"  # type: ignore
                    print_top_k_preds(
                        (scores_step[j] for scores_step in outputs.scores),  # type: ignore
                        print_top_k_pred,
                        tokenizer
                    )


        if print_gt_full:
            print("ground-truth   full:", full_str)
            gt_outputs = model(
                **full_input,
                return_dict=True,
            )
            if print_top_k_pred:
                print_top_k_preds(
                    gt_outputs.logits[0],
                    print_top_k_pred,
                    tokenizer
                )

        print("greedy outputs:")
        greedy_outputs = model.generate(
            **prefix_input,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )
        _print_outputs(greedy_outputs)

        print(f"sample with top-p={top_p:.2f} outputs:")
        sample_outputs = model.generate(
            **prefix_input,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            top_k=0,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        _print_outputs(sample_outputs)

        print("beam search outputs:")
        beam_outputs = model.generate(
            **prefix_input,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences,
        )
        _print_outputs(beam_outputs)

        if interactive:
            input()
        else:
            print()


def evaluate(
        data: datasets.Dataset,
        split_name: Optional[str],
        tokenizer,
        in_context_format: InContextFormat,
        model: torch.nn.Module,
        raw_loss_fct: torch.nn.Module,
        args,
):
    prefix = f'{split_name}_' if split_name is not None else ''
    print(f'{prefix}n_words: {len(data)}')
    dataset = sample_examples(
        data,
        args.n_examples,
        max_sample_times = args.max_sample_times,
        rng = None if args.data_order == "original" else np.random.default_rng(args.seed),
    )
    print(f'{prefix}n_episodes: {len(dataset)}')
    cls_dataset = dataset.map(in_context_format.construct_meta_cls_example)
    model.eval()

    # classification
    for n_classes in args.eval_n_classes:
        cls_dataloader = DataLoader(
            cls_dataset, # type: ignore
            batch_size=n_classes,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(cls_collate_fn, tokenizer),
        )
        value_name = f"{prefix}cls_{n_classes}_acc"
        cls_acc = evaluate_cls(model, cls_dataloader, raw_loss_fct)
        print(f"{value_name}={frac_repr(*cls_acc, prec=3)}")

    # generation
    stop_string: str = in_context_format.sep  # type: ignore
    eos_token_id: int = tokenizer(stop_string)['input_ids'][-1]
    generation_config = transformers.generation.GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        #stop_strings=stop_string,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    try:
        evaluate_generation(
            cls_dataset,
            tokenizer,
            generation_config,
            #stopping_criteria = transformers.StoppingCriteriaList([StopSubStringCriteria(tokenizer, stop_string, len(prefix_input.input_ids[0]))]),
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            early_stopping=args.early_stopping,
            num_return_sequences=args.num_return_sequences,
            print_decoded_prefix=args.print_decoded_prefix,
            print_gt_full=args.print_gt_full,
            print_top_k_pred=args.print_top_k_pred,
            skip_stop_string=stop_string,
            interactive=args.interactive,
        )
    except EOFError:
        pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir", type=Path, nargs="+",
        help="Path(s) to the dataset directory or file.",
        required=True,
    )
    argparser.add_argument(
        "--split", nargs="+", choices=["train", "val", "test"],
        default=["val"],
        help="Which split(s) to use."
    )
    argparser.add_argument(
        "--pretrained_model",
        help="Pretrained model name or path to resume from."
    )
    argparser.add_argument(
        "--revision",
    )
    argparser.add_argument(
        "--tokenizer",
    )
    group = argparser.add_argument_group("In-context format")
    add_format_arguments(group)
    argparser.add_argument(
        "--n_examples", type=int, default=4,
    )
    argparser.add_argument(
        "--max_sample_times", type=int, default=1,
    )
    argparser.add_argument(
        "--eval_n_classes", type=int, nargs="*", default=[],
        help="Number of classes for evaluation classification task."
    )
    argparser.add_argument(
        "--max_new_tokens", type=int, default=30,
    )
    argparser.add_argument(
        "--top_p", type=float, default=0.92,
    )
    argparser.add_argument(
        "--temperature", type=float, default=1.0,
    )
    argparser.add_argument(
        "--num_beams", type=int, default=5,
    )
    argparser.add_argument(
        "--length_penalty", type=float, default=4.0,
    )
    argparser.add_argument(
        "--no_repeat_ngram_size", type=int, default=4,
    )
    argparser.add_argument(
        "--early_stopping", action="store_true",
    )
    argparser.add_argument(
        "--num_return_sequences", type=int, default=5,
    )
    argparser.add_argument(
        "--print_decoded_prefix", action="store_true",
    )
    argparser.add_argument(
        "--print_gt_full", action="store_true",
    )
    argparser.add_argument(
        "--print_top_k_pred", type=int, default=0,
    )
    argparser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed."
    )
    argparser.add_argument(
        "--data_order", choices=["original"],
        help="original: episodes will consist of first examples of a word"
             " in their original order."
    )
    argparser.add_argument(
        "--interactive", action="store_true",
    )
    args = argparser.parse_args()

    set_seed(args.seed)

    tokenizer, n_added_tokens = load_tokenizer(
        args.tokenizer if args.tokenizer is not None else args.pretrained_model,
        revision=args.revision,
        new_tokens=args.add_tokens,
    )

    in_context_format = set_and_get_format(tokenizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        device_map=device,
    )
    if n_added_tokens:
        print("Warning: may use untrained token embeddings")
        model.resize_token_embeddings(len(tokenizer))
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id) # type: ignore

    meta_dataset_kwargs = dict(
        clean_up_tokenization_spaces = not is_data_tokenizer(tokenizer),
        prepend = args.prepend,
    )
    mapped_splits = [
        "validation" if split == "val" else split
        for split in args.split
    ]
    for split_name, data in load_meta_datasets(args.data_dir, mapped_splits, meta_dataset_kwargs):
        if split_name == "validation":
            split_name = "val"
        evaluate(
            data,
            split_name,
            tokenizer,
            in_context_format,
            model,
            raw_loss_fct,
            args,
        )