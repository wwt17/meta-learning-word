from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import json
import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import datasets
import tokenizers
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast, GPT2TokenizerFast, AutoModelForCausalLM, set_seed
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS
from data_loading import load_meta_dataset, sample_examples
from evaluation_cls import construct_meta_cls_example, cls_collate_fn, evaluate_cls
from main import device


def print_top_k_preds(logits, k: int, tokenizer):
    print(f"top-{k} predictions:")
    for logits_step in logits:
        probs = logits_step.softmax(dim=-1)
        topk_out = probs.topk(k, dim=-1)
        for prob, idx in zip(*topk_out):
            print(f"{prob.item():4.0%} {tokenizer.convert_ids_to_tokens(idx.item()):<10s}", end="")
        print()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir", type=Path,
        default=Path("word_use_data", "childes", "word"),
        help="Path to the dataset."
    )
    argparser.add_argument(
        "--pretrained_model",
        help="Pretrained model name or path to resume from."
    )
    argparser.add_argument(
        "--tokenizer",
    )
    argparser.add_argument(
        "--no_new_token", action="store_true",
        help="Do not replace the word with the new token."
    )
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
        "--interactive", action="store_true",
    )
    args = argparser.parse_args()

    set_seed(args.seed)
    dataset = datasets.DatasetDict({
        split: load_meta_dataset(Path(args.data_dir, f"meta.{split}.json"))
        for split in ["train", "validation", "test"]
    })

    if args.tokenizer is None:
        args.tokenizer = Path(args.data_dir, "tokenizer")
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer) # type: ignore

    fmt_kwargs = dict(t = None if args.no_new_token else NEW_TOKEN)
    if isinstance(tokenizer, GPT2TokenizerFast):
        fmt_kwargs.update(dict(sep="\n", space="", prompt=""))
        sep_token_id = 198
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # must not provide token_type_ids to the model
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        fmt_kwargs.update(dict(sep=SEP_TOKEN, space=" ", prompt=""))
        sep_token_id = tokenizer.convert_tokens_to_ids(fmt_kwargs["sep"]) # type: ignore

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model).to(device)
    model.config.bos_token_id = sep_token_id #tokenizer.bos_token_id
    model.config.eos_token_id = sep_token_id #tokenizer.eos_token_id
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id) # type: ignore

    val_dataset = sample_examples(
        dataset["validation"],
        args.n_examples,
        max_sample_times=args.max_sample_times,
        rng=np.random.default_rng(args.seed),
    )
    val_cls_dataset = val_dataset.map(partial(construct_meta_cls_example, **fmt_kwargs))
    val_cls_dataloaders = {
        n_classes: DataLoader(
            val_cls_dataset, # type: ignore
            batch_size=n_classes,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(cls_collate_fn, tokenizer),
        )
        for n_classes in args.eval_n_classes
    }
    model.eval()
    for n_classes, val_cls_dataloader in val_cls_dataloaders.items():
        value_name = f"val_cls_{n_classes}_acc"
        val_cls_acc = evaluate_cls(model, val_cls_dataloader, raw_loss_fct)
        print(f"{value_name}={val_cls_acc:.3%}")

    try:
        for i, item in enumerate(val_cls_dataset):
            print(f"Example #{i}:")
            print(f"ground-truth word:", item["word"]) # type: ignore
            print(f"ground-truth prefix:", item["prefix"]) # type: ignore
            prefix_input = tokenizer(item["prefix"], return_tensors='pt').to(device) # type: ignore
            if args.print_decoded_prefix:
                print(f"     decoded prefix:", tokenizer.decode(prefix_input.input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            print(f"ground-truth suffix:", item["suffix"]) # type: ignore
            full_str = item["prefix"] + item["suffix"] # type: ignore
            full_input = tokenizer(full_str, return_tensors='pt').to(device)

            generation_config = transformers.generation.GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            def _print_outputs(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    print_top_k_pred=args.print_top_k_pred,
                    **kwargs
            ):
                assert isinstance(outputs, transformers.utils.ModelOutput), "must set return_dict_in_generate=True"
                for j, sequence in enumerate(outputs.sequences):  # type: ignore
                    print(
                        f"cont. {j}:",
                        tokenizer.decode(
                            sequence[len(prefix_input.input_ids[0]):],
                            skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                            **kwargs
                        )
                    )
                    if print_top_k_pred:
                        assert outputs.scores is not None, "must set output_scores=True"  # type: ignore
                        print_top_k_preds(
                            (scores_step[j] for scores_step in outputs.scores),  # type: ignore
                            print_top_k_pred,
                            tokenizer
                        )


            if args.print_gt_full:
                print("ground-truth   full:", full_str)
                gt_outputs = model(
                    **full_input,
                    return_dict=True,
                )
                if args.print_top_k_pred:
                    print_top_k_preds(
                        gt_outputs.logits[0],
                        args.print_top_k_pred,
                        tokenizer
                    )

            print("greedy outputs:")
            greedy_outputs = model.generate(
                **prefix_input,
                generation_config=generation_config,
            )
            _print_outputs(greedy_outputs)

            print(f"sample with top-p={args.top_p:.2f} outputs:")
            sample_outputs = model.generate(
                **prefix_input,
                generation_config=generation_config,
                do_sample=True,
                top_k=0,
                top_p=args.top_p,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
            )
            _print_outputs(sample_outputs)

            print("beam search outputs:")
            beam_outputs = model.generate(
                **prefix_input,
                generation_config=generation_config,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=args.early_stopping,
                num_return_sequences=args.num_return_sequences,
            )
            _print_outputs(beam_outputs)

            if args.interactive:
                input()
            else:
                print()

    except EOFError:
        pass