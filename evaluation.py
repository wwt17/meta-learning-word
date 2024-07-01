from typing import Any, Iterable, TypeVar, Optional
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import sys
import re
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
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, set_seed
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS
from data_loading import load_meta_dataset, sample_examples
from in_context_format import InContextFormat
from evaluation_cls import cls_collate_fn, evaluate_cls


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
        "--revision",
    )
    argparser.add_argument(
        "--tokenizer",
    )
    argparser.add_argument(
        "--new_word", default=NEW_TOKEN,
        help="Replace word with this."
    )
    argparser.add_argument(
        "--no_new_token", action="store_true",
        help="Do not replace the word with the new token."
    )
    argparser.add_argument(
        "--prompt", default="",
        help="Prompt before examples."
    )
    argparser.add_argument(
        "--sep", default="",
        help=r'Use "\n"+sep as the separator for pretrained models.'
    )
    argparser.add_argument(
        "--prepend", default=" ",
        help="Prepend this string to each example."
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

    tokenizer = None
    if args.tokenizer is None:
        args.tokenizer = args.pretrained_model
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                revision=args.revision,
            )
        except OSError:
            m = re.search(r"_data_dir_(.+)_config_", args.pretrained_model)
            if m is None:
                raise Exception(f"Cannot extract pretraining data_dir (used to find tokenizer) from pretrained_model {args.pretrained_model}")
            pretraining_data_dir = Path(*m[1].split(":"))
            print(f"pretraining data_dir extracted from pretrained_model: {pretraining_data_dir}", file=sys.stderr)
            args.tokenizer = Path(pretraining_data_dir, "tokenizer")
    print(f"tokenizer: {args.tokenizer}", file=sys.stderr)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            revision=args.revision,
        )

    if tokenizer.eos_token != SEP_TOKEN:  # ad-hoc way to tell a pretrained tokenizer
        sep = "\n" + args.sep
        prepend = args.prepend
        tokenizer.pad_token = tokenizer.eos_token
        clean_up_tokenization_spaces = True
        #generation_config_kwargs = dict(
        #    stop_strings = sep
        #)
    else:  # my own tokenizer
        # must not provide token_type_ids to the model
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        sep = " " + SEP_TOKEN
        prepend = " "
        clean_up_tokenization_spaces = False
        #generation_config_kwargs = dict(
        #    eos_token_id = tokenizer(sep)['input_ids'][0]  # type: ignore
        #)
    in_context_format = InContextFormat(
        t = None if args.no_new_token else args.new_word,
        sep = sep,
        prepend = prepend,
        prompt = args.prompt,
    )
    stop_string: str = sep
    eos_token_id = tokenizer(stop_string)['input_ids'][-1]  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        device_map=device,
    )
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id) # type: ignore

    dataset = datasets.DatasetDict({
        split: load_meta_dataset(
            Path(args.data_dir, f"meta.{split}.json"),
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for split in ["train", "validation", "test"]
    })
    val_dataset = sample_examples(
        dataset["validation"],
        args.n_examples,
        max_sample_times=args.max_sample_times,
        rng=np.random.default_rng(args.seed),
    )
    val_cls_dataset = val_dataset.map(in_context_format.construct_meta_cls_example)
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
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                #**generation_config_kwargs
            )
            #stopping_criteria = transformers.StoppingCriteriaList([StopSubStringCriteria(tokenizer, stop_string, len(prefix_input.input_ids[0]))])

            def _print_outputs(
                    outputs,
                    skip_eos_token=False,
                    skip_stop_string=True,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                    print_top_k_pred=args.print_top_k_pred,
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
                    if skip_stop_string and stop_string is not None:  # type: ignore
                        stop_string_index = output_string.rfind(stop_string)  # type: ignore
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