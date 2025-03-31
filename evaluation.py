from typing import Any, Iterable, TypeVar, Optional, Union
from collections.abc import Sequence, Mapping, Sized, Callable
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import re
from itertools import islice, chain, combinations
from functools import partial
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, set_seed
from data_loading import load_meta_datasets, read_meta_episodes, read_and_preprocess_examples, load_tokenizer, is_data_tokenizer, set_and_get_format, sample_examples,  is_definition_dataset, is_single_token_in_vocab
from in_context_format import InContextFormat, add_format_arguments
from evaluation_cls import cls_collate_fn, evaluate_cls, evaluate_cls_with_fixed_words, get_embs_
from emb_gen import EmbeddingGenerator, EmbGener
from utils import frac_repr, merge_input_output_dict, set_token_embeddings


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


def evaluate_gen(
        dataset: Iterable,
        tokenizer,
        model: transformers.PreTrainedModel,
        model_type: str,
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
        skip_eos_token: bool = False,
        skip_stop_string: Optional[str] = None,
        interactive: bool = False,
        emb_gener: Optional[EmbGener] = None,
):
    with torch.no_grad():
        for i, item in enumerate(dataset):
            print(f"Example #{i}:")
            print(f"ground-truth word:", item["word"]) # type: ignore
            if "study" in item:
                print(f"study examples:")
                for study_ in item["study"]:
                    print(study_)
                if emb_gener is not None:
                    study_input = emb_gener.mlm_tokenizer(
                        item["study"],
                        truncation=True,
                        padding='longest',
                        return_tensors='pt',
                    ).to(device)
                    embs = get_embs_(study_input, emb_gener)
                    set_token_embeddings(model, [emb_gener.token_id], embs)
            print(f"ground-truth prefix:", item["prefix"]) # type: ignore
            prefix_input = tokenizer(item["prefix"], return_tensors='pt').to(device) # type: ignore
            if print_decoded_prefix:
                print(f"     decoded prefix:", tokenizer.decode(prefix_input.input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            print(f"ground-truth suffix:", item["suffix"]) # type: ignore
            full_str = item["prefix"] + item["suffix"] # type: ignore
            full_input = tokenizer(full_str, return_tensors='pt').to(device)

            def _print_outputs(
                    outputs,
                    skip_eos_token: bool = skip_eos_token,
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
                    if model_type == "causal":
                        start_idx = len(prefix_input.input_ids[0])
                    elif model_type == "seq2seq":
                        start_idx = 0
                        # strip starting pad tokens (for T5 models)
                        while start_idx < sequence_length and sequence[start_idx].item() == tokenizer.pad_token_id:
                            start_idx += 1
                    else:
                        raise ValueError(f"Unknown language model type {model_type}")
                    output_string = tokenizer.decode(
                        sequence[start_idx:sequence_length],
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
                assert model_type == "causal", f"Only support ground-truth full sequence for causal LMs, but have a {model_type} LM"
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

            if top_p:
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

            if num_beams > 1:
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


def evaluate_classification(
        dataset: datasets.Dataset,
        prefix: str,
        tokenizer,
        model: transformers.PreTrainedModel,
        raw_loss_fct: torch.nn.Module,
        eval_n_classes: list[int],
        emb_gener: Optional[EmbGener] = None,
):
    # classification
    for n_classes in eval_n_classes:
        dataloader = DataLoader(
            dataset, # type: ignore
            batch_size=n_classes,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(
                cls_collate_fn,
                tokenizer,
                study_tokenizer=(None if emb_gener is None else emb_gener.mlm_tokenizer)
            ),
        )
        value_name = f"{prefix}cls_{n_classes}_acc"
        cls_acc = evaluate_cls(
            model,
            dataloader,
            raw_loss_fct,
            emb_gener=emb_gener,
        )
        print(f"{value_name}={frac_repr(*cls_acc, prec=3)}")


def evaluate_generation(
        dataset: Iterable,
        tokenizer,
        in_context_format: InContextFormat,
        model: transformers.PreTrainedModel,
        model_type: str,
        args,
        skip_eos_token: bool = False,
        emb_gener: Optional[EmbGener] = None,
):
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
        evaluate_gen(
            dataset,
            tokenizer,
            model,
            model_type,
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
            skip_eos_token=skip_eos_token,
            skip_stop_string=stop_string,
            interactive=args.interactive,
            emb_gener=emb_gener,
        )
    except EOFError:
        pass


def evaluate_syntactic_category_classification(
        tokenizer,
        in_context_format: InContextFormat,
        model: transformers.PreTrainedModel,
        raw_loss_fct: Callable,
        data_kwargs: Mapping,
        data_dir: Path = Path("category-abstraction", "data"),
        file_prefix: str = "mnli",
        longer: bool = False,
        split: str = "test",
        word: str = "[MASK]",
        batch_size: int = 8,
        drop_last: bool = False,
        emb_gener: Optional[EmbGener] = None,
):
    """Evaluate model on Kim and Smolensky (2021)'s dataset.
    """
    syn_ctgs = ["n", "v", "adj", "adv"]
    ctgs = ["ctg0", "ctg1"]
    data_kinds = {
        'diff': '_different_{ctg}_{split}.txt'.format,
        'ident': '_identical_{ctg}_{split}.txt'.format,
    }
    if longer:
        data_kinds['diff_longer'] = '_different_{ctg}_{split}_longer.txt'.format
    word_pattern = re.compile(re.escape(word))

    results = {}
    for syn_ctg_pair in combinations(syn_ctgs, 2):
        sorted_syn_ctg_pair = tuple(sorted(syn_ctg_pair))
        data_dir_ = data_dir / ("".join(sorted_syn_ctg_pair)+"_f")

        with (data_dir_ / (file_prefix + "_finetune.txt")).open() as f:
            examples = read_and_preprocess_examples(word_pattern, file=f, **data_kwargs)
            word_items = [{"word": word, "examples": [example]} for example in examples]

        for data_kind, file_suffix_fn in data_kinds.items():
            for label, ctg in enumerate(ctgs):
                name = f"{split}_{'_'.join(syn_ctg_pair)}_{ctg}_{data_kind}"
                with (data_dir_ / (file_prefix + file_suffix_fn(ctg=ctg, split=split))).open() as f:
                    examples = map(
                        lambda example: [example],
                        read_and_preprocess_examples(word_pattern, file=f, **data_kwargs)
                    )
                    acc = evaluate_cls_with_fixed_words(
                        word_items,
                        examples,
                        tokenizer,
                        in_context_format,
                        model,
                        raw_loss_fct,
                        label,
                        batch_size,
                        drop_last=drop_last,
                        emb_gener=emb_gener,
                    )
                print(f"{name}_acc={frac_repr(*acc, prec=1)}")
                results[name] = acc

    return results


def load_meta_data_sources(
        data_sources: Iterable,
        splits: list[str],
        kwargs: Mapping,
        args,
):
    for data_source in data_sources:
        if data_source == "syntactic":
            yield "syntactic_", None
            continue
        if data_source == "input":
            yield "", read_meta_episodes(file=None, **kwargs)
            continue
        data_path = Path(data_source)
        if data_path.is_file() and data_path.suffix != ".json":
            with data_path.open() as file:
                # Note: from_generator will always cache the dataset
                # clean up the cache files if updated
                dataset = datasets.Dataset.from_generator(
                    read_meta_episodes,
                    gen_kwargs={"file": file, **kwargs},
                )
            yield data_path.stem+"_", dataset
            continue
        for split_name, data in load_meta_datasets(
                [data_path], splits=splits, kwargs=kwargs):
            if split_name == "validation":
                split_name = "val"
            prefix = f'{split_name}_' if split_name is not None else ''
            print(f'{prefix}n_words: {len(data)}')
            dataset = sample_examples(
                data,
                (
                    [(None, args.n_examples - 1), ("definition", 1)]
                    if is_definition_dataset(data) else
                    args.n_examples
                ),
                max_sample_times = args.max_sample_times,
                rng = None if args.data_order == "original" else np.random.default_rng(args.seed),
            )
            print(f'{prefix}n_episodes: {len(dataset)}')
            yield prefix, dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir", nargs="+", default=["input"],
        help='Sources of data. '
             'Each can be a path to the dataset directory or file, '
             'or "input" (terminal input), '
             'or "syntactic" (Najoung & Smolensky, 2021; '
             'path default to category-abstraction/data).',
    )
    argparser.add_argument(
        "--split", nargs="+", choices=["train", "val", "dev", "test"],
        default=["val"],
        help="Which split(s) to use."
    )
    argparser.add_argument(
        "--emb_gen_model_type", choices=["college"],
        help="If set, use the designated embedding generation model type. "
             "For college, the pretrained_model must be Llama-2-7b."
    )
    argparser.add_argument(
        "--emb_gen_model_path", type=Path,
        default=Path("college_pretrained_model/checkpoint_7_28000"),
    )
    argparser.add_argument(
        "--emb_gen_mlm",
        default="roberta-large",
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
        "--cls_last_n", type=int, default=1,
    )
    argparser.add_argument(
        "--gen_last_n", type=int, default=1,
    )
    argparser.add_argument(
        "--max_sample_times", type=int, default=1,
    )
    argparser.add_argument(
        "--append_to_prefix", default="",
        help="Append this string to the prefix. Can have a variable new_word to be substituted by the new word to be learned."
    )
    argparser.add_argument(
        "--append_to_prefix_for_gen", default=None,
        help="During generation, append this string to the prefix after previously appended string. If not set, use append_to_prefix. Same as append_to_prefix, can have a variable new_word."
    )
    argparser.add_argument(
        "--eval_n_classes", type=int, nargs="*", default=[],
        help="Number of classes for evaluation classification task."
    )
    argparser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for syntactic category classification."
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

    auto_model_cls_mapping = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }
    for model_type, auto_model_cls in auto_model_cls_mapping.items():
        try:
            model = auto_model_cls.from_pretrained(
                args.pretrained_model,
                revision=args.revision,
                device_map=device,
            )
        except ValueError as exception:
            pass
        else:
            break
    else:
        raise exception  # type: ignore

    if args.emb_gen_model_type is None:
        emb_gener = None
        in_context_format = set_and_get_format(
            tokenizer,
            args,
            sep_prefix = "" if model_type == "seq2seq" else "\n",
            set_pad_to_eos = model_type != "seq2seq",
        )

    elif args.emb_gen_model_type == "college":
        assert model_type == "causal"  # model should be Llama-2-7b
        assert not args.no_new_token, "Must have a new token for generated embedding"
        assert is_single_token_in_vocab(tokenizer, args.new_word), "new_word must be a single token for generated embedding"
        new_word_token_id: int = tokenizer(args.new_word, add_special_tokens=False)['input_ids'][0]  # type: ignore

        emb_gen_mlm_tokenizer = AutoTokenizer.from_pretrained(
            args.emb_gen_model_path/"tokenizerMLM",
        )
        emb_gen_mlm = AutoModelForMaskedLM.from_pretrained(
            args.emb_gen_mlm,
            device_map=device,
        )
        emb_gen_model = EmbeddingGenerator(
            emb_gen_mlm.config.hidden_size,
            emb_gen_mlm.config.num_attention_heads,
            model.config.hidden_size,
            num_layers=1,
        ).to(device)
        emb_gen_model.load_state_dict(
            torch.load(
                args.emb_gen_model_path/"pytorch_model.bin",
                map_location=device,
            )
        )
        emb_gen_mlm.eval()
        emb_gen_model.eval()
        emb_gener = EmbGener(
            emb_gen_mlm_tokenizer,
            emb_gen_mlm,
            emb_gen_model,
            new_word_token_id,
        )

        t = args.new_word
        sep = "\n" + args.sep  # TODO: do not end suffix with sep and use the model eos as the eos
        tokenizer.pad_token = tokenizer.eos_token
        in_context_format = InContextFormat(
            t = t,
            sep = sep,
            prompt = args.prompt,
            t_study = emb_gen_mlm_tokenizer.mask_token,
            no_study_in_prefix = True,  # can try False later
        )

    else:
        raise ValueError(f"Unknown emb_gen_model_type {args.emb_gen_model_type}")

    if n_added_tokens:
        print("Warning: may use untrained token embeddings")
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id) # type: ignore

    meta_dataset_kwargs = dict(
        clean_up_tokenization_spaces = not is_data_tokenizer(tokenizer),
        prepend = args.prepend,
    )
    mapped_splits = [
        "validation" if split == "val" else split
        for split in args.split
    ]
    for prefix, dataset in load_meta_data_sources(
            args.data_dir, mapped_splits, meta_dataset_kwargs, args):
        # syntactic category classification
        if dataset is None:
            assert prefix == "syntactic_"
            for split in args.split:
                evaluate_syntactic_category_classification(
                    tokenizer,
                    in_context_format,
                    model,
                    raw_loss_fct,
                    meta_dataset_kwargs,
                    split=split,
                    batch_size=args.batch_size,
                    emb_gener=emb_gener,
                )
            continue
        # classification
        if args.eval_n_classes:
            assert isinstance(dataset, datasets.Dataset), "Cannot evaluate classification on stream input"
            cls_dataset = dataset.map(
                partial(
                    in_context_format.construct_meta_cls_example,
                    last_n=args.cls_last_n,
                    append_to_prefix=args.append_to_prefix,
                )
            )
            evaluate_classification(
                cls_dataset,
                prefix,
                tokenizer,
                model,
                raw_loss_fct,
                args.eval_n_classes,
                emb_gener=emb_gener,
            )
        else:
            cls_dataset = None
        # generation
        if cls_dataset is not None and args.gen_last_n == args.cls_last_n and args.append_to_prefix_for_gen is None:
            gen_dataset = cls_dataset
        else:
            fn = partial(
                in_context_format.construct_meta_cls_example,
                last_n=args.gen_last_n,
                append_to_prefix = (
                    args.append_to_prefix
                    if args.append_to_prefix_for_gen is None else
                    args.append_to_prefix_for_gen
                ),
            )
            if isinstance(dataset, datasets.Dataset):
                gen_dataset = dataset.map(fn)
            else:
                gen_dataset = map(merge_input_output_dict(fn), dataset)
        evaluate_generation(
            gen_dataset,
            tokenizer,
            in_context_format,
            model,
            model_type,
            args,
            skip_eos_token = model_type == "seq2seq",
            emb_gener=emb_gener,
        )