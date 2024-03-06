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
from transformers import AutoModelForCausalLM, set_seed
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS
from data_loading import load_dataset, sample_examples
from evaluation_cls import construct_cls_example, cls_collate_fn, evaluate_cls
from main import device, tokenizer_cache, get_tokenizer


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
        "--n_examples", type=int, default=4,
    )
    argparser.add_argument(
        "--eval_n_classes", type=int, nargs="*", default=[],
        help="Number of classes for evaluation classification task."
    )
    argparser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed."
    )
    args = argparser.parse_args()

    set_seed(args.seed)
    dataset = datasets.DatasetDict({
        split: load_dataset(Path(args.data_dir, f"{split}.json"))
        for split in ["train", "validation", "test"]
    })

    tokenizer = tokenizer_cache(Path(args.data_dir, "tokenizer"))(get_tokenizer)((
        example["sentence"]
        for examples in dataset["train"]["examples"]
        for example in examples
    ))

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model).to(device)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)

    val_dataset = sample_examples(dataset["validation"], args.n_examples, np.random.default_rng(args.seed))
    val_cls_dataset = val_dataset.map(construct_cls_example)
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

    for i, item in enumerate(val_cls_dataset):
        print(f"Example #{i}:")
        print(f"ground-truth word: {item['word']}") # type: ignore
        prefix_input = tokenizer(item["prefix"], return_tensors='pt').to(device) # type: ignore
        print("prefix:", tokenizer.decode(prefix_input.input_ids[0], skip_special_tokens=False))
        print("sample outputs:")
        sample_outputs = model.generate(
            **prefix_input,
            max_new_tokens=20,
            do_sample=True,
            top_k=0,
            top_p=0.9,
            num_return_sequences=5,
        )
        for j, output in enumerate(sample_outputs):
            print(f"cont. {j}:", tokenizer.decode(output[len(prefix_input.input_ids[0]):], skip_special_tokens=True))
        print("beam search outputs:")
        beam_outputs = model.generate(
            **prefix_input,
            max_new_tokens=20,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=5,
        )
        for j, output in enumerate(beam_outputs):
            print(f"cont. {j}:", tokenizer.decode(output[len(prefix_input.input_ids[0]):], skip_special_tokens=True))
        input()