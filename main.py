from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import sys
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import numpy as np
import pandas as pd
import wandb
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, default_collate
import transformers
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2Config, GPTNeoXConfig, LlamaConfig, set_seed
from utils import frac_repr, frac_tuple_to_float, to, mix_iter, initialize_new_token_embeddings, freeze_non_embedding_params, zero_grad_embedding_params
from data_loading import load_dataset, load_tokenizer, is_data_tokenizer, set_and_get_format, sample_examples, sample_lm_seq
from in_context_format import InContextFormat, add_format_arguments, format_str_attrs
from evaluation_cls import cls_collate_fn, evaluate_cls
from concat_lm_dataset import ConcatLMDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(ckpt_path, model, optimizer, scheduler):
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), ckpt_path/"optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_path/"scheduler.pt")


def evaluate_lm(model, dataloader, loss_fct) -> float:
    total_loss, total_n_tokens = 0., 0
    with torch.no_grad():
        for batch in dataloader:
            batch = to(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fct(outputs.logits[..., :-1, :].movedim(-1, 1), batch["input_ids"][..., 1:])
            total_loss += loss.item()
            total_n_tokens += batch["attention_mask"][..., 1:].sum().item()
    mean_loss = total_loss / total_n_tokens
    return mean_loss


def concat_lm_collate(batch):
    input_ids = default_collate(batch)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


model_max_length_attr = {
    GPT2Config: "n_ctx",
    GPTNeoXConfig: "max_position_embeddings",
    LlamaConfig: "max_position_embeddings",
}


def main(project="meta-learning-word", info_file=sys.stderr, **kwargs):
    wandb.init(project=project, **kwargs)

    if wandb.config.seed is not None:
        set_seed(wandb.config.seed)

    meta_dataset, lm_dataset = load_dataset(
        wandb.config.data_dir,
        lm=wandb.config.lm,
        meta_dataset_kwargs=dict(
            clean_up_tokenization_spaces = wandb.config.pretrained_model is not None,
            prepend = wandb.config.prepend,
        )
    )
    tokenizer, n_added_tokens = load_tokenizer(
        wandb.config.tokenizer if wandb.config.tokenizer is not None else (
            wandb.config.pretrained_model if wandb.config.pretrained_model is not None else
            Path(wandb.config.data_dir, "tokenizer")
        ),
        revision=wandb.config.revision,
        data_tokenizer_kwargs=dict(
            meta_dataset=meta_dataset,
            lm_dataset=lm_dataset,
            freq_cutoff=wandb.config.freq_cutoff,
        ),
        new_tokens=wandb.config.add_tokens,
        info_file=info_file,
    )

    trainable_token_ids: list[int] = []
    for format_str_attr in format_str_attrs:
        token_ids: list[int] = tokenizer(wandb.config[format_str_attr])["input_ids"]  # type: ignore
        tokens = tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
        print(f"{format_str_attr} tokens: {tokens}", file=info_file)
        if format_str_attr in wandb.config.train_params:
            trainable_token_ids += token_ids

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

    if wandb.config.pretrained_model is not None:
        model = AutoModelForCausalLM.from_pretrained(
            wandb.config.pretrained_model,
            revision=wandb.config.revision,
            device_map=device,
        )
        if n_added_tokens:
            model.resize_token_embeddings(len(tokenizer))
        initialize_new_token_embeddings(
            model,
            trainable_token_ids,
            wandb.config.embedding_init,
            old_vocab_size = len(tokenizer) - n_added_tokens,
        )
    else:
        config = AutoConfig.from_pretrained(
            wandb.config.config,
            vocab_size=len(tokenizer),  # size of the full vocabulary with the added tokens
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = AutoModelForCausalLM.from_config(config).to(device)
    print("model config:", file=info_file)
    print(model.config, file=info_file)
    model_max_length = getattr(model.config, model_max_length_attr[type(model.config)])
    tokenizer.model_max_length = model_max_length  # TODO: have effect only when calling tokenizer(..., truncation=True)
    if wandb.config.context_length is None:
        wandb.config.update(dict(context_length=model_max_length), allow_val_change=True)
    n_params = sum(map(torch.Tensor.nelement, model.parameters()))
    print(f"model #parameters: {n_params}", file=info_file)
    if "all" in wandb.config.train_params:
        trainable_params = list(model.parameters())
    else:
        trainable_tokens = tokenizer.convert_ids_to_tokens(trainable_token_ids)  # type: ignore
        print(f"trainable tokens: {trainable_tokens}", file=info_file)
        trainable_params = freeze_non_embedding_params(model)

    loss_fct = CrossEntropyLoss(reduction=wandb.config.loss_reduction, ignore_index=tokenizer.pad_token_id)  # type: ignore
    raw_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)  # type: ignore
    optimizer = AdamW(trainable_params, lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=wandb.config.factor, patience=wandb.config.patience)

    in_context_format = set_and_get_format(tokenizer, wandb.config)

    step = 0
    logging_loss, logging_n_tokens = 0., 0

    def build_lm_dataloader(
            dataset,
            batch_size,
            drop_last=False,
            shuffle=True,
            seed=None,
            n_examples=wandb.config.n_examples,
    ):
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        dataset = sample_lm_seq(
            dataset,
            n_examples,
            in_context_format,
        )
        dataset = dataset.map(
            lambda batch: tokenizer(batch["examples"]),
            batched=True,
            remove_columns=["examples"],
        )
        dataloader = DataLoader(
            dataset, # type: ignore
            batch_size=batch_size,
            shuffle=False,  # no need to shuffle the sequences again
            drop_last=drop_last,
            collate_fn=collator,
        )
        return dataloader

    if wandb.config.lm:
        val_lm_dataloader = build_lm_dataloader(
            lm_dataset["validation"],
            wandb.config.eval_batch_size,
            seed=wandb.config.eval_seed,
        )
    else:
        val_lm_dataloader = None
    val_meta_ind_dataset = sample_examples(
        meta_dataset["validation"],
        wandb.config.n_examples,
        max_sample_times=wandb.config.max_sample_times,
        rng=np.random.default_rng(wandb.config.eval_seed)
    )
    val_meta_unique_dataset = sample_examples(
        meta_dataset["validation"],
        wandb.config.n_examples,
        max_sample_times=1,  # ensure different words in a classification batch
        rng=np.random.default_rng(wandb.config.eval_seed)
    )
    val_meta_ind_lm_dataset = val_meta_ind_dataset.map(in_context_format.construct_meta_lm_example).map(lambda batch: tokenizer(batch["examples"]), batched=True, remove_columns=["word", "examples"])
    val_meta_ind_lm_dataloader = DataLoader(
        val_meta_ind_lm_dataset, # type: ignore
        batch_size=wandb.config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    val_meta_unique_lm_dataset = val_meta_unique_dataset.map(in_context_format.construct_meta_lm_example).map(lambda batch: tokenizer(batch["examples"]), batched=True, remove_columns=["word", "examples"])
    val_meta_unique_lm_dataloader = DataLoader(
        val_meta_unique_lm_dataset, # type: ignore
        batch_size=wandb.config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    val_meta_cls_dataset = val_meta_unique_dataset.map(in_context_format.construct_meta_cls_example)
    val_meta_cls_dataloaders = {
        n_classes: DataLoader(
            val_meta_cls_dataset, # type: ignore
            batch_size=n_classes,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(cls_collate_fn, tokenizer),
        )
        for n_classes in wandb.config.eval_n_classes
    }
    wandb.define_metric("val_ind_lm_loss", summary="min")
    wandb.define_metric("val_unique_lm_loss", summary="min")
    for n_classes in val_meta_cls_dataloaders.keys():
        value_name = f"val_cls_{n_classes}_acc"
        wandb.define_metric(value_name, summary="max")

    global best_val_meta_ind_lm_loss
    best_val_meta_ind_lm_loss = np.inf
    def _evaluate(scheduler_step: bool = True):
        global best_val_meta_ind_lm_loss
        model.eval()

        if wandb.config.lm:
            val_lm_loss = evaluate_lm(model, val_lm_dataloader, loss_fct)
            print(f"{val_lm_loss=:.6f}")
            wandb.log({"val_lm_loss": val_lm_loss}, step=step)

        val_meta_ind_lm_loss = evaluate_lm(model, val_meta_ind_lm_dataloader, loss_fct)
        print(f"{val_meta_ind_lm_loss=:.6f}")
        wandb.log({"val_ind_lm_loss": val_meta_ind_lm_loss}, step=step)
        if best_val_meta_ind_lm_loss > val_meta_ind_lm_loss:
            best_val_meta_ind_lm_loss = val_meta_ind_lm_loss
            save_checkpoint(Path(wandb.config.ckpt_dir, kwargs["name"], "best"), model, optimizer, scheduler)

        val_meta_unique_lm_loss = evaluate_lm(model, val_meta_unique_lm_dataloader, loss_fct)
        print(f"{val_meta_unique_lm_loss=:.6f}")
        wandb.log({"val_unique_lm_loss": val_meta_unique_lm_loss}, step=step)

        for n_classes, val_meta_cls_dataloader in val_meta_cls_dataloaders.items():
            value_name = f"val_cls_{n_classes}_acc"
            val_cls_acc = evaluate_cls(model, val_meta_cls_dataloader, raw_loss_fct)
            print(f"{value_name}={frac_repr(*val_cls_acc, prec=3)}")
            wandb.log({value_name: frac_tuple_to_float(val_cls_acc)}, step=step)

        if scheduler_step:
            scheduler.step(val_meta_ind_lm_loss)
        wandb.log({"lr": scheduler._last_lr[0]}, step=step) # type: ignore

    wandb.log({"epoch": 0}, step=step)
    _evaluate()

    print(f'original train meta_dataset size: #episodes: {len(meta_dataset["train"])} #examples: {sum(map(len, meta_dataset["train"]["examples"]))}')
    for epoch_i in range(wandb.config.n_epochs):
        print(f"Epoch {epoch_i}:")
        train_meta_dataset = sample_examples(
            meta_dataset["train"],
            wandb.config.n_examples,
            max_sample_times=wandb.config.max_sample_times,
        )
        print(f'train meta_dataset size: #episodes: {len(train_meta_dataset)} #examples: {sum(map(len, train_meta_dataset["examples"]))}')
        if wandb.config.concat:
            print("Concatenate all examples")
            train_meta_dataset = sample_examples(train_meta_dataset, 1, rng=None)  # flatten examples
            train_meta_dataset = train_meta_dataset.map(partial(in_context_format.construct_meta_lm_example, start_with_sep=False))
            train_meta_dataset = train_meta_dataset.map(lambda batch: tokenizer(batch["examples"]), batched=True)
            train_meta_dataset = np.fromiter(chain.from_iterable(train_meta_dataset["input_ids"]), int)  # concatenate token ids
            context_length = wandb.config.context_length
            train_meta_dataset = ConcatLMDataset(
                train_meta_dataset,
                context_length,
                offset=np.random.randint(context_length),
            )
            train_meta_dataloader = DataLoader(
                train_meta_dataset,
                batch_size=wandb.config.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=concat_lm_collate,
            )
            train_lm_dataloader = None  # TODO: implement train_lm_dataloader
        else:
            train_meta_dataset = train_meta_dataset.map(in_context_format.construct_meta_lm_example).map(lambda batch: tokenizer(batch["examples"]), batched=True, remove_columns=["word", "examples"])
            train_meta_dataloader = DataLoader(
                train_meta_dataset, # type: ignore
                batch_size=wandb.config.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collator,
            )
            if wandb.config.lm:
                train_lm_dataloader = build_lm_dataloader(
                    lm_dataset["train"],
                    wandb.config.batch_size,
                    drop_last=True,
                )
            else:
                train_lm_dataloader = None
        train_dataloader = train_meta_dataloader
        if train_lm_dataloader is not None:
            train_dataloader = mix_iter(train_dataloader, train_lm_dataloader)

        total_loss, total_n_tokens = 0., 0
        model.train()
        for batch in train_dataloader:
            batch = to(batch, device)
            try:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            except:
                input_ids = batch["input_ids"]
                print(f"{input_ids.shape=}")
                decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                print("input:")
                for s in decoded_input:
                    print(s)
                raise
            loss = loss_fct(outputs.logits[..., :-1, :].movedim(-1, 1), batch["input_ids"][..., 1:])
            loss.backward()
            if wandb.config.train_params == "new_word":
                zero_grad_embedding_params(model, except_token_ids=trainable_token_ids)
            optimizer.step()
            model.zero_grad()
            loss_value = loss.item()
            n_tokens = batch["attention_mask"][..., 1:].sum().item()
            wandb.log({"train_loss_step": loss_value/n_tokens}, step=step)
            total_loss += loss_value
            total_n_tokens += n_tokens
            logging_loss += loss_value
            logging_n_tokens += n_tokens
            step += 1
            if step % wandb.config.logging_step == 0:
                logging_mean_loss = logging_loss / logging_n_tokens
                print(f"{step=} loss={logging_mean_loss:.6f}")
                wandb.log({"train_loss_mean": logging_mean_loss}, step=step)
                logging_loss, logging_n_tokens = 0., 0
            if wandb.config.eval_step and step % wandb.config.eval_step == 0:
                _evaluate()
                save_checkpoint(Path(wandb.config.ckpt_dir, kwargs["name"], "last"), model, optimizer, scheduler)
        train_loss = total_loss / total_n_tokens
        print(f"{train_loss=:.6f}")
        wandb.log({"train_loss": train_loss}, step=step)

        save_checkpoint(Path(wandb.config.ckpt_dir, kwargs["name"], "last"), model, optimizer, scheduler)

        wandb.log({"epoch": epoch_i+1}, step=step)
        _evaluate()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--name", required=True,
        help="Experiment/run name."
    )
    argparser.add_argument(
        "--data_dir", type=Path,
        default=Path("word_use_data", "childes", "word"),
        help="Path to the dataset."
    )
    argparser.add_argument(
        "--ckpt_dir", type=Path,
        default=Path("ckpt"),
        help="Path to the checkpoint directory."
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
        "--train_params",
        nargs = "+",
        choices = ["all"] + format_str_attrs,
        default = ["all"],
        help = r"Train only these parameters and freeze others. "
               r"all: all parameters (default); "
               f"*others* ({', '.join(format_str_attrs)}): token embeddings constituting these parts."
    )
    argparser.add_argument(
        "--config",
        help="pretrained_model_name_or_path for AutoConfig."
    )
    argparser.add_argument(
        "--lm", action="store_true",
        help="Do language modeling as an objective."
    )
    argparser.add_argument(
        "--concat", action="store_true",
        help="Train LM on concatenated text."
    )
    argparser.add_argument(
        "--context_length", type=int,
        help="Context length used in training LM on concatenated text."
    )
    argparser.add_argument(
        "--freq_cutoff", type=int, default=4,
        help="Exclude tokens with frequency <= freq_cutoff from the vocabulary."
    )
    argparser.add_argument(
        "--n_epochs", type=int, default=80,
        help="Number of epochs for training."
    )
    argparser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for training."
    )
    argparser.add_argument(
        "--eval_batch_size", type=int,
        help="Batch size for lm evaluation."
    )
    argparser.add_argument(
        "--n_examples", type=int, default=4,
    )
    argparser.add_argument(
        "--max_sample_times", type=int, default=0,
    )
    argparser.add_argument(
        "--eval_n_classes", type=int, nargs="+", default=[2],
        help="Number of classes for evaluation classification task."
    )
    argparser.add_argument(
        "--loss_reduction", default="sum",
        help="Loss reduction."
    )
    argparser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate."
    )
    argparser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay."
    )
    argparser.add_argument(
        "--factor", type=float, default=0.1,
        help="Factor by which the learning rate will be reduced."
    )
    argparser.add_argument(
        "--patience", type=int, default=2,
        help="Number of epochs with no improvement after which learning rate will be reduced."
    )
    argparser.add_argument(
        "--seed", type=int,
        help="Random seed."
    )
    argparser.add_argument(
        "--eval_seed", type=int, default=0,
        help="Random seed for evaluation."
    )
    argparser.add_argument(
        "--logging_step", type=int, default=100,
    )
    argparser.add_argument(
        "--eval_step", type=int,
    )
    args = argparser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    name = args.name
    del args.name # type: ignore
    main(config=args, name=name)