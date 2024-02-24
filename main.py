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
import pandas as pd
import seaborn as sns
import wandb
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import datasets
import tokenizers
import transformers
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoConfig, set_seed
from utils import frac_repr, cache, to, example_str, concat_examples
from text_configs import PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, NEW_TOKEN, SPECIAL_TOKENS, NEW_TOKENS
from data_processing import build_vocab
from data_loading import load_dataset, sample_examples
from evaluation_cls import construct_cls_example, cls_collate_fn, evaluate_cls


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer_cache = partial(
    cache,
    loader=PreTrainedTokenizerFast.from_pretrained,
    saver=PreTrainedTokenizerFast.save_pretrained,
)


def get_tokenizer(sentences: Iterable[str]):
    pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()  # type: ignore
    vocab = build_vocab(
        map(pre_tokenizer.pre_tokenize_str, sentences)
    )
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, UNK_TOKEN)  # type: ignore
    )
    tokenizer.pre_tokenizer = pre_tokenizer # type: ignore
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN),
        pad_token=PAD_TOKEN,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=SEP_TOKEN,
        sep_token=SEP_TOKEN,
    )
    return tokenizer


def save_checkpoint(ckpt_path, model, optimizer, scheduler):
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), ckpt_path/"optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_path/"scheduler.pt")


def construct_lm_example(item, sep_token=SEP_TOKEN):
    return {"examples": concat_examples(item["examples"], sep_token=sep_token)}


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


def main(project="meta-learning-word", **kwargs):
    wandb.init(project=project, **kwargs)

    if wandb.config.seed is not None:
        set_seed(wandb.config.seed)

    dataset = datasets.DatasetDict({
        split: load_dataset(Path(wandb.config.data_dir, f"{split}.json"))
        for split in ["train", "validation", "test"]
    })

    tokenizer = tokenizer_cache(Path(wandb.config.data_dir, "tokenizer"))(get_tokenizer)((
        example["sentence"]
        for examples in dataset["train"]["examples"]
        for example in examples
    ))

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

    config = AutoConfig.from_pretrained(wandb.config.config)
    model = AutoModelForCausalLM.from_config(config).to(device)

    loss_fct = CrossEntropyLoss(reduction=wandb.config.loss_reduction, ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=wandb.config.factor, patience=wandb.config.patience)

    step = 0
    logging_loss, logging_n_tokens = 0., 0
    val_dataset = sample_examples(dataset["validation"], wandb.config.n_examples, np.random.default_rng(wandb.config.eval_seed))
    val_lm_dataset = val_dataset.map(construct_lm_example).map(lambda batch: tokenizer(batch["examples"]), batched=True).remove_columns(["word", "examples"])
    val_lm_dataloader = DataLoader(
        val_lm_dataset, # type: ignore
        batch_size=wandb.config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    val_cls_dataset = val_dataset.map(construct_cls_example)
    val_cls_dataloader = DataLoader(
        val_cls_dataset, # type: ignore
        batch_size=wandb.config.eval_n_classes,
        shuffle=False,
        drop_last=True,
        collate_fn=partial(cls_collate_fn, tokenizer),
    )
    model.eval()
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_cls_acc", summary="max")
    val_loss = evaluate_lm(model, val_lm_dataloader, loss_fct)
    print(f"{val_loss=:.6f}")
    wandb.log({"epoch": 0, "val_loss": val_loss}, step=step)
    val_cls_acc = evaluate_cls(model, val_cls_dataloader, loss_fct)
    print(f"{val_cls_acc=:.3%}")
    wandb.log({"val_cls_acc": val_cls_acc}, step=step)
    best_val_cls_acc = val_cls_acc
    wandb.log({"lr": wandb.config.lr}, step=step)

    for epoch_i in range(wandb.config.n_epochs):
        print(f"Epoch {epoch_i}:")
        train_dataset = sample_examples(dataset["train"], wandb.config.n_examples, np.random.default_rng()).map(construct_lm_example).map(lambda batch: tokenizer(batch["examples"]), batched=True).remove_columns(["word", "examples"])
        train_dataloader = DataLoader(
            train_dataset, # type: ignore
            batch_size=wandb.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collator,
        )

        total_loss, total_n_tokens = 0., 0
        model.train()
        for batch in train_dataloader:
            batch = to(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fct(outputs.logits[..., :-1, :].movedim(-1, 1), batch["input_ids"][..., 1:])
            loss.backward()
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
        train_loss = total_loss / total_n_tokens
        print(f"{train_loss=:.6f}")
        wandb.log({"epoch": epoch_i+1, "train_loss": train_loss}, step=step)

        save_checkpoint(Path(wandb.config.ckpt_dir, kwargs["name"], "last"), model, optimizer, scheduler)

        model.eval()
        val_loss = evaluate_lm(model, val_lm_dataloader, loss_fct)
        print(f"{val_loss=:.6f}")
        wandb.log({"val_loss": val_loss}, step=step)
        val_cls_acc = evaluate_cls(model, val_cls_dataloader, loss_fct)
        print(f"{val_cls_acc=:.3%}")
        wandb.log({"val_cls_acc": val_cls_acc}, step=step)
        if best_val_cls_acc < val_cls_acc:
            best_val_cls_acc = val_cls_acc
            save_checkpoint(Path(wandb.config.ckpt_dir, kwargs["name"], "best"), model, optimizer, scheduler)
        scheduler.step(val_cls_acc)
        wandb.log({"lr": scheduler._last_lr[0]}, step=step) # type: ignore


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
        "--config", default="gpt2",
        help="pretrained_model_name_or_path for AutoConfig."
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
        "--eval_n_classes", type=int, default=2,
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
    args = argparser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    name = args.name
    del args.name # type: ignore
    main(config=args, name=name)