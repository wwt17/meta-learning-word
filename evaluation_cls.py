from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import transformers
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig, set_seed
from text_configs import SEP_TOKEN
from utils import map_structure, to, get_device, example_str, concat_examples


def construct_cls_example(item, sep_token=SEP_TOKEN):
    return {
        "prefix": concat_examples(item["examples"][:-1]),
        "suffix": " "+example_str(item["examples"][-1])+" "+sep_token,
    }


def cls_collate_fn(tokenizer, batch):
    """Construct a classification evaluation batch."""
    batch = {
        key: type(batch)(example[key] for example in batch)
        for key in batch[0].keys()
    }
    for key in ["prefix", "suffix"]:
        batch[key] = tokenizer(
            batch[key],
            padding="longest", return_tensors="pt"
        )
    return batch


def get_suffix_losses(prefix_input, suffix_input, model, loss_fct):
    batch_size = prefix_input["input_ids"].size(0)
    suffix_losses = []
    prefix_output = model(input_ids=prefix_input["input_ids"], attention_mask=prefix_input["attention_mask"], use_cache=True)
    prefix_lengths = prefix_input["attention_mask"].sum(1)
    for i_prefix in range(batch_size):
        prefix_length = prefix_lengths[i_prefix].item()
        prefix_past_key_values = map_structure(
            lambda t: t[i_prefix, ..., :prefix_length, :].expand(batch_size, *((t.dim()-1)*[-1])),
            prefix_output.past_key_values
        )
        attention_mask = torch.concat(
            [prefix_input["attention_mask"][i_prefix, :prefix_length].expand(batch_size, -1),
                suffix_input["attention_mask"]],
            dim=1
        )
        suffix_output = model(input_ids=suffix_input["input_ids"], attention_mask=attention_mask, past_key_values=prefix_past_key_values)
        suffix_loss = loss_fct(
            torch.concat(
                [prefix_output.logits[i_prefix, prefix_length-1:prefix_length, :].expand(batch_size, -1, -1),
                 suffix_output.logits[..., :-1, :]],
                dim=-2
            ).movedim(-1, 1),
            suffix_input["input_ids"]
        )
        suffix_losses.append(suffix_loss)
    suffix_losses = torch.stack(suffix_losses, dim=0)
    return suffix_losses


def get_nll_matrix(prefix_input, suffix_input, model, loss_fct):
    suffix_losses = get_suffix_losses(prefix_input, suffix_input, model, loss_fct)
    nll_matrix = suffix_losses.sum(-1)
    return nll_matrix


def evaluate_cls(model, dataloader, loss_fct) -> float:
    device = get_device(model)
    n_acc, n = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            for key in ["prefix", "suffix"]:
                batch[key] = to(batch[key], device)
            prefix_input = batch["prefix"]
            suffix_input = batch["suffix"]
            batch_size = prefix_input["input_ids"].size(0)
            nll_matrix = get_nll_matrix(prefix_input, suffix_input, model, loss_fct)
            pred_cls = nll_matrix.argmin(dim=0)
            acc = pred_cls == torch.arange(batch_size, device=pred_cls.device)
            batch_n_acc = acc.sum().item()
            n_acc += batch_n_acc
            n += batch_size

    return n_acc / n


if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        batch = [
            {"prefix": "32767 + 1 =", "suffix": " 32768"},
            {"prefix": "1 + 1 =", "suffix": " 2"},
            {"prefix": "2 + 1 =", "suffix": " 3"},
        ]
        batch_size = len(batch)
        full_batch = [example["prefix"] + example["suffix"] for example in batch]
        batch = cls_collate_fn(tokenizer, batch)
        prefix_input = batch["prefix"]
        suffix_input = batch["suffix"]
        full_input = tokenizer(full_batch, padding="longest", return_tensors="pt")
        print(prefix_input)
        print(suffix_input)
        print(full_input)
        full_output = model(input_ids=full_input.input_ids, attention_mask=full_input.attention_mask, use_cache=True)
        full_loss = loss_fct(full_output.logits[..., :-1, :].movedim(-1, 1), full_input.input_ids[..., 1:])
        print(full_loss)
        suffix_losses = get_suffix_losses(prefix_input, suffix_input, model, loss_fct)
        print(suffix_losses.shape)
        print(suffix_losses)
        nll_matrix = suffix_losses.sum(-1)
        print(nll_matrix)
        pred_cls = nll_matrix.argmin(dim=0)
        print(pred_cls)
        acc = pred_cls == torch.arange(batch_size, device=pred_cls.device)
        print(acc)
        batch_n_acc = acc.sum().item()
        print(batch_n_acc)