from typing import Any, Iterable, TypeVar, Optional
from collections.abc import Sequence, Mapping, Sized, Callable
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from itertools import islice, chain
from functools import partial
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig, set_seed
from text_configs import NEW_TOKEN, SEP_TOKEN
from in_context_format import InContextFormat
from emb_gen import EmbGener
from utils import map_structure, to, get_device, batchify, set_token_embeddings


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


def get_prefix_output(
        prefix_input,
        model: PreTrainedModel,
):
    prefix_output = model(input_ids=prefix_input["input_ids"], attention_mask=prefix_input["attention_mask"], use_cache=True)
    return prefix_output


def get_embs_(
        study_input_,
        emb_gener: EmbGener,
):
    mlm_out = emb_gener.mlm(
        input_ids=study_input_["input_ids"],
        attention_mask=study_input_["attention_mask"],
        output_hidden_states=True,
    )
    embs_ = emb_gener.model(
        mlm_out.hidden_states[-1],
        study_input_["attention_mask"],
    )
    return embs_


def get_suffix_loss(
        prefix_input,
        prefix_output,
        i_prefix: int,
        suffix_input,
        model: PreTrainedModel,
        loss_fct: Callable,
        emb_gener: Optional[EmbGener] = None,
        study_input: Optional[list] = None,
        embs: Optional[list] = None,
):
    n_suffix = suffix_input["input_ids"].size(0)
    prefix_length = prefix_input["attention_mask"][i_prefix].sum()
    if emb_gener is not None:
        assert study_input is not None
        study_input_ = study_input[i_prefix]
        if embs is None:
            embs_ = get_embs_(study_input_, emb_gener)
        else:
            embs_ = embs[i_prefix]
        set_token_embeddings(model, [emb_gener.token_id], embs_)
    if prefix_output is None:
        prefix_input_ = map_structure(
            lambda t: t[i_prefix:i_prefix+1, ..., :prefix_length],
            prefix_input
        )
        prefix_output_ = get_prefix_output(prefix_input_, model)
        i_prefix_ = 0
    else:
        prefix_input_ = prefix_input
        prefix_output_ = prefix_output
        i_prefix_ = i_prefix
    prefix_past_key_values = map_structure(
        lambda t: t[i_prefix_, ..., :prefix_length, :].expand(n_suffix, *((t.dim()-1)*[-1])),
        prefix_output_.past_key_values
    )
    attention_mask = torch.concat(
        [
            prefix_input_["attention_mask"][i_prefix_, :prefix_length].expand(n_suffix, -1),
            suffix_input["attention_mask"]
        ],
        dim=1
    )
    suffix_output = model(input_ids=suffix_input["input_ids"], attention_mask=attention_mask, past_key_values=prefix_past_key_values)
    suffix_loss = loss_fct(
        torch.concat(
            [
                prefix_output_.logits[i_prefix_, prefix_length-1:prefix_length, :].expand(n_suffix, -1, -1),
                suffix_output.logits[..., :-1, :]
            ],
            dim=-2
        ).movedim(-1, 1),
        suffix_input["input_ids"]
    )
    return suffix_loss


def get_suffix_losses(
        prefix_input,
        suffix_input,
        model: PreTrainedModel,
        loss_fct: Callable,
        prefix_output=None,
        emb_gener: Optional[EmbGener] = None,
        study_input: Optional[list] = None,
        embs: Optional[list] = None,
):
    n_prefix = prefix_input["input_ids"].size(0)
    suffix_losses = []
    if emb_gener is None:
        if prefix_output is None:
            prefix_output = get_prefix_output(prefix_input, model)
    for i_prefix in range(n_prefix):
        suffix_loss = get_suffix_loss(
            prefix_input,
            prefix_output,
            i_prefix,
            suffix_input,
            model,
            loss_fct,
            emb_gener=emb_gener,
            study_input=study_input,
            embs=embs,
        )
        suffix_losses.append(suffix_loss)
    suffix_losses = torch.stack(suffix_losses, dim=0)
    return suffix_losses


def get_nll_matrix(
        prefix_input,
        suffix_input,
        model: PreTrainedModel,
        loss_fct: Callable,
        prefix_output=None,
        emb_gener: Optional[EmbGener] = None,
        study_input: Optional[list] = None,
        embs: Optional[list] = None,
):
    suffix_losses = get_suffix_losses(
        prefix_input,
        suffix_input,
        model,
        loss_fct,
        prefix_output=prefix_output,
        emb_gener=emb_gener,
        study_input=study_input,
        embs=embs,
    )
    nll_matrix = suffix_losses.sum(-1)
    return nll_matrix


def evaluate_cls(
        model: PreTrainedModel,
        dataloader: Iterable,
        loss_fct: Callable,
) -> tuple[int, int]:
    """Evaluate classification.
    Args:
        model: the LM to evaluate.
        dataloader: Classification batches. In each batch, try matching each
            prefix to each suffix.
        loss_fct: CrossEntropyLoss with reduction="none".
    Return:
        Accuracy in the fractional form (n_acc, n)
    """
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
            batch_n_acc: int = acc.sum().item()  # type: ignore
            n_acc += batch_n_acc
            n += batch_size

    return (n_acc, n)


def evaluate_cls_with_fixed_prefixes(
        model: PreTrainedModel,
        prefix_input,
        suffix_input_batches,
        loss_fct: Callable,
        label: int,
        emb_gener: Optional[EmbGener] = None,
        study_input: Optional[list] = None,
) -> tuple[int, int]:
    """Evaluate classification with a fixed set of in-context-learned words.
    Args:
        model: the LM to evaluate.
        tokenizer: the tokenizer used to tokenize the text.
        prefix_input: the tokenized batch of prefixes.
        suffix_input_batches: Tokenized batches of suffix inputs. In each
            batch, try matching each prefix to each suffix.
        loss_fct: CrossEntropyLoss with reduction="none".
        label: the correct label (word index) for all suffixes.
    Return:
        Accuracy in the fractional form (n_acc, n)
    """
    device = get_device(model)
    prefix_input = to(prefix_input, device)
    if study_input is not None:
        study_input = to(study_input, device)
    embs = None
    prefix_output = None
    if emb_gener is None:
        prefix_output = get_prefix_output(prefix_input, model)
    else:
        assert study_input is not None
        if False:
            embs = []
            with torch.no_grad():
                for i, study_input_ in enumerate(study_input):
                    embs_ = get_embs_(study_input_, emb_gener)
                    embs.append(embs_)
                    prefix_output = None  # TODO: get prefix output
    n_acc, n = 0, 0
    with torch.no_grad():
        for suffix_input in suffix_input_batches:
            suffix_input = to(suffix_input, device)
            batch_size = suffix_input["input_ids"].size(0)
            nll_matrix = get_nll_matrix(
                prefix_input,
                suffix_input,
                model,
                loss_fct,
                prefix_output=prefix_output,
                emb_gener=emb_gener,
                study_input=study_input,
                embs=embs,
            )
            pred_cls = nll_matrix.argmin(dim=0)
            acc = pred_cls == label
            batch_n_acc: int = acc.sum().item()  # type: ignore
            n_acc += batch_n_acc
            n += batch_size

    return (n_acc, n)


def evaluate_cls_with_fixed_words(
        word_items: list[dict],
        test_examples: Iterable[list[dict]],
        tokenizer,
        in_context_format: InContextFormat,
        model: PreTrainedModel,
        loss_fct: Callable,
        label: int,
        batch_size: int,
        drop_last: bool = False,
        emb_gener: Optional[EmbGener] = None,
) -> tuple[int, int]:
    """Evaluate classification with a fixed set of in-context-learned words.
    Args:
        word_items: a list, each element is a dict item for a
            word-to-be-learned, which will be converted into a prefix for
            in-context learning.
        test_examples: an Iterable of list of examples, each for a suffix.
        tokenizer: the tokenizer used to tokenize the text.
        model: the LM to evaluate.
        loss_fct: CrossEntropyLoss with reduction="none".
        label: the correct label (word index) for all suffixes.
        batch_size: Batch size.
    Return:
        Accuracy in the fractional form (n_acc, n)
    """
    prefixes, studies = [], None
    for word_item in word_items:
        word_item_data = in_context_format.construct_meta_cls_example(word_item, last_n=0)
        prefixes.append(word_item_data["prefix"])
        if "study" in word_item_data:
            if studies is None:
                studies = []
            studies.append(word_item_data["study"])
    prefix_input = tokenizer(
        prefixes,
        padding="longest",
        return_tensors="pt",
    )
    if emb_gener is None:
        study_input = None
    else:
        assert studies is not None
        study_input = [
            emb_gener.mlm_tokenizer(
                study,
                truncation=True,
                padding='longest',
                return_tensors='pt',
            )
            for study in studies
        ]
    suffixes = map(
        partial(in_context_format.concat_examples, start_with_sep=False),
        test_examples
    )
    suffix_batches = batchify(suffixes, batch_size=batch_size, drop_last=drop_last)
    suffix_input_batches = map(
        partial(
            tokenizer,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        ),
        suffix_batches
    )
    return evaluate_cls_with_fixed_prefixes(
        model,
        prefix_input,
        suffix_input_batches,
        loss_fct,
        label,
        emb_gener=emb_gener,
        study_input=study_input,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretrained_model", default="gpt2")
    argparser.add_argument("--tokenizer")
    args = argparser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.pretrained_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    if "gpt2" in args.tokenizer:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(tokenizer.vocab_size - 1)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map=device)
    model.eval()
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
        prefix_input = to(batch["prefix"], device)
        suffix_input = to(batch["suffix"], device)
        full_input = to(tokenizer(full_batch, padding="longest", return_tensors="pt"), device)
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