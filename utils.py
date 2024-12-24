from typing import Any, Optional, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial
from itertools import islice, chain
import re
import pickle
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import transformers


def frac_repr(a, b, prec=2):
    return f"{a}/{b}={a/b:.{prec}%}"


def frac_tuple_to_float(t):
    a, b = t
    return a / b


def normalize_dict(d):
    sum_value = sum(d.values())
    return {key: value / sum_value for key, value in d.items()}


def zipdict(d: Mapping):
    for value_tuple in zip(*d.values()):
        yield {key: value for key, value in zip(d.keys(), value_tuple)}


def merge_input_output_dict(fn):
    def wrapper(input_dict):
        output_dict = fn(input_dict)
        return {**input_dict, **output_dict}
    return wrapper


def batchify(examples, batch_size=2, drop_last=True):
    examples = iter(examples)
    while True:
        batch = tuple(islice(examples, batch_size))
        if len(batch) == 0 or (drop_last and len(batch) < batch_size):
            break
        yield batch


def map_structure(func, *structure, classinfo: Any = torch.Tensor):
    if isinstance(structure[0], classinfo):
        return func(*structure)
    elif isinstance(structure[0], Sequence):
        return type(structure[0])(map_structure(func, *substructure, classinfo=classinfo) for substructure in zip(*structure))
    elif isinstance(structure[0], Mapping):
        return type(structure[0])({key: map_structure(func, *(struct[key] for struct in structure), classinfo=classinfo) for key in structure[0].keys()})
    else:
        raise ValueError(f"Unknown structure {structure}")


def to(data, *args, **kwargs) -> Any:
    if hasattr(data, "to"):
        return data.to(*args, **kwargs)
    elif isinstance(data, Sequence):
        return type(data)(to(e, *args, **kwargs) for e in data)
    elif isinstance(data, Mapping):
        return {key: to(value, *args, **kwargs) for key, value in data.items()}
    else:
        raise ValueError(f"Unknown data {data} of type {type(data)}")


def get_device(module: torch.nn.Module):
    return next(module.parameters()).device


def _pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def cache(cache_path, loader=_pickle_load, saver=_pickle_save):
    cache_path = Path(cache_path)

    def decorator(fn):
        def wrapper(*args, **kwargs):
            if cache_path.exists():
                # load from cache
                print(f'load from {cache_path}')
                data = loader(cache_path)

            else:
                data = fn(*args, **kwargs)
                # save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                saver(data, cache_path)

            return data

        return wrapper

    return decorator


torch_cache = partial(cache, loader=torch.load, saver=torch.save)


def get_pos_tags(sentences, nlp):
    pos_tags = []
    for doc in nlp.pipe(tqdm.tqdm(sentences, desc="POS tagging")):
        sentence_pos_tags = [token.tag_ for token in doc]
        pos_tags.append(sentence_pos_tags)
    return pos_tags


def count_tokens(
        pre_tokenized_sentences: Iterable[Sequence[tuple[str, Any]]]
):
    counter = defaultdict(list)
    for i, word_offsets in enumerate(pre_tokenized_sentences):
        for word, offset in word_offsets:
            counter[word].append((i, offset))
    return counter


def count_token_pos(
        pre_tokenized_sentences: Iterable[Sequence[tuple[str, Any]]],
        tags: Iterable[Sequence[Any]],
):
    counter = defaultdict(list)
    tag_counter = defaultdict(lambda: defaultdict(int))
    for i, (word_offsets, tags_) in enumerate(zip(pre_tokenized_sentences, tags)):
        assert len(word_offsets) == len(tags_), f"{word_offsets=} {tags_=}"
        for (word, offset), tag in zip(word_offsets, tags_):
            counter[word].append((i, offset))
            tag_counter[word][tag] += 1
    return counter, tag_counter


KeyType = TypeVar('KeyType')
def sorted_counter_list(counter: Mapping[KeyType, Sized]) -> list[tuple[KeyType, Sized]]:
    return sorted(counter.items(), key=lambda item: (-len(item[1]), item[0]))

def sorted_counter_dict(counter: Mapping[KeyType, Sized]) -> dict[KeyType, Sized]:
    return dict(sorted_counter_list(counter))

TagType = TypeVar('TagType')
def get_max_freq_tag(tag_counts: Mapping[TagType, int]) -> tuple[TagType, int]:
    return min(tag_counts.items(), key=(lambda item: -item[1]))

def get_max_freq_tag_vocab(tag_counter: Mapping[KeyType, Mapping[TagType, int]]) -> Mapping[KeyType, tuple[TagType, int]]:
    return {
        word: get_max_freq_tag(tag_counts)
        for word, tag_counts in tag_counter.items()
    }


def plot_bar(
        x, y, bottom=None, color=None, hue=None, palette=None,
        width=1, align='center',
        x_min=None, x_max=None,
        x_tick_step=None,
        y_min=None, y_max=None,
        y_tick_step=None,
        y_position='right',
        legend=False,
        ax=None,
        **kwargs
):
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = len(x)
    if (x_min, x_max) != (0, len(x)):
        x_slice = slice(x_min, x_max)
        x = x[x_slice]
        y = y[x_slice]
        if bottom is not None:
            bottom = bottom[x_slice]
        if color is not None:
            color = color[x_slice]
        if hue is not None:
            hue = hue[x_slice]
    if ax is None:
        ax = plt.gca()

    if hue is not None:
        if palette is not None:
            color = [palette[h] for h in hue]
            if legend:
                used_hue = set(hue)
                if isinstance(palette, dict):
                    legend_palette = {key: value for key, value in palette.items() if key in used_hue}
                else:
                    legend_palette = {i: value for i, value in enumerate(palette) if i in used_hue}
                patches = [Patch(color=color, label=label) for label, color in legend_palette.items()]
                ax.legend(handles=patches)
        else:
            color = hue

    align_dx = {'center': -0.5, 'edge': 0}[align]
    ax.set_xlim(x_min + align_dx, x_max + align_dx)
    if x_tick_step is not None:
        ax.set_xticks(np.arange(x_min, x_max, x_tick_step))

    if (y_min, y_max) != (None, None):
        ax.set_ylim(y_min, y_max)
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = max(y)+1
    if y_tick_step is not None:
        ax.set_yticks(np.arange(y_min, y_max, y_tick_step))
    if y_position is not None:
        ax.yaxis.set_ticks_position(y_position)
        ax.yaxis.set_label_position(y_position)
    ax.grid(axis='y')

    return ax.bar(x, y, bottom=bottom, color=color, width=width, align=align, **kwargs)


tokenization_space_pattern = re.compile(
    r'(?<=\bcan)\s(?=not\b)'
    r'|(?<=\bcan)\sn(?=\'t\b)'
    r'|\s(?=[\.\?\!,]|(n\'t|\'m|\'s|\'re|\'ve|\'d|\'ll)\b)'
    r'|(?<=\b(gon|wan))\s(?=na\b)'
    r'|(?<=\bgot)\s(?=ta\b)'
    r'|(?<=\(|\[|\{)\s'
    r'|\s(?=\)|\]|\})'
)
def clean_up_tokenization_spaces_for_example(
        example,
        tokenization_space_pattern=tokenization_space_pattern,
):
    sentence, offsets = example["sentence"], example["offsets"]
    char_is_retained = np.full(len(sentence), True)
    def _repl(m):
        char_is_retained[m.start():m.end()] = False
        return ""
    new_sentence = tokenization_space_pattern.sub(_repl, sentence)
    offset_mapping = np.insert(np.cumsum(char_is_retained), 0, 0)
    new_offsets = [list(offset_mapping[np.array(offset)]) for offset in offsets]
    return {**example, "sentence": new_sentence, "offsets": new_offsets}


def prepend_to_example(prepend_str: str, example):
    sentence, offsets = example["sentence"], example["offsets"]
    new_sentence = prepend_str + sentence
    offset_shift = len(prepend_str)
    new_offsets = [[offset_shift + o for o in offset] for offset in offsets]
    return {**example, "sentence": new_sentence, "offsets": new_offsets}


def _offset_with_leading_space(
        offset: tuple[int, int],
        s: str,
        leading_space: str,
):
    start, end = offset
    while start > 0 and s[start - 1] == leading_space:
        start -= 1
    return (start, end)


def replace_at_offsets(
        s: str,
        offsets: Sequence[tuple[int, int]],
        t: str,
        leading_space: Optional[str] = None,
) -> str:
    """Replace s at offsets by t.
    """
    if leading_space is not None:
        offsets = [
            _offset_with_leading_space(offset, s, leading_space)
            for offset in offsets
        ]
    offsets = sorted(offsets)
    for offset in reversed(offsets):
        s = s[:offset[0]] + t + s[offset[1]:]
    return s


def example_str(
        example,
        t: Optional[str],
        leading_space: Optional[str] = " ",
) -> str:
    s = example["sentence"]
    if t is not None:
        s = replace_at_offsets(s, example["offsets"], t, leading_space)
    return s


def mix_iter(*iters):
    its = []
    n = []
    for it in iters:
        n_it = len(it)
        if n_it:
            its.append(iter(it))
            n.append(n_it)
    if not its:
        return
    n_left = n[:]
    while True:
        max_left_i = 0
        for i in range(len(iters)):
            if n_left[i] * n[max_left_i] > n_left[max_left_i] * n[i]:
                max_left_i = i
        if n_left[max_left_i] > 0:
            n_left[max_left_i] -= 1
            yield next(its[max_left_i])
        else:
            break


def unique(elements, equal_func = lambda a, b: a == b):
    unique_elements = []
    for element in elements:
        for element_ in unique_elements:
            if equal_func(element, element_):
                break
        else:
            unique_elements.append(element)
            yield element


def get_embedding_params(model: transformers.PreTrainedModel):
    yield from unique(
        chain(
            model.get_input_embeddings().parameters(),
            model.get_output_embeddings().parameters()
        ),
        lambda a, b: a is b
    )


def initialize_new_token_embeddings(
        model: transformers.PreTrainedModel,
        new_token_ids,
        method: str,
        old_vocab_size: Optional[int] = None,
):
    if method == "none":
        return
    for param in get_embedding_params(model):
        _requires_grad = param.requires_grad
        param.requires_grad = False
        if method == "mean":
            assert old_vocab_size is not None, "Must provide old_vocab_size for mean"
            mean = param[:old_vocab_size].mean(0).detach()
            param[new_token_ids] = mean
        else:
            raise ValueError(f"Unknown embedding intialization method: {method}")
        param.requires_grad = _requires_grad


def freeze_non_embedding_params(model: transformers.PreTrainedModel):
    for param in model.parameters():
        param.requires_grad = False
    embedding_params = list(get_embedding_params(model))
    for param in embedding_params:
        param.requires_grad = True
    return embedding_params


def get_frozen_tokens_mask(vocab_size: int, except_token_ids, device):
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    mask[except_token_ids] = False
    return mask


def zero_grad_embedding_params(model: transformers.PreTrainedModel, except_token_ids=[], vocab_size=None, mask=None):
    if vocab_size is None:
        vocab_size = next(get_embedding_params(model)).size(0)

    if mask is None:
        mask = get_frozen_tokens_mask(vocab_size, except_token_ids, model.device)
    else:
        assert vocab_size == mask.size(0)

    for param in get_embedding_params(model):
        assert param.size(0) == vocab_size, f"Unexpected embedding parameter size {param.size()}"
        param.grad.data[mask] = 0  # type: ignore