from typing import Any, Optional, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial
from itertools import islice, chain
import pickle
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from text_configs import SEP_TOKEN, NEW_TOKEN


def frac_repr(a, b, prec=2):
    return f"{a}/{b}={a/b:.{prec}%}"


def normalize_dict(d):
    sum_value = sum(d.values())
    return {key: value / sum_value for key, value in d.items()}


def zipdict(d: Mapping):
    for value_tuple in zip(*d.values()):
        yield {key: value for key, value in zip(d.keys(), value_tuple)}


def batchify(examples, batch_size=2, drop_last=True):
    examples = iter(examples)
    while True:
        batch = tuple(islice(examples, batch_size))
        if len(batch) == 0 or (drop_last and len(batch) < batch_size):
            break
        yield batch


def map_structure(func, *structure):
    if isinstance(structure[0], torch.Tensor):
        return func(*structure)
    elif isinstance(structure[0], Sequence):
        return type(structure[0])(map_structure(func, *substructure) for substructure in zip(*structure))
    elif isinstance(structure[0], Mapping):
        return {key: map_structure(func, *(struct[key] for struct in structure)) for key in structure[0].keys()}
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


def replace_at_offsets(s: str, offsets: Sequence[tuple[int, int]], t: str) -> str:
    """Replace s at offsets by t.
    """
    offsets = sorted(offsets)
    for offset in reversed(offsets):
        s = s[:offset[0]] + t + s[offset[1]:]
    return s


def example_str(example, t: Optional[str] = NEW_TOKEN) -> str:
    s = example["sentence"]
    if t is not None:
        s = replace_at_offsets(s, example["offsets"], t)
    return s


def concat_strs(strs: Iterable[str], sep: str = SEP_TOKEN, space: str = " ", start_with_sep=True) -> str:
    return (sep + space if start_with_sep else "") + space.join((s + space + sep for s in strs))


def concat_examples(examples, sep: str = SEP_TOKEN, space: str = " ", t: Optional[str] = NEW_TOKEN, start_with_sep=True) -> str:
    return concat_strs(
        (example_str(example, t=t) for example in examples),
        sep=sep, space=space, start_with_sep=start_with_sep,
    )


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
