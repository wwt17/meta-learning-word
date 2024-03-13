from typing import Any, Optional, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial
from itertools import islice, chain
import pickle
import torch
import tqdm
from text_configs import SEP_TOKEN, NEW_TOKEN


def frac_repr(a, b, prec=2):
    return f"{a}/{b}={a/b:.{prec}%}"


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


def concat_examples(examples, sep: str = SEP_TOKEN, space: str = " ", t: Optional[str] = NEW_TOKEN) -> str:
    return sep + space + space.join((example_str(example, t=t) + space + sep for example in examples))