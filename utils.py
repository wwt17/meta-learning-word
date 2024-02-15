from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict
from pathlib import Path
from functools import partial
from itertools import islice, chain
import pickle
import torch
import tqdm


def frac_repr(a, b, prec=2):
    return f"{a}/{b}={a/b:.{prec}%}"


def paired(g, n=2):
    g = iter(g)
    while True:
        p = tuple(islice(g, n))
        if len(p) < n:
            break
        yield p


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