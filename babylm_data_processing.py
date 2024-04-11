from typing import Any, Iterable, TypeVar
from collections.abc import Sequence, Mapping, Sized
from collections import Counter, defaultdict, namedtuple, deque
import argparse
from pathlib import Path
from itertools import islice, chain
import re
import logging
import tqdm
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from utils import get_pos_tags


quote_pairs = [
    (r'“', r'”'),
    (r"`", r"'"),
    (r"``", r"''"),
    (r'"', r'"'),
    (r"'", r"'"),
]
bracket_pairs = [
    (r"(", r")"),
    (r"[", r"]"),
    (r"{", r"}"),
]

eos_puncts = [".", "?", "!", ";", ":"]


hyphen_edge_regexes = [
    re.compile(r"(?<=[^\s-])(?=--)|(?<=--)(?=[^\s-])"),
    re.compile(r"(?<=[^\s-])(?=—)|(?<=—)(?=[^\s-])"),
]
def add_space_around_hyphens(s):
    for hyphen_edge_regex in hyphen_edge_regexes:
        s = hyphen_edge_regex.sub(" ", s)
    return s


def refine_tokenizer(tokenizer):
    tokenizer.add_special_case("``", [{ORTH: "``"}])

    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?:{h})+".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            r"_+",
            r"=+",
            r"[\(\)\[\]\{\}\"]",
        ]
    )
    infix_re = compile_infix_regex(infixes)
    tokenizer.infix_finditer = infix_re.finditer

    return tokenizer


data_source_configs = {
    "aochildes": dict(line_sentences=True),
    "bnc_spoken": dict(line_sentences=True),
    "cbt": dict(space_tokenized=True, concat=True),
    "children_stories": dict(),
    "gutenberg": dict(concat=True),
    "open_subtitles": dict(line_sentences=True),
    "simple_wikipedia": dict(),
    "switchboard": dict(line_sentences=True),
    "wikipedia": dict(),
}


@Language.component("single_sentencizer")
def single_sentencizer(doc):
    for token in doc:
        if token.is_sent_start is None:
            token.is_sent_start = False
    return doc


@Language.factory("my_sentencizer")
def my_sentencizer_factory(nlp, name):
    quotes = set(chain.from_iterable(quote_pairs))
    brackets = set(chain.from_iterable(bracket_pairs))

    def my_sentencizer(doc):

        is_sent_start = [False] * len(doc)
        if is_sent_start:
            is_sent_start[0] = True

        q_stack = deque()
        for i, token in enumerate(doc):
            if token.text not in quotes or (token.text == "'" and (i-1 > 0 and doc[i-1].text[-1] in ["s", "S"])):
                continue
            t = len(q_stack) - 1
            while t >= 0 and quote_pairs[q_stack[t][0]][0] == "'":
                t -= 1
            if q_stack and (token.text == quote_pairs[q_stack[-1][0]][1] or (t >= 0 and token.text == quote_pairs[q_stack[t][0]][1])):  # close
                if token.text == quote_pairs[q_stack[-1][0]][1]:
                    t = len(q_stack) - 1
                quote_type, open_i = q_stack.pop()
                while len(q_stack) > t:
                    quote_type, open_i = q_stack.pop()
                quote_len = i - open_i - 1
                if quote_len <= 1:  # too short
                    pass
                else:
                    if i-1 > 0 and doc[i-1].is_punct:
                        is_sent_start[open_i] = True
                        is_sent_start[open_i+1] = True
                        is_sent_start[i] = True
                        if i+1 < len(doc):
                            is_sent_start[i+1] = True
                    elif False and i+1 < len(doc) and doc[i+1].is_punct:
                        is_sent_start[open_i] = True
                        if i+2 < len(doc):
                            is_sent_start[i+2] = True
            else:
                for quote_type, quote_pair in enumerate(quote_pairs):
                    if token.text == quote_pair[0]:  # open
                        q_stack.append((quote_type, i))
                        break

            while q_stack and i - q_stack[0][1] > 1000:
                quote_type, open_i = q_stack.popleft()
                if quote_type < 4:
                    is_sent_start[open_i] = True
                    if open_i + 1 < len(doc):
                        is_sent_start[open_i+1] = True

        else:
            while q_stack:
                quote_type, open_i = q_stack.popleft()
                if quote_type < 4:
                    is_sent_start[open_i] = True
                    if open_i + 1 < len(doc):
                        is_sent_start[open_i+1] = True

        start = 0
        seen_eos = False
        for i, token in enumerate(doc):
            if token.text in eos_puncts and (token.text not in [":", ";"] or (i - start >= 10 and all(doc[j].text not in ['(', '[', '{', '"'] for j in range(max(start, i-10), i)))):
                seen_eos = True
            elif seen_eos:
                is_sent_start[start] = True
                start = i
                seen_eos = False

        else:
            is_sent_start[start] = True

        doc = Doc(
            doc.vocab,
            [token.text for token in doc],
            tags=[token.tag_ for token in doc],
            sent_starts=[1 if flag else -1 for flag in is_sent_start],
        )

        return doc

    return my_sentencizer


def concat_paragraphs(lines):
    paragraph = []
    for line in lines:
        if line:
            paragraph.append(line)
        else:
            if paragraph:
                yield " ".join(paragraph)
                paragraph = []
    else:
        yield " ".join(paragraph)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=Path, default=Path("babylm_data", "babylm_10M"))
    args = argparser.parse_args()

    for file_path in filter(Path.is_file, sorted(args.data_path.iterdir())):
        if file_path.suffix.removeprefix(".") not in ["train", "dev", "test"]:
            continue
        data_source = file_path.stem
        try:
            config = data_source_configs[data_source]
        except KeyError:
            logging.warning(f"Unknown data source: {data_source}")
            continue
        else:
            print(f"Processing {file_path}")

        nlp = spacy.load("en_core_web_sm", enable=["tagger"])
        nlp.max_length = int(1e9)
        if config.get("space_tokenized", False):
            nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        else:
            refine_tokenizer(nlp.tokenizer)
        if config.get("line_sentences", False):
            nlp.add_pipe("single_sentencizer")
        else:
            nlp.add_pipe("my_sentencizer")

        with open(file_path, "r") as f:
            lines = list(map(str.strip, f))
        if config.get("concat", False):
            lines = list(concat_paragraphs(lines))
        else:
            lines = list(filter(bool, lines))

        lines = list(map(add_space_around_hyphens, lines))

        with open(file_path.parent/(file_path.name+".txt"), "w") as f:
            for doc in nlp.pipe(tqdm.tqdm(lines)):
                for sent in doc.sents:
                    print(" ".join(map(str, sent)), file=f)