from typing import Mapping, Iterable, Sequence
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import itertools
import csv
import pandas as pd
import inflect
import re
import json
import datasets


def try_decode(b: bytes, encodings: Iterable[str]):
    for encoding in encodings:
        try:
            return b.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Cannot decode {b}")


def longest_common_prefix(strs: Sequence[str]):
    lcp = 0
    while True:
        c = None
        for s in strs:
            if len(s) <= lcp:
                break
            if c is None:
                c = s[lcp]
            elif s[lcp] != c:
                break
        else:
            lcp += 1
            continue
        break
    return lcp


def process_chimeras(raw_file: Path):
    assert raw_file.suffix != ".tsv", "The file must not have suffix .tsv"
    clean_tsv_file = raw_file.with_suffix(".tsv")
    print(f"Generating the clean tsv file {clean_tsv_file} ...", file=sys.stderr)
    with open(raw_file, "rb") as in_f, \
            open(clean_tsv_file, "w") as out_f:
        for line in in_f:
            decoded_line = try_decode(line, ["utf-8", "iso-8859-1"])
            out_f.write(decoded_line.replace("\r", ""))

    print(f"Reading the DataFrame from the clean tsv file {clean_tsv_file} ...", file=sys.stderr)
    df = pd.read_csv(clean_tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE)
    print(f"Preprocessing the DataFrame...", file=sys.stderr)
    df = df.sort_values("TRIAL", key = lambda col: col.str.split("_").map(lambda t: [int(t[0])] + t[1:]))
    df = df.drop(axis="columns", labels=["IMAGE_PROBE_URL", "RESPONSE", "VARIANCE", "IMAGE_QUALITY", "INFORMATIVENESS_CHIMERA", "INFORMATIVENESS_CHIMERA_A", "INFORMATIVENESS_CHIMERA_B"])
    df = df[df["TRIAL"].str.endswith("_L6_A")]
    print(df)
    print(df.dtypes)
    result_tsv_file = raw_file.with_suffix(".result.tsv")
    print(f"Writing to the result tsv file {result_tsv_file} ...", file=sys.stderr)
    df.to_csv(result_tsv_file, sep="\t", quoting=csv.QUOTE_NONE, index=False)
    print(f"Processing...", file=sys.stderr)
    p = inflect.engine()
    meta_dataset = defaultdict(list)
    for row in df.itertuples(index=False):
        passage: str = row.PASSAGE  # type: ignore
        word: str = row.NONCE  # type: ignore
        word_plural = p.plural(word)  # type: ignore
        word = word.upper()
        word_plural = word_plural.upper()
        chimera: str = row.CHIMERA  # type: ignore
        sentences = passage.split("@@")
        sentences = [sentence.strip() for sentence in sentences]
        pattern = re.compile(rf"\b({word}|{word_plural})\b")
        examples = []
        for sentence in sentences:
            new_sentence = ""
            offsets = []
            last_end = 0
            for match in pattern.finditer(sentence):
                new_sentence += sentence[last_end:match.start()]
                new_start = len(new_sentence)
                new_sentence += chimera
                new_end = len(new_sentence)
                offsets.append((new_start, new_end))
                if match[0] == word:
                    pass
                elif match[0] == word_plural:
                    new_sentence += "s"  # append "-s"
                else:
                    assert False
                last_end = match.end()
            new_sentence += sentence[last_end:]
            example = {
                "sentence": new_sentence,
                "offsets": offsets,
            }
            examples.append(example)
        meta_dataset[chimera].extend(examples)

    return meta_dataset


def process_definition(
        raw_data_path: Path,
        ph_pattern: re.Pattern = re.compile("<nonce>"),
        correct_errors: bool = True,
):
    raw_data = datasets.load_from_disk(str(raw_data_path))
    my_dataset = {}
    cnt = 0
    for row in raw_data:
        assert isinstance(row, dict)
        assert len(row["gpt_examples"]) == len(row["replaced_examples"])
        my_examples = []
        replaced_texts = []
        for example, replaced_example in zip(row["gpt_examples"], row["replaced_examples"]):
            assert isinstance(example, str)
            assert isinstance(replaced_example, str)
            retained_texts_ = []
            last_index = 0
            for match in ph_pattern.finditer(replaced_example):
                retained_texts_.append(replaced_example[last_index:match.start()])
                last_index = match.end()
            else:
                retained_texts_.append(replaced_example[last_index:])
            last_index = 0
            replaced_texts_ = []
            offsets = []
            for retained_text in retained_texts_:
                index = example.index(retained_text, last_index)
                offsets.append((last_index, index))
                replaced_texts_.append(example[last_index:index])
                last_index = index + len(retained_text)
            assert replaced_texts_[0] == ""
            offsets.pop(0)
            replaced_texts_.pop(0)
            assert last_index == len(example)
            replaced_texts.append(replaced_texts_)
            my_example = dict(sentence=example, offsets=offsets)
            my_examples.append(my_example)

        if correct_errors:
            # special cases
            if row["word"] == "capital gains tax":
                # simply search all occurrences
                word_pattern = re.compile(row["word"])
                for my_example in my_examples:
                    my_example["offsets"] = [match.span() for match in word_pattern.finditer(my_example["sentence"])]
            else: # general cases
                unique_replaced_texts = set(itertools.chain.from_iterable(replaced_texts))
                uncased_unique_replaced_texts = sorted(
                    set(map(str.lower, unique_replaced_texts)),
                    key = lambda w: (len(w), w)
                )
                assert uncased_unique_replaced_texts[0] == row["word"].lower(), f"{row['word']}: {uncased_unique_replaced_texts}"
                if len(uncased_unique_replaced_texts) == 1:
                    pass # assume no errors
                elif len(uncased_unique_replaced_texts) == 2:
                    lcp = longest_common_prefix(uncased_unique_replaced_texts)
                    suffix = uncased_unique_replaced_texts[1][lcp:]
                    if suffix.endswith("s"):
                        adjust_suffix_length = 1
                    elif suffix.endswith("d"):
                        assert uncased_unique_replaced_texts[1].endswith("ed")
                        adjust_suffix_length = 2
                    elif suffix.endswith("ly"):
                        adjust_suffix_length = 2
                    else:
                        print(f"{lcp=}")
                        cnt += 1
                        print(f"{cnt=}")
                        print(json.dumps(row, indent=2))
                        print(replaced_texts)
                        assert False
                    # adjust offsets
                    for my_example in my_examples:
                        for i, offset in enumerate(my_example["offsets"]):
                            if my_example["sentence"][offset[0]:offset[1]].lower() == uncased_unique_replaced_texts[1]:
                                offset = (offset[0], offset[1] - adjust_suffix_length)
                            my_example["offsets"][i] = offset
                else:
                    assert False, f"Word used in too many forms: {uncased_unique_replaced_texts}"

        my_definition_example = dict(label="definition", sentence=row["scoring_definition"], offsets=[])
        my_examples.append(my_definition_example)
        if row["word"] in my_dataset:
            print(f'Warning: Duplicated word "{row["word"]}"')
        else:
            my_dataset[row["word"]] = my_examples

    return my_dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["chimeras", "definition"])
    argparser.add_argument("raw_data_path", type=Path)
    args = argparser.parse_args()

    result_dataset = {
        "chimeras": process_chimeras,
        "definition": process_definition,
    }[args.mode](args.raw_data_path)

    print(f"Total: {len(result_dataset)} words")
    result_json_file = args.raw_data_path.with_suffix(".json")
    print(f"Writing to the result json file {result_json_file} ...", file=sys.stderr)
    with open(result_json_file, "w") as f:
        json.dump(result_dataset, f, indent="\t")