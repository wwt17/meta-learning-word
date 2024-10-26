from typing import Mapping
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import csv
import pandas as pd
import inflect
import re
import json


def try_decode(b: bytes, encodings):
    for encoding in encodings:
        try:
            return b.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Cannot decode {b}")


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
    result_json_file = raw_file.with_suffix(".json")
    print(f"Writing to the result json file {result_json_file} ...", file=sys.stderr)
    with open(result_json_file, "w") as f:
        json.dump(meta_dataset, f, indent="\t")


def process_definition(raw_data_path: Path):
    pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=["chimeras", "definition"])
    argparser.add_argument("raw_data_path", type=Path)
    args = argparser.parse_args()

    {
        "chimeras": process_chimeras,
        "definition": process_definition,
    }[args.mode](args.raw_data_path)