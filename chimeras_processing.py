from typing import Mapping
from collections import defaultdict
from pathlib import Path
import argparse
from utils import batchify
import csv
import pandas as pd
import inflect
import re
import json


def merge_every_k_lines(in_file, out_file, k, header=True):
    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        if header:
            line = in_f.readline()
            out_f.write(line)
        for line_batch in batchify(in_f, k, drop_last=False):
            for line in line_batch:
                out_f.write(line.rstrip("\n"))
            out_f.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=Path, default=Path("chimeras.txt"))
    args = argparser.parse_args()
    merged_file = args.file.with_suffix(".tsv")
    merge_every_k_lines(args.file, merged_file, 2, header=True)

    if False:
        with open(merged_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True)
            for row_i, row in enumerate(reader):
                print(f"{row_i}: {'|'.join(row)}")

    df = pd.read_csv(merged_file, delimiter="\t", quoting=csv.QUOTE_NONE)
    df = df.sort_values("TRIAL", key = lambda col: col.str.split("_").map(lambda t: [int(t[0])] + t[1:]))
    df = df.drop(axis="columns", labels=["IMAGE_PROBE_URL", "RESPONSE", "VARIANCE", "IMAGE_QUALITY", "INFORMATIVENESS_CHIMERA", "INFORMATIVENESS_CHIMERA_A", "INFORMATIVENESS_CHIMERA_B"])
    df = df[df["TRIAL"].str.endswith("_L6_A")]
    print(df)
    print(df.dtypes)
    df.to_csv(args.file.with_suffix(".result.tsv"), sep="\t", quoting=csv.QUOTE_NONE, index=False)
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
    with open(args.file.with_suffix(".json"), "w") as f:
        json.dump(meta_dataset, f, indent="\t")