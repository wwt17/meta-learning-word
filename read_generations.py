from collections.abc import Iterable


input_fields = [
    "ground-truth word",
    "ground-truth prefix",
    "     decoded prefix",
    "ground-truth suffix",
]
output_fields = [
    "greedy outputs",
    "sample with top-p=0.92 outputs",
    "beam search outputs",
]
fields = input_fields + output_fields


def parse_example_with_generations(
        lines: Iterable[str],
        fields=fields,
        cont_prefix_pattern="cont. {}: ".format,
):
    example = {}
    last_lines = []
    for line in lines:
        field, sep, rest = line.partition(":")
        if sep and field in fields:
            last_lines = [rest]
            example[field.lstrip()] = last_lines
        else:
            last_lines.append(line)
    for field, value_lines in example.items():
        if field.endswith("outputs"):
            assert len(value_lines) >= 1
            assert value_lines[0] == "\n"
            value_lines.pop(0)
            value = []
            n_cont = 0
            cont_prefix = cont_prefix_pattern(n_cont)
            while value_lines:
                assert value_lines[0].startswith(cont_prefix)
                cont = value_lines.pop(0).removeprefix(cont_prefix)
                n_cont += 1
                cont_prefix = cont_prefix_pattern(n_cont)
                while value_lines and not value_lines[0].startswith(cont_prefix):
                    cont += value_lines.pop(0)
                cont = cont.removesuffix("\n")
                value.append(cont)
        else:
            value = "".join(value_lines)
            value = value.removeprefix(" ")
            value = value.removesuffix("\n")
        example[field] = value
    return example


def read_generations(file, example_first_line_pattern="Example #{}:\n".format):
    n_example = 0
    example_first_line = example_first_line_pattern(n_example)
    line = file.readline()
    while True:
        while line and line != example_first_line:
            line = file.readline()
        if not line:  # EOF
            break
        line = file.readline()
        n_example += 1
        example_first_line = example_first_line_pattern(n_example)
        lines = []
        while line and line != example_first_line:
            lines.append(line)
            line = file.readline()
        if lines and lines[-1] == "\n":
            lines.pop()
        yield parse_example_with_generations(lines)


def extract_definition_from_generation(gen: str):
    try:
        # Assume the prompt ends with a double-quote, so we expect the generated definition ends at another double-quote.
        quote_index = gen.index('"')
    except ValueError: # Cannot find the double-quote
        pass # use the whole generation
    else:
        gen = gen[:quote_index] # use the quoted content
    # for better formatting
    gen = ' ' + gen.strip()
    return gen


def extract_definition_from_reference_generation(gen: str, nlp):
    gen = gen.strip()  # strip spaces to avoid potential empty sents
    doc = nlp(gen)
    sent = next(doc.sents)
    return str(sent)