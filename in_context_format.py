from typing import Optional, Iterable
from utils import example_str
from text_configs import NEW_TOKEN


def add_in_context_format_arguments(argparser):
    argparser.add_argument(
        "--new_word", default=NEW_TOKEN,
        help="Replace the meta-learned word with this."
    )
    argparser.add_argument(
        "--no_new_token", action="store_true",
        help="Do not replace the meta-learned word with the new word."
    )
    argparser.add_argument(
        "--prompt", default="",
        help="Prompt before examples."
    )
    argparser.add_argument(
        "--sep", default="",
        help=r'Use "\n"+sep as the separator for pretrained models.'
    )
    argparser.add_argument(
        "--prepend", default=" ",
        help="Prepend this string to each example."
    )


class InContextFormat:
    def __init__(
            self,
            t: Optional[str],
            sep: str,
            prepend: str = " ",
            start_with_sep: bool = True,
            prompt: str = "",
    ):
        """Format of an In-Context Learning episode.
            The format is:
                {prompt}{sep if start_with_sep else ''}({prepend}{example}{sep})*
            When prompting to generate the next example given several
            in-context examples, it has the same format as above and will not
            include prepend for the next example, so it fits the tokenizers
            with leading whitespaces.
        Args:
            t: The new word. If None, do not replace the original word.
            sep: The separator between examples.
            prepend: The string prepended to each example.
            start_with_sep: whether to have sep before examples.
            prompt: the prompt (instruction) before all examples.
        """
        self.t = t
        self.sep = sep
        self.prepend = prepend
        self.start_with_sep = start_with_sep
        self.prompt = prompt

    def concat_strs(
            self,
            strs: Iterable[str],
            start_with_sep: Optional[bool] = None,
    ) -> str:
        if start_with_sep is None:
            start_with_sep = self.start_with_sep
        return ((self.sep if start_with_sep else '') +
                ''.join(self.prepend + s + self.sep for s in strs))

    def concat_examples(
            self,
            examples,
            start_with_sep: Optional[bool] = None,
    ) -> str:
        return self.concat_strs(
            (example_str(example, t=self.t) for example in examples),
            start_with_sep=start_with_sep,
        )

    def __call__(self, examples, next_examples=None, start_with_sep=None):
        """If next_examples is None, return the full sequence of examples;
        otherwise return a pair(tuple) of sequences, first one for examples
        and the second one for next_examples.
        """
        s = self.prompt + self.concat_examples(examples, start_with_sep=start_with_sep)
        if next_examples is None:
            return s
        next_s = self.concat_examples(next_examples, start_with_sep=False)
        return s, next_s

    def construct_meta_lm_example(self, item, start_with_sep=None):
        examples = item["examples"]
        return {"examples": self(examples, start_with_sep=start_with_sep)}

    def construct_meta_cls_example(self, item, last_n=1):
        examples = item["examples"]
        prefix, suffix = self(examples[:-last_n], examples[-last_n:])
        return {"prefix": prefix, "suffix": suffix}