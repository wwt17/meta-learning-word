from typing import Optional, Iterable, Callable
import re
from utils import example_str
from text_configs import NEW_TOKEN


format_str_attrs = ["new_word", "prompt", "sep"]


def add_format_arguments(
        argparser,
        new_word=NEW_TOKEN,
        embedding_init="none",
        prompt="",
        sep="",
        prepend=" ",
):
    argparser.add_argument(
        "--new_word", default=new_word,
        help="Replace the meta-learned word with this. "
             "For tokenizers incorporating leading spaces into first tokens of "
             "words, if this is not a special token, this should also have a "
             "leading space."
    )
    argparser.add_argument(
        "--no_new_token", action="store_true",
        help="Do not replace the meta-learned word with the new word."
    )
    argparser.add_argument(
        "--embedding_init", choices=["none", "mean"], default=embedding_init,
        help=r'Initialization method of embeddings of added new tokens. '
             r'"none" for default initialization in '
             r'`model.resize_token_embeddings`, which is a sample from the '
             r'initialization distribution; '
             r'"mean" for mean of all pretrained embeddings.'
    )
    argparser.add_argument(
        "--prompt", default=prompt,
        help="Prompt before examples."
    )
    argparser.add_argument(
        "--sep", default=sep,
        help=r'The separator between examples. '
             r'Use "\n"+sep as the separator for pretrained models.'
    )
    argparser.add_argument(
        "--prepend", default=prepend,
        help="Prepend this string to each example."
    )
    argparser.add_argument(
        "--add_tokens", nargs="*",
        help="Add tokens to the tokenizer."
    )


class InContextFormat:
    def __init__(
            self,
            t: Optional[str],
            sep: Optional[str] = None,
            sep_formatter: Optional[Callable[[int], str]] = None,
            start_with_sep: bool = True,
            prompt: str = "",
            t_study: Optional[str] = None,
            no_study_in_prefix: bool = False,
    ):
        """Format of an In-Context Learning episode.
            The format is:
                {prompt}{sep if start_with_sep else ''}({example}{sep})*
            When prompting to generate the next example given several
            in-context examples, it has the same format as above and will not
            include prepend for the next example, so it fits the tokenizers
            with leading whitespaces.
        Args:
            t: The new word. If None, do not replace the original word.
            sep: The separator between examples.
            start_with_sep: whether to have sep before examples.
            prompt: the prompt (instruction) before all examples.
            t_study: The new word for study examples for embedding generation model.
            study_in_prefix: Not to include study examples in the prefix. Default to False, but should set to True for embedding generation model.
        """
        self.t = t
        self.sep = sep
        self.sep_formatter = sep_formatter
        self.start_with_sep = start_with_sep
        self.prompt = prompt
        self.t_study = t_study
        self.no_study_in_prefix = no_study_in_prefix

    def concat_strs(
            self,
            strs: Iterable[str],
            start_with_sep: Optional[bool] = None,
            end_with_sep: bool = True,
            start_index: int = 0,
    ) -> str:
        if start_with_sep is None:
            start_with_sep = self.start_with_sep
        if self.sep_formatter is not None:
            sep_formatter: Callable[[int], str] = self.sep_formatter
        elif self.sep is not None:
            sep_formatter: Callable[[int], str] = lambda index: self.sep  # type: ignore
        else:
            assert False, "Must have either sep_formatter or sep"
        text = ''
        index = start_index
        for s in strs:
            if index != start_index or start_with_sep:
                text += sep_formatter(index)
            text += s
            index += 1
        if end_with_sep and (index != start_index or start_with_sep):
            text += sep_formatter(index)
        return text

    def concat_examples(
            self,
            examples,
            start_with_sep: Optional[bool] = None,
            end_with_sep: bool = True,
            start_index: int = 0,
    ) -> str:
        return self.concat_strs(
            (example_str(example, t=self.t) for example in examples),
            start_with_sep=start_with_sep,
            end_with_sep=end_with_sep,
            start_index=start_index,
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

    def construct_meta_cls_example(self, item, last_n=1, append_to_prefix=""):
        examples = item["examples"]
        prefix_examples, suffix_examples = (
            (examples, [])
            if last_n == 0 else
            (examples[:-last_n], examples[-last_n:])
        )
        prefix, suffix = self(prefix_examples, suffix_examples)
        if self.no_study_in_prefix:
            prefix = ""
        study = [example_str(example, t=self.t_study) for example in prefix_examples]
        if self.t is None:
            word = item["word"]
            # extract the word form from word sense in the ishiwatari dataset
            m = re.fullmatch(r'(\S+)%(\S+\.\d+)', word)
            if m:
                word = m[1]
            new_word = ' ' + word
        else:
            new_word = self.t
        prefix += append_to_prefix.format(new_word=new_word)
        return {"prefix": prefix, "suffix": suffix, "study": study}