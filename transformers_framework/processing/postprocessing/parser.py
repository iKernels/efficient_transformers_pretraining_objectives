from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

from datasets import Features
from transformers import PreTrainedTokenizerBase

from transformers_framework.processing.postprocessing.base import PostProcessor
from transformers_framework.utilities.processors import (
    check_descriptor_compatibility_with_dataset,
    expand_and_concatenate_input_fields,
)


class ParserPostProcessor(PostProcessor):
    r""" Just process inputs to a standard format:
    {
        'text': Union[List[str], str],
        'label': Union[List[int], int] = None,
        'key': Union[List[int], int] = None,
    }

    This should be the first post_processor in you run.
    """

    name = 'parser'

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        super().__init__(hyperparameters, features, tokenizer, seed, stage_name)

        check_descriptor_compatibility_with_dataset(self.hyperparameters.data_columns, features)

        if len(hyperparameters.data_columns) > 2:
            raise ValueError("`--data_columns` must contain 1 or 2 arguments")

        if self.hyperparameters.key_column is not None:
            if (self.hyperparameters.key_column not in features) and (
                features[self.hyperparameters.key_column].dtype != "int64"
            ):
                raise ValueError(
                    f"Key column `{self.hyperparameters.key_column}` must be a column contining int64 values"
                )

        if self.hyperparameters.label_column is not None:
            if self.hyperparameters.label_column not in features:
                raise ValueError(f"Labels column `{self.hyperparameters.label_column}` not found in dataset")

        if self.hyperparameters.prefixes is not None:
            if len(self.hyperparameters.prefixes) != len(self.hyperparameters.data_columns):
                raise ValueError(
                    "`prefixes` must have same lengths of `data_columns`, "
                    "use '' for not adding a prefix to some column"
                )
        else:
            self.hyperparameters.prefixes = [''] * len(self.hyperparameters.data_columns)

        if self.hyperparameters.postfixes is not None:
            if len(self.hyperparameters.postfixes) != len(self.hyperparameters.data_columns):
                raise ValueError(
                    "`postfixes` must have same lengths of `data_columns`, "
                    "use '' for not adding a postfix to some column"
                )
        else:
            self.hyperparameters.postfixes = [''] * len(self.hyperparameters.data_columns)

    def process_input_fields(self, sample: Dict) -> Tuple[List[str], List[List[int]]]:
        r""" Process input fields as described by `data_columns`.
        Returns composed fields and positions indicating the first characters of the
        original parts in the composed one.
        """
        data, positions = [], []
        for prefix, field, postfix in zip(
            self.hyperparameters.prefixes,
            self.hyperparameters.data_columns,
            self.hyperparameters.postfixes,
        ):
            res, position = expand_and_concatenate_input_fields(
                sample=sample,
                descriptor=field,
                concat_character=self.hyperparameters.concat_character,
                prefix=prefix,
                postfix=postfix,
            )
            if res is not None and positions is not None:
                data += res
                positions += position
        return data, positions

    def __call__(self, sample: Dict) -> Dict:
        r""" Parse input examples to extract and pre-process important features. """

        input_data, _ = self.process_input_fields(sample)

        res = dict(text=input_data)

        if sample.get(self.hyperparameters.label_column, None) is not None:
            res['labels'] = sample[self.hyperparameters.label_column]
        if sample.get(self.hyperparameters.key_column, None) is not None:
            res['keys'] = sample[self.hyperparameters.key_column]

        return res

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(ParserPostProcessor, ParserPostProcessor).add_argparse_args(parser)
        parser.add_argument(
            '--data_columns',
            required=True,
            nargs='+',
            help=(
                "Names of the fields of the input data to use for training. "
                "Use `:` to concatenate fields together and `*` to flatten lists."
            )
        )
        parser.add_argument('--prefixes', type=str, default=None, required=False, nargs='+')
        parser.add_argument('--postfixes', type=str, default=None, required=False, nargs='+')
        parser.add_argument('--key_column', required=False, default=None, help="Name of the key field")
        parser.add_argument('--label_column', required=False, default=None, help="Name of the label field")
        parser.add_argument(
            '--concat_character',
            default=" ",
            type=str,
            required=False,
            help="Char or string to use when concatenating input fields.",
        )
