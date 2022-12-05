from argparse import ArgumentParser, Namespace
from typing import Dict

from datasets import Features
from transformers import PreTrainedTokenizerBase

from transformers_framework.processing.postprocessing.base import PostProcessor
from transformers_framework.utilities.tokenization import encode_single_or_pair


class TokenizerPostProcessor(PostProcessor):

    name = "tokenizer"
    REQUIRED_PREVIOUS_POST_PROCESSORS = ['parser']

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        super().__init__(hyperparameters, features, tokenizer, seed, stage_name)

    def __call__(self, sample: Dict) -> Dict:
        r""" Generinc encoder. Encodes every input field separately and then possibly chains them.
        If the inputs are 2 or less, will use normal transformers encoding of single sentences or pairs.
        """

        input_data = sample['text']  # expects parser as previous post_processor

        # common tokenization arguments
        tok_args = dict(
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            truncation=self.hyperparameters.truncation,
            padding=self.hyperparameters.padding,
            return_special_tokens_mask=self.hyperparameters.return_special_tokens_mask,
        )

        encoded = encode_single_or_pair(*input_data, **tok_args)

        if 'labels' in sample:
            encoded['labels'] = sample['labels']
        if 'keys' in sample:
            encoded['keys'] = sample['keys']

        # try to put word_ids as dict attribute
        if 'word_ids' not in encoded:
            try:
                encoded['word_ids'] = encoded.word_ids()
            except Exception:
                pass

        if not self.hyperparameters.return_word_ids and 'word_ids' in encoded:
            del encoded['word_ids']

        return encoded

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(TokenizerPostProcessor, TokenizerPostProcessor).add_argparse_args(parser)
        parser.add_argument(
            '--max_sequence_length', type=int, required=True, help="Input max sequence length"
        )
        parser.add_argument(
            '--padding',
            type=str,
            default="max_length",
            choices=('longest', 'max_length', 'do_not_pad'),
            help="Padding technique, more details in `transformers` library",
        )
        parser.add_argument(
            '--truncation',
            type=str,
            default="longest_first",
            choices=('longest', 'only_first', 'only_second', 'do_not_truncate'),
            help="Truncation technique, more details in `transformers` library",
        )
        parser.add_argument(
            '--return_word_ids', action="store_true", help="Return words ids."
        )
        parser.add_argument(
            '--return_special_tokens_mask', action="store_true", help="Return special tokens mask."
        )
