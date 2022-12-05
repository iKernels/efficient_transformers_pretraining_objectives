from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np
from datasets import Features
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.processing.postprocessing.base import PostProcessor
from transformers_framework.utilities.processors import whole_word_tails_mask


class RandomTokenDetectionPostProcessor(PostProcessor):
    r"""
    Prepare tokens inputs/labels for random token substutition modeling.
    We sample a few tokens in each sequence for RTS training (with probability `probability` defaults to 0.15 in
    Bert/RoBERTa). If `whole_word_substitution` is True, either every or no token in a word will be replaced.

    Args:
        `probability`: probability that a token is chosen for replacement, default 0.15
    """

    name = "random_token_detection"
    REQUIRED_PREVIOUS_POST_PROCESSORS = ['parser', 'tokenizer']

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        super().__init__(hyperparameters, features, tokenizer, seed, stage_name)

        if not (0.0 <= hyperparameters.probability <= 1.0):
            raise ValueError(
                f"Argument `probability` must be a float between 0.0 and 1.0, got: {hyperparameters.probability}"
            )

        self.generator = np.random.default_rng(self.get_seed())

    def __call__(self, sample: Dict) -> Dict:
        r""" Process a single sample, for example by tokenizing text or converting lists to tensors. """

        input_ids = np.array(sample.pop('input_ids'))
        replaced_input_ids = input_ids.copy()
        replaced_labels = np.full(input_ids.shape, fill_value=0, dtype=int)

        # We sample a few tokens in each sequence for TD training
        # (with probability probability defaults to 0.15 in ELECTRA)
        probability_matrix = np.full(replaced_input_ids.shape, fill_value=self.hyperparameters.probability, dtype=float)

        # create whole work masking mask -> True if the token starts with ## (following token in composed words)
        if self.hyperparameters.whole_word_substitution:
            word_tails = whole_word_tails_mask(replaced_input_ids, self.tokenizer).astype(bool)
            # with whole word masking probability matrix should average probability over the entire word
            probability_matrix[word_tails] = 0.0

        special_tokens_mask = np.array(
            self.tokenizer.get_special_tokens_mask(replaced_input_ids, already_has_special_tokens=True)
        ).astype(bool)

        probability_matrix[special_tokens_mask] = 0.0
        replaced_labels[special_tokens_mask] = IGNORE_IDX

        substituted_indices = self.generator.binomial(n=1, p=probability_matrix).astype(dtype=bool)

        # with whole word masking, assure all tokens in a word are either all masked or not
        if self.hyperparameters.whole_word_substitution:
            for i in range(1, substituted_indices.shape[-1]):
                substituted_indices[i] = substituted_indices[i] | (substituted_indices[i - 1] & word_tails[i])

        # replace tokens
        new_tokens = self.generator.choice(self.tokenizer.vocab_size, size=len(replaced_input_ids))
        replaced_input_ids[substituted_indices] = new_tokens[substituted_indices]
        replaced_labels[substituted_indices] = 1

        replaced_input_ids = replaced_input_ids.tolist()
        replaced_labels = replaced_labels.tolist()

        return dict(input_ids=replaced_input_ids, labels=replaced_labels, original_input_ids=input_ids, **sample)

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(RandomTokenDetectionPostProcessor, RandomTokenDetectionPostProcessor).add_argparse_args(parser)
        parser.add_argument('--probability', type=float, default=0.15, help="Probability of replacing a token")
        parser.add_argument('--whole_word_substitution', action="store_true", help="Enables whole word replacing")
