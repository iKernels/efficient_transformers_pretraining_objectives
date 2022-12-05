from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np
from datasets import Features
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.processing.postprocessing.base import PostProcessor
from transformers_framework.utilities.processors import whole_word_tails_mask


class MaskedLanguageModelingPostProcessor(PostProcessor):
    r"""
    Prepare masked tokens inputs/labels for masked language modeling: 80% masked, 10% randomly replaced, 10% original.
    If `whole_word_masking` is True, either every or no token in a word will be masked.

    Args:
        `probability`: probability that a token is chosen for replacement, masking or leaving unchanged, default 0.15
        `probability_masked`: sub-probability that a chosen token is masked, default 0.80
        `probability_replaced`: probability that a chosen token is replaced, default 0.10
        `probability_unchanged`: probability that a chosen token is left unchanged, default 0.10
        ``
    """

    name = "masked_language_modeling"
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
        if not (0.0 <= hyperparameters.probability_masked <= 1.0):
            raise ValueError(
                f"Argument `probability_masked` must be a float between 0.0 and 1.0, "
                f"got: {hyperparameters.probability_masked}"
            )
        if not (0.0 <= hyperparameters.probability_replaced <= 1.0):
            raise ValueError(
                f"Argument `probability_replaced` must be a float between 0.0 and 1.0, "
                f"got: {hyperparameters.probability_replaced}"
            )
        if not (0.0 <= hyperparameters.probability_unchanged <= 1.0):
            raise ValueError(
                f"Argument `probability_unchanged` must be a float between 0.0 and 1.0, "
                f"got: {hyperparameters.probability_unchanged}"
            )

        if (
            hyperparameters.probability_masked
            + hyperparameters.probability_replaced
            + hyperparameters.probability_unchanged
        ) != 1:
            raise ValueError(
                "The sum of `probability_masked`, `probability_replaced` and `probability_unchanged` must be 1.0"
            )

        if self.hyperparameters.whole_word_masking is True and self.hyperparameters.return_word_ids is False:
            raise ValueError("add `--return_word_ids` to use `--whole_word_masking`")

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
            )

        self.generator = np.random.default_rng(self.get_seed())

    def __call__(self, sample: Dict) -> Dict:
        r""" Process a single sample, for example by tokenizing text or converting lists to tensors. """

        input_ids = np.array(sample['input_ids'])
        masked_input_ids = input_ids.copy()
        mlm_labels = input_ids.copy()

        # We sample a few tokens in each sequence for masked-LM training
        # (with probability probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = np.full(masked_input_ids.shape, fill_value=self.hyperparameters.probability, dtype=float)

        # create whole work masking mask -> True if the token starts with ## (following token in composed words)
        if self.hyperparameters.whole_word_masking:
            word_tails = whole_word_tails_mask(
                masked_input_ids, self.tokenizer, word_ids=sample['word_ids']
            ).astype(bool)
            # with whole word masking probability matrix should average probability over the entire word
            probability_matrix[word_tails] = 0.0

        special_tokens_mask = np.array(
            self.tokenizer.get_special_tokens_mask(masked_input_ids, already_has_special_tokens=True)
        ).astype(bool)

        probability_matrix[special_tokens_mask] = 0.0

        # draw mask indices from bernoulli distribution
        masked_indices = self.generator.binomial(n=1, p=probability_matrix).astype(dtype=bool)

        # with whole word masking, assure all tokens in a word are either all masked or not
        if self.hyperparameters.whole_word_masking:
            for i in range(1, len(masked_indices)):
                masked_indices[i] = masked_indices[i] | (masked_indices[i - 1] & word_tails[i])

        mlm_labels[~masked_indices] = IGNORE_IDX    # We only compute loss on masked tokens

        # create array from which we will sample at once whether tokens are masked, replaced or left unchanged
        positions = self.generator.choice(3, size=len(masked_input_ids), p=[
            self.hyperparameters.probability_masked,
            self.hyperparameters.probability_replaced,
            self.hyperparameters.probability_unchanged,
        ])

        indexes_to_mask = (positions == 0) & masked_indices
        indexes_to_replace = (positions == 1) & masked_indices
        # indexes_to_leave_unchanged = (positions == 2) & masked_indices

        # mask tokens
        masked_input_ids[indexes_to_mask] = self.tokenizer.mask_token_id

        # replace tokens
        new_tokens = self.generator.choice(self.tokenizer.vocab_size, size=len(masked_input_ids))
        masked_input_ids[indexes_to_replace] = new_tokens[indexes_to_replace]

        return dict(
            masked_input_ids=masked_input_ids.tolist(),
            masked_labels=mlm_labels.tolist(),
            original_input_ids=input_ids,
            **sample
        )

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(MaskedLanguageModelingPostProcessor, MaskedLanguageModelingPostProcessor).add_argparse_args(parser)
        parser.add_argument('--probability', type=float, default=0.15, help="Probability of changing a token")
        parser.add_argument(
            '--probability_masked', type=float, default=0.80, help="Sub-probability of masking a token when selected"
        )
        parser.add_argument(
            '--probability_replaced',
            type=float,
            default=0.10,
            help="Sub-probability of replacing a token when selected"
        )
        parser.add_argument(
            '--probability_unchanged',
            type=float,
            default=0.10,
            help="Sub-probability of leaving unchanged a token when selected"
        )
        parser.add_argument('--whole_word_masking', action="store_true", help="Enables whole word masking")
