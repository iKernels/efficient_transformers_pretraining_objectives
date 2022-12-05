from typing import Dict, List

from transformers import PreTrainedTokenizerBase


def encode_single_or_pair(
    *data: List[str],
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    truncation: str = 'longest_first',
    padding: str = 'max_length',
    return_overflowing_tokens: bool = False,
    return_offsets_mapping: bool = False,
    return_special_tokens_mask: bool = False,
    stride: int = 0,
) -> Dict:
    r""" Encode a first-second pair as
    [CLS] first [SEP] second [SEP]
    such that the total length is equal to `max_sequence_length`.
    """
    tok_args = dict(
        add_special_tokens=True,
        return_attention_mask=None,
        return_token_type_ids=None,
        max_length=max_sequence_length,
        truncation=truncation,
        padding=padding,
        return_overflowing_tokens=return_overflowing_tokens,
        return_offsets_mapping=return_offsets_mapping,
        return_special_tokens_mask=return_special_tokens_mask,
        stride=stride,
    )

    encoded = tokenizer(*data, **tok_args)
    return encoded
