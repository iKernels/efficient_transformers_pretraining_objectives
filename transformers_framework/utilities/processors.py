import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import PreTrainedTokenizerBase
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from transformers_framework.processing.processor import Processor


logger = logging.getLogger('transformers_framework')


def apply_pre_processors(dataset: Dataset, pre_processors: List):
    r""" Apply many pre-processors to a dataset in given order. """
    for pre_processor in pre_processors:
        dataset = pre_processor(dataset)
    return dataset


def apply_post_processors(sample: Dict, post_processors: List) -> Dict:
    r""" Apply many post-processors to a sample in given order. """
    for post_processor in post_processors:
        sample = post_processor(sample)
    return sample


def expand_and_concatenate_input_fields(
    sample: Dict, descriptor: str, concat_character: str = "", prefix: str = "", postfix: str = "",
) -> Tuple[List[str], List[List[int]]]:
    r""" Given a descriptor in the following grammar, create the corresponding output string.
    The input descriptor should be compatible with the following grammar
    (pardon the mix of production rules and regex):
    
    S = P | P\?
    P -> +\w+ | G
    G -> G:G
    G -> \w+
    G -> *\w+

    * is used to concatenate the content of a list (which must be a list of strings)
    + is used to return the entire list
    : is used to concatenate two strings together

    Example:
    sample = {
        'question': 'This is a question',
        'candidates': ['candidate 1', 'candidate 2']
    }

    With "question:*candidates" (and `concat_character` equal to space) the output will be
    "This is a question candidate 1 candidate 2"

    Along with the resulting string, this method will return a sequence of integers which point to the start
    character of each field that was considered.
    """

    # if descriptor ends with ?, the field may be null and will be ignored for this example
    allow_null = False
    if descriptor.endswith("?"):
        allow_null = True
        descriptor = descriptor[:-1]

    # if descriptor starts with +, the corresponding array will be returned
    if descriptor.startswith("+"):
        data = sample[descriptor[1:]]  # should be an array of strings
        if prefix or postfix:
            data = [prefix + d + postfix for d in data]

        positions = [len(d) for d in data]  # should be an array of integers
        if data is None and not allow_null:
            raise ValueError(f"field {descriptor[1:]} is None in example {sample}")

        return data, positions

    # finally, create field by concatenating over : and *
    first = True
    data, positions = "", []

    for field in descriptor.split(':'):
        for part in (sample[field[1:]] if field.startswith("*") else [sample[field]]):
            if part is None:
                if allow_null:
                    continue
                else:
                    raise ValueError(f"field {field} contains None in example {sample}")

            if not first:
                data += concat_character
            else:
                first = False
                part = prefix + part

            positions.append(len(data))
            data += part

    if data and positions:
        return [data + postfix], positions
    else:
        return None, None


def check_feature_is_sequence_of_strings(feature: Features) -> bool:
    r""" Check that a feature is of type `Sequence(feature=Value(dtype='string'))` """
    return isinstance(feature, Sequence) and isinstance(feature.feature, Value) and feature.feature.dtype == 'string'


def check_feature_string(feature: Features) -> bool:
    r""" Check that a feature is of type `Value(dtype='string')` """
    return isinstance(feature, Value) and feature.dtype == 'string'


def check_descriptor_compatibility_with_dataset(list_of_descriptors: List[str], features: Features):
    for field in list_of_descriptors:
        if field.endswith('?'):
            field = field[:-1]
        if field.startswith('+'):
            if not check_feature_is_sequence_of_strings(features[field[1:]]):
                raise ValueError(
                    f"field `{field[1:]}` of dataset must be Sequence of strings "
                    f"(`Sequence(feature=Value(dtype='string'))`)"
                )
        else:
            for part in field.split(":"):
                if part.startswith('*'):
                    if not check_feature_is_sequence_of_strings(features[part[1:]]):
                        raise ValueError(
                            f"field `{part[1:]}` of dataset must be Sequence of strings "
                            f"(`Sequence(feature=Value(dtype='string'))`)"
                        )
                else:
                    if not check_feature_string(features[part]):
                        raise ValueError(f"field `{part}` of dataset must be a string (`Value(dtype='string')`)")


def whole_word_tails_mask(
    inputs: Union[List, np.ndarray],
    tokenizer: PreTrainedTokenizerBase,
    word_ids: List[int] = None,
    special_tokens_mask: List[bool] = None,
) -> Union[List, np.ndarray]:
    r"""
    Create whole work masking mask -> 1 if the token starts with ## (token is not first in composed word), 0 otherwise.
    """

    if isinstance(tokenizer, PreTrainedTokenizerFast) and word_ids is None:
        raise ValueError("You are using a fast tokenizer but `word_ids` is None, check call of this function")
 
    # automatic method for fast tokenizers
    if word_ids is not None:
        res = [False] + [word_ids[i - 1] == word_ids[i] for i in range(1, len(word_ids))]

    # manual method for non fast tokenizers
    elif isinstance(tokenizer, BertTokenizer):
        res = [
            _id.startswith('##') for _id in tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=False)
        ]
    elif isinstance(tokenizer, T5Tokenizer):
        res = [
            not _id.startswith('_') for _id in tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=False)
        ]

    else:
        raise ValueError(
            f"`whole_word_tails_mask` does not support {tokenizer.__class__.__name__} tokenizers. "
            f"Open an issue to ask for the implementation for other tokenizer types."
        )

    if special_tokens_mask is None:
        special_tokens_mask = tokenizer.get_special_tokens_mask(inputs, already_has_special_tokens=True)

    # put to none on special tokens positions
    res = [None if special_tokens_mask[i] else res[i] for i in range(len(res))]

    if isinstance(inputs, np.ndarray):
        res = np.array(res)
    
    return res


def create_dict_from_processors(*processors: List[Processor]) -> Dict[str, Processor]:
    return {p.name: p for p in processors}
