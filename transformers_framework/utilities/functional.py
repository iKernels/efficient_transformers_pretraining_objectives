import hashlib
from functools import partial
from itertools import islice
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor


def bool_to_int(string: str) -> int:
    return int(string.lower().strip() in ('yes', 'pos', 'positive', '1', 'correct', 'true'))


def to_int(string: str) -> int:
    return int(string.lower().strip())


def to_float(string: str) -> float:
    return float(string.strip())


def is_whitespace(c: str) -> bool:
    return (c == " ") or (c == "\t") or (c == "\r") or (c == "\n") or (ord(c) == 0x202F)


def dict2list(data: Dict[Any, Iterable]) -> Iterable[Dict]:
    r""" Convert a dict of lists to a list of dicts. """

    # get all the data and assert each value is list
    values = list(data.values())
    assert all(isinstance(v, Iterable) for v in values)

    # assert each value has same length to be able to create list of small dicts
    assert all(len(v) == len(values[0]) for v in values)

    if not data or any(len(v) == 0 for v in values):
        return []

    # create output dictionary using the same keys for all entries
    keys = data.keys()
    res = [dict(zip(keys, values)) for values in zip(*[data[key] for key in keys])]
    return res


def list2dict(data: Iterable[Dict]) -> Dict[Any, Iterable]:
    r""" Convert a list of dicts to a dict of lists. """

    data = list(data)

    if not data:
        return {}

    # check all instances in the input list are dicts
    assert all(isinstance(d, Mapping) for d in data)

    # check all input dicts have the same keys
    keys = data[0].keys()
    assert all(d.keys() == keys for d in data)

    # merge data
    res = {k: [d[k] for d in data] for k in keys}
    return res


def do_overlap(a: Tuple[int], b: Tuple[int]) -> bool:
    r""" Check whether 2 intervals overlaps. """
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def _check_types(argument: str, types=[]) -> Any:
    r""" Parse argument in one of the given types (in order) and return converted value. """
    for _type in types:
        try:
            if _type is bool:
                if argument.lower() not in ('true', 'false'):
                    raise ValueError()
                x = (argument.lower() == 'true')
            else:
                x = _type(argument)
            return x
        except ValueError:
            pass
    raise TypeError(f"Argument {argument} is not of allowed types: {types}")


def check_types(*types):
    r""" Parse argument in one of the given types (in order) and return converted value. """
    return partial(_check_types, types=types)


def split(_list: Iterable, part_length: int) -> Iterable:
    r"""
    Split an Iterable `_list` in parts of length `part_length`.
    Eventually drop last piece if it would have been shorter. """
    assert isinstance(part_length, int) and part_length > 0
    assert isinstance(_list, Iterable)

    item = list(islice(_list, part_length))
    while item:
        yield item
        item = list(islice(_list, part_length))


def l2_norm(x, y, dim: int = -1, keepdim: bool = False, normalize: bool = True):  # noqa: E741
    r""" Computes L-Norm between two tensors on the given dimension. """
    if normalize:
        x = x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
        y = y / torch.linalg.norm(y, ord=2, dim=dim, keepdim=True)

    return (x - y).pow(2).sum(dim=dim, keepdim=keepdim).sqrt()


def expand_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    logits = torch.stack([1 - probs, probs], dim=-1).log()
    return logits


def expand_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.stack([1 - probabilities, probabilities], dim=-1)


def get_rng_index(list_or_tuple) -> int:
    return torch.randint(0, len(list_or_tuple), size=()).item()


def shrink_batch(
    input_ids: torch.Tensor, *args: torch.Tensor, pad_token_id: int = 0, shrink_to_multiples_of: int = None
):
    r""" Remove data on the sequence length dimension in the positions where every example is padded. """
    indexes = (input_ids != pad_token_id).any(dim=0)

    if shrink_to_multiples_of is not None:
        original_indexes_shape = indexes.shape
        indexes = indexes.view(-1, shrink_to_multiples_of).any(dim=-1, keepdim=True)
        indexes = indexes.expand((-1, shrink_to_multiples_of)).reshape(original_indexes_shape)

    if not len(args):
        return input_ids[slice(None), indexes]
    return (
        input_ids[slice(None), indexes],
        *[tensor[slice(None), indexes] if tensor is not None else None for tensor in args],
    )


def string_to_signature(string, length: int = 16):
    return hashlib.sha1(string.encode("utf-8"), usedforsecurity=False).hexdigest()[:length]


def sample_from_distribution(logits: Tensor, sample_function: str = 'gumbel'):
    r"""
    Sample from generator logits either using gumbel distrib or multinomial distribution.
    Reimplement gumbel softmax because there is a bug in torch.nn.functional.gumbel_softmax
    when fp16 is used (https://github.com/pytorch/pytorch/issues/41663).
    Code taken from
    https://github.com/richarddwang/electra_pytorch/blob/9b2533e62cd1b6126feca323fb7b48480b8c2df0/pretrain.py#L318.
    Gumbel softmax is equal to what official ELECTRA code do,
    standard gumbel dist. = -ln(-ln(standard uniform dist.))
    """
    if sample_function == 'gumbel':
        loc = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        scale = torch.tensor(1., device=logits.device, dtype=logits.dtype)
        gumbel_dist = torch.distributions.gumbel.Gumbel(loc, scale)
        return (logits + gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif sample_function == 'multinomial':
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze()
    else:
        raise ValueError("`sample_function` not valid, choose between 'gumbel' and 'multinomial'")


def index_multi_tensors(*tensors: Sequence[Tensor], positions: Tensor = None, flatten: bool = False):
    r""" Index many tensors where positions is True and eventually flatten results. """
    return (ten[positions].flatten() if flatten else ten[positions] for ten in tensors)


def multi_get_from_dict(dictionary, *keys, default: Any = None):
    r""" Get many keys from dictionary without having to index multiple times.
    Returns `default` is key is not found.
    """

    return (dictionary.get(k, default) for k in keys)


def collate_flexible_fn(data: List[Dict]) -> Dict[str, torch.Tensor]:
    r"""
    Merge n dicts with identical keys creating list of value tensors. If elements are not tensorable,
    they will be left in the original state.
    """
    res = concat_dict_values(data)

    # convert values to tensors if possible
    for k in res.keys():
        try:
            res[k] = torch.tensor(res[k])
        except (ValueError, RuntimeError):
            pass
    return res


def concat_dict_values(data: List[Dict]) -> Dict[str, List]:
    r"""
    Given a list of dictionaries with the same keys, return a dictionary in which
    each value is the contatenation of the values in the original dictionaries.
    """
    if not data:
        return dict()

    assert all(a.keys() == data[0].keys() for a in data), "all examples used to create a batch must have the same keys"

    res = {}
    for dictionary in data:
        for key in dictionary.keys():
            if key in res:
                res[key].append(dictionary[key])
            else:
                res[key] = [dictionary[key]]
    return res
