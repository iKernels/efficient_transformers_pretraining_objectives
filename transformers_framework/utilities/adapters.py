import os
from json import JSONDecodeError
from typing import List, Optional, Tuple

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from pytorch_lightning import _logger as logger
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


def clean_parameters(
    filepaths: List[str], splits: List[str], shards: Optional[List[int]]
) -> Tuple[List[str], List[str], List[int]]:
    r""" Clean adapter main parameters. """
    assert all(isinstance(f, str) for f in filepaths), (
        f"Provided filepaths are not all strings: {filepaths}"
    )
    assert len(splits) == 1 or len(splits) == len(filepaths), (
        "The number of splits provided is not 1 or equal to the number of filepaths"
    )
    assert shards is None or len(shards) == 1 or len(shards) == len(filepaths), (
        "The number of shards provided is not None, 1 or equal to the number of filepaths"
    )

    if len(splits) == 1 and len(filepaths) > 1:
        splits = splits * len(filepaths)

    if shards is None:
        shards = [shards]

    if len(shards) == 1 and len(filepaths) > 1:
        shards = shards * len(filepaths)

    assert len(splits) == len(filepaths) == len(shards), (
        "You must provide a single split for every dataset or a split for every dataset"
    )
    assert all(os.path.isdir(filepath) for filepath in filepaths), (
        "all datasets must be valid folders on disk"
    )

    return filepaths, splits, shards


def load_dataset_from_disk(path: str, keep_in_memory: bool = False, split: str = None) -> DatasetDict:
    r""" Load both Dataset's dumps and json folders transparently from disk. """
    try:
        res = load_from_disk(path, keep_in_memory=keep_in_memory)
        if split is not None and split != "-":
            if split not in res:
                raise ValueError(f"Dataset split {split} not in allowed splits {res.keys()}")
            res = res[split]
    except FileNotFoundError:
        try:
            res = load_dataset('json', data_dir=path, keep_in_memory=keep_in_memory)['train']
            if split is not None and split != "-":
                rank_zero_warn(
                    "Jsonl dataset does not require a split, just use `--splits -`."
                    " For this run I will set `--splits -` for you."
                )
        except JSONDecodeError as exception:
            logger.error(
                f"Could not load dataset from {path}. "
                f"Make sure this path is a valid folder containing jsonl files or a dataset dump."
            )
            raise exception
    return res


def load_many_datasets_together(
    filepaths: List[str], splits: List[str], shards: List[int], keep_in_memory: bool = False
):
    r""" Load many datasets sequentially, possibly shard them and eventually concatenate them. """
    data = [
        load_dataset_from_disk(filepath, keep_in_memory=keep_in_memory, split=split)
        for split, filepath in zip(splits, filepaths)
    ]

    if any(s is not None for s in shards):
        rank_zero_warn("Sharding datasets...")
        data = [
            d.shard(s, 0, writer_batch_size=100000) if s is not None and s > 1 else d for s, d in zip(shards, data)
        ]

    if len(data) > 1:
        rank_zero_warn("Concatenating datasets...")
        data = concatenate_datasets(data)
    else:
        data = data[0]
    
    return data
