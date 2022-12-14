from argparse import ArgumentParser, Namespace

from datasets import Dataset, DatasetDict


class Loader:

    dataset: Dataset

    def __init__(self, hparams: Namespace):
        self.hparams = hparams

    def __call__(self) -> Dataset:
        r""" Return dataset for input data. """
        if self.hparams.split is not None:
            self.dataset = self.dataset[self.hparams.split]

        if self.hparams.shard is not None:
            if isinstance(self.dataset, DatasetDict):
                self.dataset = DatasetDict({
                    k: d.shard(self.hparams.shard, 0, contiguous=False) for k, d in self.dataset.items()
                })
            else:
                self.dataset = self.dataset.shard(self.hparams.shard, 0, contiguous=False)

        return self.dataset

    def add_loader_specific_args(parser: ArgumentParser):
        parser.add_argument('--keep_in_memory', action="store_true", help="Whether to keep in memory input dataset.")
        parser.add_argument('--split', default=None, required=False, type=str, help="Split to be loaded")
        parser.add_argument('--shard', default=None, required=False, type=int, help="Shard input dataset for debugging")
