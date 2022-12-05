from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from typing import Callable

from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers_lightning.datamodules.super_datamodule import SuperDataModule
from transformers_lightning.datasets.iterable_dataset import TransformersIterableDataset
from transformers_lightning.datasets.map_dataset import TransformersMapDataset

from transformers_framework.adapters import adapters
from transformers_framework.adapters.transformer_adapter import TransformersAdapter
from transformers_framework.utilities.datamodules import TrainerFn_to_Names
from transformers_framework.utilities.functional import collate_flexible_fn


class TransformersDataModule(SuperDataModule):
    r""" TransformersDataModule implements some simple methods to check whether training, validation or
    testing is required.
    """

    def __init__(
        self,
        hyperparameters: Namespace,
        trainer: Trainer,
        collate_fn: Callable = collate_flexible_fn,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        super().__init__(hyperparameters, trainer, collate_fn=collate_fn)
        self.tokenizer = tokenizer
        self.adapter_class = adapters[hyperparameters.adapter]

        self.train_adapter = self.get_adapter(TrainerFn.FITTING)
        self.valid_adapter = self.get_adapter(TrainerFn.VALIDATING)
        self.test_adapter = self.get_adapter(TrainerFn.TESTING)

        # prepare data on local_rank 0 of each node
        self.prepare_data_per_node = True

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: str = None):
        r""" Load datasets only if respective file is defined. """

        if stage is None:
            return

        if stage == TrainerFn.FITTING.value or stage == TrainerFn.VALIDATING.value:
            if self.do_train():
                self.train_adapter.do_preprocessing(self.trainer.local_rank, self.trainer.global_rank)
                self.train_dataset = self.load_dataset(TrainerFn.FITTING)
            if self.do_validation():
                self.valid_adapter.do_preprocessing(self.trainer.local_rank, self.trainer.global_rank)
                self.valid_dataset = self.load_dataset(TrainerFn.VALIDATING)

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                self.test_adapter.do_preprocessing(self.trainer.local_rank, self.trainer.global_rank)
                self.test_dataset = [self.load_dataset(TrainerFn.TESTING)]

    def get_adapter(self, stage: TrainerFn) -> TransformersAdapter:
        r""" Return the adapter to use. """
        return self.adapter_class(
            self.hyperparameters, self.tokenizer, TrainerFn_to_Names[stage], seed=self.trainer.current_epoch,
        )

    def load_dataset(self, stage: TrainerFn = None):
        r""" Load a dataset given the stage name. """
        rank_zero_info(f"Loading {stage.value} dataset...")
        adapter = getattr(self, f"{TrainerFn_to_Names[stage]}_adapter")
        dataset_class = TransformersIterableDataset if self.hyperparameters.iterable else TransformersMapDataset

        # map dataset must be told not to load everything in memory
        kwargs = {}
        if not self.hyperparameters.iterable:
            kwargs = dict(keep_in_memory=False)

        dataset = dataset_class(self.hyperparameters, adapter, self.trainer, **kwargs)
        rank_zero_info(
            f"{stage.value.capitalize()} dataset has length "
            f"{len(dataset) if not self.hyperparameters.iterable else 'inf'}"
        )
        return dataset

    def do_train(self):
        return self.train_adapter.is_active()

    def do_validation(self):
        return self.valid_adapter.is_active()

    def do_test(self):
        return self.test_adapter.is_active()

    def do_predict(self):
        return False

    def default_dataloader(self, dataset: Dataset, batch_size: int, **kwargs):
        r""" Return a dataloader with all usual default parameters. """

        if self.hyperparameters.iterable and kwargs.get('shuffle', False) is True:
            raise ValueError(
                "Found shuffle=True while using IterableDataset"
            )

        if not self.hyperparameters.pin_memory:
            rank_zero_warn("Memory pinning is disabled and this may affect performance.")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hyperparameters.num_workers,
            pin_memory=self.hyperparameters.pin_memory,
            collate_fn=self.collate_fn,
            prefetch_factor=self.hyperparameters.prefetch_factor,
            **kwargs,
        )

    def train_dataloader(self):
        r""" Return the training dataloader. """
        if self.do_train():
            params = dict(shuffle=not self.hyperparameters.iterable)
            return self.default_dataloader(self.train_dataset, self.hyperparameters.train_batch_size, **params)
        return None

    def val_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_validation():
            params = dict(shuffle=False) if not self.hyperparameters.iterable else dict()
            return self.default_dataloader(self.valid_dataset, self.hyperparameters.valid_batch_size, **params)
        return None

    def test_dataloader(self):
        r""" Return the test dataloader. """
        if self.do_test():
            params = dict(shuffle=False) if not self.hyperparameters.iterable else dict()
            return [
                self.default_dataloader(dataset, self.hyperparameters.test_batch_size, **params)
                for dataset in self.test_dataset
            ]
        return None

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        parser.add_argument('--num_workers', type=int, required=False, default=cpu_count())
        parser.add_argument('--pin_memory', action="store_true", help='Whether to use memory pinning.')
        for stage_name in TrainerFn_to_Names.values():
            parser.add_argument(f'--{stage_name}_batch_size', type=int, default=32)

        parser.add_argument('--iterable', action="store_true")
        parser.add_argument(
            '--prefetch_factor', default=2, type=int, required=False, help='Number of examples to prepare in advance.'
        )

        parser.add_argument('--adapter', type=str, required=False, default='arrow', choices=adapters.keys())

        # check args and add additional parameters for each stage
        tmp_hyperparameters, _ = parser.parse_known_args()
        adapters[tmp_hyperparameters.adapter].add_argparse_args(parser)
        for stage_name in TrainerFn_to_Names.values():
            adapters[tmp_hyperparameters.adapter].add_argparse_stage_args(parser, stage_name=stage_name)
