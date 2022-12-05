from argparse import ArgumentParser, Namespace
from typing import Dict, List, Union

import torch
from lightning_lite.utilities.distributed import _distributed_available as distributed_available
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from transformers_framework.adapters.map_adapters.map_adapter import MapAdapter
from transformers_framework.processing import post_processors, pre_processors
from transformers_framework.processing.preprocessing.base import PreProcessor
from transformers_framework.utilities.adapters import clean_parameters, load_many_datasets_together
from transformers_framework.utilities.datamodules import TrainerFn_to_Names
from transformers_framework.utilities.processors import apply_post_processors, apply_pre_processors


class ArrowAdapter(MapAdapter):
    r""" Superclass of Arrow File readers, which implements filtering on scores and limits. """

    def __init__(self, hyperparameters: Namespace, tokenizer: PreTrainedTokenizerBase, stage_name: str, seed: int = 0):
        super().__init__(hyperparameters, tokenizer, stage_name, seed=seed)

        if self.is_active():
            self.data = self.load_data()
            self.pre_processors = self.__get_preprocessors__()
            self.post_processors = self.__get_postprocessors__()

    def __get_preprocessors__(self) -> Union[PreProcessor, List[PreProcessor]]:
        r""" Get list of pre-processors and instantiate corresponding classes. """
        return [
            pre_processors[name](
                self.hyperparameters, self.data.features, self.tokenizer, seed=self.seed, stage_name=self.stage_name
            )
            for name in self.hyperparameters.pre_processors
        ]

    def __get_postprocessors__(self) -> Union[PreProcessor, List[PreProcessor]]:
        r""" Get list of post-processors and instantiate corresponding classes. """
        return [
            post_processors[name](
                self.hyperparameters, self.data.features, self.tokenizer, seed=self.seed, stage_name=self.stage_name
            )
            for name in self.hyperparameters.post_processors
        ]

    def do_preprocessing(self, local_rank: int, global_rank: int):
        r""" Preprocess datasets only on local_rank=0 and using cached values for `local_rank!=0`. """
        rank = global_rank if self.hyperparameters.preprocess_only_on_global_zero else local_rank
        if distributed_available() and rank > 0:
            rank_zero_warn(
                f"Waiting for {'global' if self.hyperparameters.preprocess_only_on_global_zero else 'local'} "
                f"rank 0 to process dataset..."
            )
            torch.distributed.barrier()
        self.data = apply_pre_processors(self.data, self.pre_processors)
        if distributed_available() and rank == 0:
            rank_zero_warn("Waiting for other processes to process dataset or load it from cache files...")
            torch.distributed.barrier()

    def load_data(self):
        r""" Load data from disk first parsing input parameters. This method should be protected by `is_active`. """
        filepaths = self.hyperparameters[f'{self.stage_name}_filepaths']
        splits = self.hyperparameters[f'{self.stage_name}_splits']
        shards = self.hyperparameters[f'{self.stage_name}_shards']

        filepaths, splits, shards = clean_parameters(filepaths, splits, shards)
        rank_zero_warn(f"Loading {self.stage_name} datasets from disk...")

        data = load_many_datasets_together(
            filepaths, splits, shards, keep_in_memory=self.hyperparameters.keep_in_memory
        )
        return data

    def is_active(self) -> bool:
        return self.hyperparameters[f'{self.stage_name}_filepaths'] is not None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        r""" Get dict of data at a given position. """
        sample = self.data[idx]
        if self.hyperparameters.auto_generate_keys is True and ('key' not in sample or sample['key'] is None):
            sample['key'] = idx
        return sample

    def __iter__(self):
        for idx, sample in enumerate(self.data):
            if self.hyperparameters.auto_generate_keys is True and ('key' not in sample or sample['key'] is None):
                sample['key'] = idx
            yield sample

    def preprocess_line(self, sample: Dict) -> Dict:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __getitem__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do tokenization,
        padding and so on.
        """
        return apply_post_processors(sample, self.post_processors)

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(ArrowAdapter, ArrowAdapter).add_argparse_args(parser)
        parser.add_argument('--keep_in_memory', action="store_true", help="Read whole Dataset into memory.")
        parser.add_argument(
            '--preprocess_only_on_global_zero',
            action="store_true",
            help="Preprocess data only on rank 0 node 0 (shared FS).",
        )
        parser.add_argument(
            '--pre_processors',
            type=str,
            default=[],
            nargs='*',
            choices=pre_processors.keys(),
            help="List of pre-processing operations to apply",
        )
        parser.add_argument(
            '--post_processors',
            type=str,
            default=[],
            nargs='*',
            choices=post_processors.keys(),
            help="List of post-processing operations to apply",
        )
        parser.add_argument('--auto_generate_keys', action="store_true", help="Generate sequential integer keys.")

        tmp_hyperparameters, _ = parser.parse_known_args()
        for pre_processor in tmp_hyperparameters.pre_processors:
            pre_processors[pre_processor].add_argparse_args(parser)
        for post_processor in tmp_hyperparameters.post_processors:
            post_processors[post_processor].add_argparse_args(parser)

    @staticmethod
    def add_argparse_stage_args(parser: ArgumentParser, stage_name: str):
        super(ArrowAdapter, ArrowAdapter).add_argparse_stage_args(parser, stage_name=stage_name)
        parser.add_argument(
            f'--{stage_name}_filepaths',
            type=str,
            required=False,
            default=None,
            nargs='+',
            help=f"Path to {stage_name} dataset dump",
        )
        parser.add_argument(
            f'--{stage_name}_splits',
            type=str,
            required=False,
            default=[stage_name],
            nargs='+',
            help="The dataset split to load.",
        )
        parser.add_argument(
            f'--{stage_name}_shards',
            type=int,
            required=False,
            default=None,
            nargs='+',
            help="Whether to shard input datasets.",
        )

        tmp_hyperparameters, _ = parser.parse_known_args()
        for stage_name in TrainerFn_to_Names.values():
            for pre_processor in tmp_hyperparameters.pre_processors:
                pre_processors[pre_processor].add_argparse_stage_args(parser, stage_name=stage_name)
            for post_processor in tmp_hyperparameters.post_processors:
                post_processors[post_processor].add_argparse_stage_args(parser, stage_name=stage_name)
