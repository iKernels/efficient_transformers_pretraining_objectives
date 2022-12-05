from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from typing import Dict, Generator, List

from datasets import Dataset, Features
from transformers import PreTrainedTokenizerBase

from transformers_framework.processing.processor import Processor
from transformers_framework.utilities.functional import dict2list, list2dict


class PreProcessor(Processor):
    r""" Super class of all pre-processors. Pre-processors work at the dataset level and are applied before training.
    Pre-processors should be used for all the operations like data augmentation, data filtering, splitting and so on.
    While pre-processing, transformations of data may be 1-1, 1-many, many-1 or many-many. So the final dataset size
    may be different from the original.
    Tokenization and sample level operations should be done in post-processors instead.
    """

    REQUIRED_PREVIOUS_PRE_PROCESSORS: List[str] = []

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        super().__init__(hyperparameters, features, tokenizer, seed, stage_name)
        self.check_previous_processors()

    def check_previous_processors(self):
        r""" Check every processor in `REQUIRED_PREVIOUS_PRE_PROCESSORS` is present before this. """
        self_position = self.hyperparameters.pre_processors.index(self.name)
        assert self_position >= 0
        for name in self.REQUIRED_PREVIOUS_PRE_PROCESSORS:
            if name not in self.hyperparameters.pre_processors[:self_position]:
                raise ValueError(f"pre_processor {self.name} requires pre_processor {name} before it")

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        r""" Process dataset, possibly taking advantage of multiprocessing and caching. """

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        r""" Add arguments specific of this pre-processor. """
        super(PreProcessor, PreProcessor).add_argparse_args(parser)
        parser.add_argument(
            '--preprocessing_workers',
            type=int,
            required=False,
            default=cpu_count(),
            help="Number of CPUs for dataset preprocessing.",
        )
        parser.add_argument(
            '--keep_columns', action="store_true", help="Keep old columns when mapping."
        )


class OneToOnePreProcessor(PreProcessor):
    r""" One to one dataset transformation (injective), where each example is mapped in exactly one other example. """

    def __call__(self, dataset: Dataset) -> Dataset:
        r""" Process dataset, possibly taking advantage of multiprocessing and caching. """
        return dataset.map(
            self.mapping,
            num_proc=self.hyperparameters.preprocessing_workers,
            remove_columns=dataset.column_names if not self.hyperparameters.keep_columns else None,
        )

    @abstractmethod
    def mapping(self, sample: Dict) -> Dict:
        r""" Process a single example and return processed result. """


class OneToManyPreProcessor(PreProcessor):
    r""" One to many dataset transformation, where each examples is mapped to one or more other examples. """

    def __call__(self, dataset: Dataset) -> Dataset:
        r""" Receive a batch of examples and return processed results. """
        return dataset.map(
            self.batch_mapping,
            num_proc=self.hyperparameters.preprocessing_workers,
            batch_size=self.hyperparameters.preprocessing_batch_size,
            batched=True,
            remove_columns=(dataset.column_names if not self.hyperparameters.keep_columns else None),
        )

    def batch_mapping(self, batch: Dict[str, List]) -> Dict[str, List]:
        r""" Need to convert dict to list and viceversa to be free to work on single examples. """
        # batch from dict of lists (as when slicing with datasets) to list of small dicts
        batch = dict2list(batch)
        batch = [res for sample in batch for res in self.mapping(sample)]
        return list2dict(batch)

    @abstractmethod
    def mapping(self, sample: Dict) -> Generator[Dict, None, None]:
        r""" Process a single example and yield processed result(s). """

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        r""" Add arguments specific of this transformation. """
        super(OneToManyPreProcessor, OneToManyPreProcessor).add_argparse_args(parser)
        parser.add_argument(
            '--preprocessing_batch_size',
            type=int,
            required=False,
            default=1000,
            help="Batch size for dataset preprocessing.",
        )


class ManyToManyPreProcessor(PreProcessor):
    r""" Many to many dataset transformation, where each examples is mapped to one or more other examples. """

    def __call__(self, dataset: Dataset) -> Dataset:
        r""" Receive a batch of examples and return processed results. """
        return dataset.map(
            self.batch_mapping,
            num_proc=self.hyperparameters.preprocessing_workers,
            batch_size=self.hyperparameters.preprocessing_batch_size,
            batched=True,
            remove_columns=(dataset.column_names if not self.hyperparameters.keep_columns else None),
        )

    def batch_mapping(self, batch: Dict[str, List]) -> Dict[str, List]:
        r""" Need to convert dict to list and viceversa to be free to work on single examples. """
        # batch from dict of lists (as when slicing with datasets) to list of small dicts
        batch = dict2list(batch)
        batch = list(self.mapping(batch))
        return list2dict(batch)

    @abstractmethod
    def mapping(self, batch: List[Dict]) -> Generator[Dict, None, None]:
        r""" Process examples and yield processed result(s). """

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        r""" Add arguments specific of this transformation. """
        super(OneToManyPreProcessor, OneToManyPreProcessor).add_argparse_args(parser)
        parser.add_argument(
            '--preprocessing_batch_size',
            type=int,
            required=False,
            default=1000,
            help="Batch size for dataset preprocessing.",
        )


class FilteringPreProcessor(PreProcessor):
    r""" Transformation in which every example is kept or discarded based on a condition. """

    def __call__(self, dataset: Dataset) -> Dataset:
        r""" Receive a batch of examples and return processed results. """
        return dataset.filter(
            self.filtering,
            num_proc=self.hyperparameters.preprocessing_workers,
        )

    @abstractmethod
    def filtering(self, sample: Dict) -> bool:
        r""" Decide to keep or discard a single example. """
