from abc import abstractmethod
from argparse import ArgumentParser, Namespace

from transformers import PreTrainedTokenizerBase
from transformers_lightning.adapters import SuperAdapter


class TransformersAdapter(SuperAdapter):

    def __init__(self, hyperparameters: Namespace, tokenizer: PreTrainedTokenizerBase, stage_name: str, seed: int = 0):
        super().__init__(hyperparameters)

        self.tokenizer = tokenizer
        self.stage_name = stage_name
        self.seed = seed

    @abstractmethod
    def is_active(self) -> bool:
        r""" Return True or False based on whether this adapter could return data or not. """

    def do_preprocessing(self, local_rank: int, global_rank: int):
        r""" Do pre-processing of your dataset here. You may use `local_rank` and `global_rank` to avoid preparing
        the data multiple times on the same machine. """

    @staticmethod
    def add_argparse_stage_args(parser: ArgumentParser, stage_name: str):
        r""" In the case many adapters are used, it could be useful
        to organize the arguments of every adapter using a different prefix.
        Put here all the arguments that are not shared by every adapter instance, for
        example the path of the data on the disk. """
