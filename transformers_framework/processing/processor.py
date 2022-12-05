from abc import ABC
from argparse import ArgumentParser, Namespace

import torch
from datasets import Features
from transformers import PreTrainedTokenizerBase


class Processor(ABC):
    r""" Super class of all data processing strategies. """

    name: str = None

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        self.hyperparameters = hyperparameters
        self.features = features
        self.tokenizer = tokenizer
        self.seed = seed
        self.stage_name = stage_name

    def get_seed(self):
        r""" Return a seed for random generators. """
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        r""" Add arguments specific to this processor. """

    @staticmethod
    def add_argparse_stage_args(parser: ArgumentParser, stage_name: str):
        r""" Add arguments specific to this processor and this stage. """
