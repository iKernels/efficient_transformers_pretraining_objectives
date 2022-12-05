import logging
from argparse import ArgumentParser, Namespace

import datasets

from process_datasets.loaders.loader import Loader


ALL_DATASET_NAMES = datasets.list_datasets()


class DatasetLoader(Loader):
    r""" Load a dataset from the datasets library. """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        if len(hparams.name) > 1:
            hparams.config, hparams.name = hparams.name
        else:
            hparams.config = None
            hparams.name = hparams.name[0]

        assert hparams.name in ALL_DATASET_NAMES, (
            f"dataset {hparams.name} is not available. Use `--help` to see all available datasets"
        )

        logging.info(f"Loading input dataset {hparams.name} with config {hparams.config}")
        self.dataset = datasets.load_dataset(hparams.name, hparams.config, keep_in_memory=hparams.keep_in_memory)

    def add_loader_specific_args(parser: ArgumentParser):
        super(DatasetLoader, DatasetLoader).add_loader_specific_args(parser)
        parser.add_argument('--name', type=str, required=True, nargs='+')
