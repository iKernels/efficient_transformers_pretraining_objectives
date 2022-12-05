import logging
from argparse import ArgumentParser, Namespace

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(args: Namespace):
    logger.info("Loading datasets from disk")
    datasets = [load_from_disk(f) for f in args.inputs]

    assert all(isinstance(d, Dataset) for d in datasets) or all(isinstance(d, DatasetDict) for d in datasets), (
        "datasets must be either all `Dataset` or all `DatasetDict`, not a mix of them"
    )

    logger.info("Concatenating datasets")
    if isinstance(datasets[0], DatasetDict):
        dataset = DatasetDict({
            key: concatenate_datasets([d[key] for d in datasets]) for key in datasets[0].keys()
        })
    else:
        dataset = concatenate_datasets(datasets)

    logger.info("Saving resulting dataset to disk")
    dataset.save_to_disk(args.output)


if __name__ == '__main__':
    parser = ArgumentParser("Concatenate datasets")
    parser.add_argument('--inputs', type=str, required=True, nargs='+', help="Input dataset paths to concatenate")
    parser.add_argument('--output', type=str, required=True, help="Output dataset path")
    args = parser.parse_args()
    main(args)
