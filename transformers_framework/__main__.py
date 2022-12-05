import os

import datasets
import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
import transformers_lightning
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from transformers_lightning.callbacks import RichProgressBar, TransformersModelCheckpointCallback
from transformers_lightning.defaults import DefaultConfig
from transformers_lightning.loggers.jsonboard_logger import JsonBoardLogger

from transformers_framework import __version__
from transformers_framework.datamodules.transformers_datamodule import TransformersDataModule
from transformers_framework.pipelines import pipelines
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.logging import setup_logging


def main(hyperparameters):

    # too much complains of the tokenizers
    transformers.logging.set_verbosity_error()

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    datasets.config.IN_MEMORY_MAX_SIZE = 1024 * 1024 * 1024 * hyperparameters.datasets_max_in_memory  # in GB

    # setup logging
    setup_logging()

    # Print systems info
    rank_zero_info(
        f"Starting experiment {hyperparameters.name}, with model "
        f"{hyperparameters.model} and pipeline {hyperparameters.pipeline}..."
    )
    rank_zero_info(
        f"Running on\n"
        f"  - transformers_framework={__version__}\n"
        f"  - torch={torch.__version__}\n"
        f"  - transformers={transformers.__version__}\n"
        f"  - pytorch-lightning={pl.__version__}\n"
        f"  - transformers-lightning={transformers_lightning.info.__version__}\n"
        f"  - datasets={datasets.__version__}\n"
        f"  - torchmetrics={torchmetrics.__version__}\n"
    )

    # set the random seed
    seed_everything(seed=hyperparameters.seed, workers=True)

    # instantiate PL model
    model = pl_model_class(hyperparameters)

    # default jsonboard logger
    js_logger = JsonBoardLogger(hyperparameters)
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hyperparameters.output_dir, hyperparameters.tensorboard_dir),
        name=hyperparameters.name,
    )
    loggers = [tb_logger, js_logger]

    # save pre-trained models to
    save_transformers_callback = TransformersModelCheckpointCallback(hyperparameters)

    # and log learning rate
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # and normal checkpoints with
    checkpoints_dir = os.path.join(hyperparameters.output_dir, hyperparameters.checkpoints_dir, hyperparameters.name)
    checkpoint_callback_args = dict(verbose=True, dirpath=checkpoints_dir, save_weights_only=False)

    if hyperparameters.monitor is not None:
        checkpoint_callback_args = dict(
            **checkpoint_callback_args,
            monitor=hyperparameters.monitor,
            save_last=True,
            mode=hyperparameters.monitor_direction,
            save_top_k=1,
        )
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_args)

    # rich progress bar
    rich_progress_bar = RichProgressBar(leave=True)

    # modelsummary callback
    model_summary = RichModelSummary(max_depth=2)

    # all callbacks
    callbacks = [
        save_transformers_callback,
        lr_monitor_callback,
        checkpoint_callback,
        rich_progress_bar,
        model_summary,
    ]

    # early stopping if defined
    if hyperparameters.early_stopping:
        if hyperparameters.monitor is None:
            raise ValueError("cannot use early_stopping without a monitored variable")

        early_stopping_callback = EarlyStopping(
            monitor=hyperparameters.monitor,
            patience=hyperparameters.patience,
            verbose=True,
            mode=hyperparameters.monitor_direction,
        )
        callbacks.append(early_stopping_callback)

    # disable find unused parameters to improve performance
    if hyperparameters.strategy in ("dp", "ddp2"):
        raise ValueError(
            "This repo is not designed to work with DataParallel. Use strategy `ddp` or others instead."
        )

    kwargs = dict(callbacks=callbacks, logger=loggers, default_root_dir=hyperparameters.output_dir)
    if hyperparameters.strategy == "ddp":
        kwargs['strategy'] = DDPStrategy(
            find_unused_parameters=hyperparameters.find_unused_parameters,
            static_graph=hyperparameters.static_graph
        )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(hyperparameters, **kwargs)

    # DataModules
    datamodule = TransformersDataModule(hyperparameters, trainer, tokenizer=model.tokenizer)

    # Train!
    if datamodule.do_train():
        rank_zero_info(f"Training experiment {hyperparameters.name}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=hyperparameters.ckpt_path)

    # Test!
    if datamodule.do_test():
        rank_zero_info(f"Testing experiment {hyperparameters.name}")
        if datamodule.do_train() and hyperparameters.monitor is not None:
            rank_zero_warn(
                f"Going to test on best ckpt chosen over "
                f"{hyperparameters.monitor}: {checkpoint_callback.best_model_path}"
            )
            trainer.test(datamodule=datamodule, ckpt_path='best')
        else:
            rank_zero_warn("Going to test on last or pretrained ckpt")
            trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':

    # Read config for defaults and eventually override with hyper-parameters from command line
    parser = FlexibleArgumentParser(
        prog="Transforers Framework", description="Flexible experiments with Transformers", add_help=True
    )

    parser.add_argument('--version', action="store_true")
    tmp_params, _ = parser.parse_known_args()
    if tmp_params.version is True:
        print(f"Transformers-Framework version: {__version__}")
        exit(0)

    # model class name
    parser.add_argument('--pipeline', type=str, required=True, choices=pipelines.keys())

    # retrieving model with temporary parsed arguments
    tmp_params, extra = parser.parse_known_args()
    parser.add_argument('--model', type=str, required=True, choices=pipelines[tmp_params.pipeline].keys())

    # experiment name, used both for checkpointing, pre_trained_names, logging and tensorboard
    parser.add_argument('--name', type=str, required=True, help='Name of the model')

    # various options
    parser.add_argument('--seed', type=int, default=1337, help='Set the random seed')
    parser.add_argument('--monitor', type=str, help='Value to monitor for best checkpoint', default=None)
    parser.add_argument(
        '--monitor_direction', type=str, help='Monitor value direction for best', default='max', choices=['min', 'max']
    )
    parser.add_argument('--early_stopping', action="store_true", help="Use early stopping")
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        required=False,
        help="Number of non-improving validations to wait before early stopping"
    )
    parser.add_argument('--ckpt_path', type=str, default=None, help="Restore from checkpoint.")
    parser.add_argument(
        '--find_unused_parameters',
        action="store_true",
        help="Whether to check for unused params at each iteration"
    )
    parser.add_argument(
        '--static_graph', action="store_true", help="Improve speed if model graph is unchanged in every iteration."
    )
    parser.add_argument(
        '--datasets_max_in_memory', type=int, default=0, help="Datasets max in memory cache (in GB)"
    )

    DefaultConfig.add_argparse_args(parser)

    # retrieving model with temporary parsed arguments
    tmp_params, _ = parser.parse_known_args()

    # get pl_model_class in advance to know which params it needs
    pipeline = pipelines[tmp_params.pipeline]
    pl_model_class = pipeline[tmp_params.model]
    pl_model_class.add_argparse_args(parser)
    TransformersDataModule.add_argparse_args(parser)

    # loggers
    JsonBoardLogger.add_argparse_args(parser)

    # add callback / logger specific parameters
    TransformersModelCheckpointCallback.add_argparse_args(parser)

    # add all the available trainer options to argparse
    # ie: now --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # get NameSpace of parameters
    args = parser.parse_args()
    args = ExtendedNamespace.from_namespace(args)
    main(args)
