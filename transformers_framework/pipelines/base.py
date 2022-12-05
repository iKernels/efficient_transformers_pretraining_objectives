import math
from argparse import ArgumentParser
from typing import Iterable, List, Mapping, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers_lightning.optimizers.super_optimizer import SuperOptimizer
from transformers_lightning.schedulers.super_scheduler import SuperScheduler

from transformers_framework.optimizers import optimizers
from transformers_framework.schedulers import schedulers
from transformers_framework.utilities.datamodules import TrainerStage_to_Names
from transformers_framework.utilities.functional import shrink_batch
from transformers_framework.utilities.models import load_model


class BaseModel(LightningModule):

    config_class: PretrainedConfig = None
    model_class: PreTrainedModel = None
    tokenizer_class: PreTrainedTokenizerBase = None

    REQUIRED_PRE_PROCESSORS = []
    REQUIRED_POST_PROCESSORS = []

    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters(hyperparameters)

        # check required processors are defined
        self.check_processors()

        # checking and fixing pretrained paths
        if self.hyperparameters.pre_trained_config is None:
            self.hyperparameters.pre_trained_config = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_config`, setting equal to `pre_trained_model`')

        if self.hyperparameters.pre_trained_tokenizer is None:
            self.hyperparameters.pre_trained_tokenizer = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_tokenizer`, setting equal to `pre_trained_model`')

        if self.hyperparameters.pre_trained_config is None or self.hyperparameters.pre_trained_tokenizer is None:
            raise ValueError(
                "Cannot instantiate model without at least a `pre_trained_config` and a `pre_trained_tokenizer`"
            )

        self.config = self.setup_config()
        self.model = self.setup_model(self.config)
        self.tokenizer = self.setup_tokenizer()

    def setup_config(self, **kwargs) -> PretrainedConfig:
        r""" Load or create the configuration and return it. """
        kwargs['num_labels'] = self.hyperparameters.num_labels
        return self.config_class.from_pretrained(self.hyperparameters.pre_trained_config, **kwargs)

    def setup_model(self, config: PretrainedConfig, **kwargs) -> PreTrainedModel:
        r"""
        Load model from scratch or from disk and return it.
        Side effects may be used to instantiate another model, i.e. a generator.
        """
        return load_model(self.model_class, self.hyperparameters.pre_trained_model, config=config, **kwargs)

    def setup_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        r""" Load the tokenizer from disk and return it. """
        return self.tokenizer_class.from_pretrained(self.hyperparameters.pre_trained_tokenizer, **kwargs)

    def forward(self, *args, **kwargs):
        r""" Simply call the `model` attribute with the given args and kwargs """
        return self.model(*args, **kwargs)

    def get_optimizer(self) -> SuperOptimizer:
        r""" Get optimizer as defined by hyperparameters. """
        optim_class = optimizers[self.hyperparameters.optimizer]
        return optim_class(self.hyperparameters, self.named_parameters())

    def get_scheduler(self, optimizer) -> SuperScheduler:
        r""" Get scheduler as defined by hyperparameters. """
        sched_class = schedulers[self.hyperparameters.scheduler]
        return sched_class(self.hyperparameters, optimizer)

    def infer_batch_size(self, batch_size: Optional[int] = None) -> int:
        r""" Automatically get the batch size for the actual step. """
        new_batch_size = self.hyperparameters[f"{TrainerStage_to_Names[self.trainer.state.stage]}_batch_size"]

        if batch_size is not None and (new_batch_size != batch_size):
            raise ValueError(
                f"Inferred batch size {new_batch_size} is different "
                f"from provided batch size {batch_size}, check what batch size you are logging"
            )

        return new_batch_size

    def check_metrics(self):
        r""" Check every metric in this model has a name starting either with 'train', 'valid', 'test'.
        If not, raise ValueError.
        """
        for name, module in self.named_children():
            if isinstance(module, Metric):
                if not (
                    name.startswith('train')
                    or name.startswith('valid')
                    or name.startswith('test')
                ):
                    raise ValueError("All metrics in the model must start with 'train', 'valid', 'test'")

    def setup(self, stage: Optional[str] = None):
        r""" Just check metrics are defined correctly. """
        rank_zero_warn(f"Running setup for stage {stage}")
        self.check_metrics()

    def reset_metrics(self, stage: str):
        r""" Reset all metrics in model for the given stage. """
        rank_zero_warn(f"Resetting {stage} metrics")
        for name, module in self.named_children():
            if isinstance(module, Metric) and name.startswith(stage):
                module.reset()

    def on_train_epoch_start(self):
        r""" Reset training metrics. """
        self.reset_metrics('train')

    def on_validation_epoch_start(self):
        r""" Reset validation metrics. """
        self.reset_metrics('valid')

    def on_test_epoch_start(self):
        r""" Reset test metrics. """
        self.reset_metrics('test')

    def log(self, *args: List, batch_size: Optional[int] = None, **kwargs):
        r""" Automatically manages logging:
        - Adds 'train/', 'valid/', 'test/' prefixes
        - Logs single values, iterables and dicts
        - Adds correcy batch size
        """

        if not (1 <= len(args) <= 2):
            raise ValueError(
                "log method should be called with only 1 (Mapping) or two "
                "(name, Union[Iterable, int, float, Tensor]) positional arguments"
            )

        if len(args) == 1:  # use must have passed a dict
            data = args[0]

        else:
            name, value = args
            # start with single built-in or Tensor values
            if (isinstance(value, torch.Tensor) and value.dim() == 0) or isinstance(value, (int, float)):
                data = {name: value}
            elif isinstance(value, Iterable):
                data = {f"{name}_class_{i}": v for i, v in enumerate(value)}
                data[f"{name}_mean"] = sum(value) / len(value)

        if not isinstance(data, Mapping):
            raise ValueError("Log value must be an Iterable, a Mapping or a single int/float")

        # check user is not logging metrics but only allowed values
        for v in data.values():
            if not isinstance(v, (int, float, torch.Tensor)):
                raise ValueError(
                    "`transformers_framework` automatically manages metrics, just log integers, floats or an "
                    "iterable or the previouses. Please use metric(...) and not metric.compute() to avoid overheads"
                )

        # add automatically stage prefix
        data = {f"{TrainerStage_to_Names[self.trainer.state.stage]}/{name}": value for name, value in data.items()}

        # add batch size
        batch_size = self.infer_batch_size(batch_size)

        # use super logger for synchronization and accumulation
        for k, v in data.items():
            super().log(k, v, batch_size=batch_size, **kwargs)

    def shrink_batch(self, input_ids: torch.Tensor, *args: torch.Tensor, pad_token_id: int = 0):
        r""" Remove data on the sequence length dimension in the positions where every example is padded. """
        return shrink_batch(
            input_ids,
            *args,
            pad_token_id=pad_token_id,
            shrink_to_multiples_of=8 if self.hyperparameters.precision == 16 else None,
        )

    def check_processors(self):
        r""" Check required pre- and post-processors are defined in `hyperparameters`. """
        for pre_processor in self.REQUIRED_PRE_PROCESSORS:
            if pre_processor not in self.hyperparameters.pre_processors:
                raise ValueError(f"{self.__class__.__name__} needs `{pre_processor}` pre_processor")

        for post_processor in self.REQUIRED_POST_PROCESSORS:
            if post_processor not in self.hyperparameters.post_processors:
                raise ValueError(f"{self.__class__.__name__} needs `{post_processor}` post_processor")

    def num_training_steps(self) -> int:
        r""" Total training steps inferred from datasets length, number of nodes and devices. """
        if self.trainer.max_steps is not None and self.trainer.max_steps >= 0:
            return self.trainer.max_steps

        if not has_len(self.trainer.datamodule.train_dataset):
            rank_zero_warn("Using IterableDataset, cannot compute max_steps, returning None")
            return None

        # train samples
        train_samples = len(self.trainer.datamodule.train_dataset)

        # number of training devices
        total_devices = self.trainer.num_devices * self.trainer.num_nodes
        rank_zero_warn(f"Number of training devices is {total_devices}")

        # the number of training samples may be modified in distributed training
        # to be divisible by the number of GPUs...
        train_samples_per_device = math.ceil(train_samples / total_devices)

        # train batches from the dataloader
        train_batches_per_device = math.ceil(train_samples_per_device / self.hyperparameters.train_batch_size)

        # eventually limit train batches
        limit_batches = self.trainer.limit_train_batches
        train_batches_per_device = (
            min(train_batches_per_device, limit_batches)
            if isinstance(limit_batches, int) else int(limit_batches * train_batches_per_device)
        )

        # train steps for each device
        train_steps_per_device = math.ceil(train_batches_per_device / self.trainer.accumulate_grad_batches)

        # total train steps across all epochs
        total_train_steps = train_steps_per_device * self.trainer.max_epochs
        rank_zero_warn(f"Automatically computed total steps equal to {total_train_steps}")

        return total_train_steps

    def configure_optimizers(self):
        r"""
        Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate.
        """

        # fix max number of steps
        self.hyperparameters.max_steps = self.num_training_steps()

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,  # The LR schduler
                    'interval': self.hyperparameters.scheduler_interval,  # The unit of the scheduler's step size
                    'frequency': self.hyperparameters.scheduler_frequency,  # The frequency of the scheduler
                }
        }

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        # add pre_trained model, tokenizer and config arguments. default config and tokenizer to model if missing
        parser.add_argument('--optimizer', type=str, default='adamw', choices=optimizers.keys())
        parser.add_argument(
            '--scheduler', type=str, default='linear_warmup', choices=schedulers.keys()
        )
        parser.add_argument('--scheduler_interval', type=str, default='step', choices=('step', 'epoch'))
        parser.add_argument('--scheduler_frequency', type=int, default=1)

        parser.add_argument('--pre_trained_model', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_tokenizer', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_config', type=str, required=False, default=None)
        parser.add_argument('--num_labels', type=int, required=False, default=2)

        # retrieving classes with temporary parsered arguments
        tmp_params, _ = parser.parse_known_args()

        # get pl_model_class in advance to know which params it needs
        optimizers[tmp_params.optimizer].add_argparse_args(parser)
        schedulers[tmp_params.scheduler].add_argparse_args(parser)
