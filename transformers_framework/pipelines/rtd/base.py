from argparse import ArgumentParser
from typing import Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_outputs import TokenDetectionOutput, token_detection_adaptation
from transformers_framework.pipelines.base import BaseModel
from transformers_framework.utilities.arguments import add_token_detection_arguments
from transformers_framework.utilities.functional import index_multi_tensors, multi_get_from_dict
from transformers_framework.utilities.interfaces import TokenDetectionStepOutput


class BaseModelRTD(BaseModel):
    r"""
    A model that use RTD loss where the probability of swapping each token is weighted
    by the experience of previous similar switchings.
    """

    REQUIRED_POST_PROCESSORS = ["random_token_detection"]

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=2)
        self.train_acc = MulticlassAccuracy(**metrics_kwargs)
        self.train_f1 = MulticlassF1Score(**metrics_kwargs)
        self.valid_acc = MulticlassAccuracy(**metrics_kwargs)
        self.valid_f1 = MulticlassF1Score(**metrics_kwargs)
        self.test_acc = MulticlassAccuracy(**metrics_kwargs)
        self.test_f1 = MulticlassF1Score(**metrics_kwargs)

    def forward(self, *args, **kwargs):
        return token_detection_adaptation(super().forward(*args, **kwargs))

    def step(self, batch: Dict) -> TokenDetectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        keys, input_ids, attention_mask, token_type_ids, labels = (
            multi_get_from_dict(batch, 'keys', 'input_ids', 'attention_mask', 'token_type_ids', 'labels')
        )

        results: TokenDetectionOutput = self(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
        )

        predictions = results.token_detection_logits.argmax(dim=-1)
        predictions, labels = index_multi_tensors(predictions, labels, positions=labels != IGNORE_IDX)

        # scale loss
        loss = results.token_detection_loss * self.hyperparameters.td_weight

        return TokenDetectionStepOutput(
            keys=keys,
            loss=loss,
            token_detection_loss=results.token_detection_loss,
            token_detection_predictions=predictions,
            token_detection_labels=labels,
            replaced_ids=input_ids,
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output: TokenDetectionStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=False)
        train_acc = self.train_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        train_f1 = self.train_f1(step_output.token_detection_predictions, step_output.token_detection_labels)

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('rtd_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('rtd_accuracy', train_acc, **log_kwargs)
        self.log('rtd_f1', train_f1, **log_kwargs)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output: TokenDetectionStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        valid_acc = self.valid_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        valid_f1 = self.valid_f1(step_output.token_detection_predictions, step_output.token_detection_labels)

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('rtd_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('rtd_accuracy', valid_acc, **log_kwargs)
        self.log('rtd_f1', valid_f1, **log_kwargs)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output: TokenDetectionStepOutput = self.step(batch)
       
        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        test_acc = self.test_acc(step_output.token_detection_predictions, step_output.token_detection_labels)
        test_f1 = self.test_f1(step_output.token_detection_predictions, step_output.token_detection_labels)

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('rtd_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('rtd_accuracy', test_acc, **log_kwargs)
        self.log('rtd_f1', test_f1, **log_kwargs)

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(BaseModelRTD, BaseModelRTD).add_argparse_args(parser)
        add_token_detection_arguments(parser)
