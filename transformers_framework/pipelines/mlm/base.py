from typing import Dict

from torchmetrics.classification.accuracy import MulticlassAccuracy
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_outputs import MaskedLMOutput, masked_lm_adaptation
from transformers_framework.pipelines.base import BaseModel
from transformers_framework.utilities.functional import index_multi_tensors, multi_get_from_dict
from transformers_framework.utilities.interfaces import MaskedLanguageModelingStepOutput


class BaseModelMLM(BaseModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        metrics_args = (self.tokenizer.vocab_size, )
        metrics_kwargs = dict(average='micro', ignore_index=IGNORE_IDX)
        self.train_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.valid_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)
        self.test_mlm_acc = MulticlassAccuracy(*metrics_args, **metrics_kwargs)

    def forward(self, *args, **kwargs):
        return masked_lm_adaptation(super().forward(*args, **kwargs))

    def step(self, batch: Dict) -> MaskedLanguageModelingStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        keys, masked_input_ids, attention_mask, token_type_ids, masked_labels = (
            multi_get_from_dict(batch, 'keys', 'masked_input_ids', 'attention_mask', 'token_type_ids', 'masked_labels')
        )

        results: MaskedLMOutput = self(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=masked_labels,
        )

        mlm_predictions = results.masked_lm_logits.argmax(dim=-1)
        mlm_predictions, masked_labels = index_multi_tensors(
            mlm_predictions, masked_labels, positions=masked_labels != IGNORE_IDX
        )

        return MaskedLanguageModelingStepOutput(
            keys=keys,
            loss=results.masked_lm_loss,
            masked_lm_loss=results.masked_lm_loss,
            masked_lm_predictions=mlm_predictions,
            masked_lm_labels=masked_labels,
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output: MaskedLanguageModelingStepOutput = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=False)
        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)

        self.log('mlm_loss', step_output.loss, **log_kwargs)
        self.log('mlm_accuracy', train_mlm_acc, **log_kwargs)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output: MaskedLanguageModelingStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        valid_mlm_acc = self.valid_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)

        self.log('mlm_loss', step_output.loss, **log_kwargs)
        self.log('mlm_accuracy', valid_mlm_acc, **log_kwargs)

    def test_step(self, batch, *args):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest such as accuracy.
        """
        step_output: MaskedLanguageModelingStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        test_mlm_acc = self.test_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)

        self.log('mlm_loss', step_output.loss, **log_kwargs)
        self.log('mlm_accuracy', test_mlm_acc, **log_kwargs)
