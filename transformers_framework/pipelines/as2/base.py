from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
)
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_outputs import (
    SequenceClassificationOutput,
    sequence_classification_adaptation,
)
from transformers_framework.pipelines.base import BaseModel
from transformers_framework.utilities.arguments import add_answer_sentence_selection_arguments
from transformers_framework.utilities.functional import index_multi_tensors, multi_get_from_dict
from transformers_framework.utilities.interfaces import AnswerSentenceSelectionStepOutput


class BaseModelAS2(BaseModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        if self.hyperparameters.num_labels != 2:
            raise ValueError("answer_sentence_selection pipeline requires `--num_labels 2`")

        self.train_acc = BinaryAccuracy()

        val_metrics_kwargs = dict(
            empty_target_action=hyperparameters.val_metrics_empty_target_action,
            compute_on_step=False,
        )
        self.valid_acc = BinaryAccuracy()
        self.valid_map = RetrievalMAP(**val_metrics_kwargs)
        self.valid_mrr = RetrievalMRR(**val_metrics_kwargs)
        self.valid_p1 = RetrievalPrecision(k=1, **val_metrics_kwargs)
        self.valid_hr5 = RetrievalHitRate(k=5, **val_metrics_kwargs)
        self.valid_ndgc = RetrievalNormalizedDCG(**val_metrics_kwargs)

        test_metrics_kwargs = dict(
            empty_target_action=hyperparameters.test_metrics_empty_target_action,
            compute_on_step=False,
        )
        self.test_acc = BinaryAccuracy()
        self.test_map = RetrievalMAP(**test_metrics_kwargs)
        self.test_mrr = RetrievalMRR(**test_metrics_kwargs)
        self.test_p1 = RetrievalPrecision(k=1, **test_metrics_kwargs)
        self.test_hr5 = RetrievalHitRate(k=5, **test_metrics_kwargs)
        self.test_ndgc = RetrievalNormalizedDCG(**test_metrics_kwargs)

    def forward(self, *args, **kwargs):
        return sequence_classification_adaptation(super().forward(*args, **kwargs))

    def validation_update_as2_metrics(
        self, ranker_scores: torch.Tensor, ranker_labels: torch.Tensor, keys: torch.Tensor
    ):
        r""" AS2 metrics should only be computed globally.
        Partial accumulation for each step is provided by this method.
        """
        kwargs = dict(preds=ranker_scores, target=ranker_labels, indexes=keys)
        self.valid_map.update(**kwargs)
        self.valid_mrr.update(**kwargs)
        self.valid_p1.update(**kwargs)
        self.valid_hr5.update(**kwargs)
        self.valid_ndgc.update(**kwargs)

    def test_update_as2_metrics(
        self, ranker_scores: torch.Tensor, ranker_labels: torch.Tensor, keys: torch.Tensor
    ):
        r""" AS2 metrics should only be computed globally.
        Partial accumulation for each step is provided by this method.
        """
        kwargs = dict(preds=ranker_scores, target=ranker_labels, indexes=keys)
        self.test_map.update(**kwargs)
        self.test_mrr.update(**kwargs)
        self.test_p1.update(**kwargs)
        self.test_hr5.update(**kwargs)
        self.test_ndgc.update(**kwargs)

    def step(self, batch: Dict) -> AnswerSentenceSelectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        keys, input_ids, attention_mask, token_type_ids, ranker_labels = (
            multi_get_from_dict(batch, 'keys', 'input_ids', 'attention_mask', 'token_type_ids', 'labels')
        )

        results: SequenceClassificationOutput = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=ranker_labels,
        )

        # results.seq_class_logits may have size (batch_size, num_labels) or (batch_size, k, num_labels)
        ranker_predictions, ranker_labels, keys = index_multi_tensors(
            results.seq_class_logits, ranker_labels, keys, positions=ranker_labels != IGNORE_IDX
        )
        ranker_scores = ranker_predictions.softmax(dim=-1)[:, -1]
        ranker_predictions = ranker_predictions.argmax(dim=-1)

        return AnswerSentenceSelectionStepOutput(
            keys=keys,
            loss=results.seq_class_loss,
            seq_class_loss=results.seq_class_loss,
            ranker_predictions=ranker_predictions,
            ranker_scores=ranker_scores,
            ranker_labels=ranker_labels,
        )

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        step_output: AnswerSentenceSelectionStepOutput = self.step(batch)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=False)
        train_acc = self.train_acc(step_output.ranker_predictions, step_output.ranker_labels)
        self.log('loss', step_output.loss, **log_kwargs)
        self.log('accuracy', train_acc, **log_kwargs)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        step_output: AnswerSentenceSelectionStepOutput = self.step(batch)

        # AS2 metrics should only be computed globally
        self.validation_update_as2_metrics(step_output.ranker_scores, step_output.ranker_labels, step_output.keys)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        val_acc = self.valid_acc(step_output.ranker_predictions, step_output.ranker_labels)
        self.log('loss', step_output.loss, **log_kwargs)
        self.log('accuracy', val_acc, **log_kwargs)

    def test_step(self, batch, *args):
        r""" Compute predictions and log retrieval results. """
        step_output: AnswerSentenceSelectionStepOutput = self.step(batch)

        # AS2 metrics should only be computed globally
        self.test_update_as2_metrics(step_output.ranker_scores, step_output.ranker_labels, step_output.keys)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        test_acc = self.test_acc(step_output.ranker_predictions, step_output.ranker_labels)
        self.log('loss', step_output.loss, **log_kwargs)
        self.log('accuracy', test_acc, **log_kwargs)

    def validation_epoch_end(self, outputs: List[Any]):
        r""" Just log metrics. """
        self.log('map', self.valid_map.compute())
        self.log('mrr', self.valid_mrr.compute())
        self.log('p1', self.valid_p1.compute())
        self.log('hr5', self.valid_hr5.compute())
        self.log('ndcg', self.valid_ndgc.compute())

    def test_epoch_end(self, outputs: List[Any]):
        r""" Just log metrics. """
        self.log('map', self.test_map.compute())
        self.log('mrr', self.test_mrr.compute())
        self.log('p1', self.test_p1.compute())
        self.log('hr5', self.test_hr5.compute())
        self.log('ndcg', self.test_ndgc.compute())

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(BaseModelAS2, BaseModelAS2).add_argparse_args(parser)
        add_answer_sentence_selection_arguments(parser)
