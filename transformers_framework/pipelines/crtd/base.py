from argparse import ArgumentParser

import torch
from typing import Dict

from transformers_framework.language_modeling.cluster_random_token_substitution import ClusterRandomTokenSubstitution
from transformers_framework.pipelines.rtd.base import BaseModelRTD
from transformers_framework.utilities.interfaces import TokenDetectionStepOutput
from transformers_framework.architectures.modeling_outputs import TokenDetectionOutput
from transformers_framework.utilities.models import read_clusters
from transformers_lightning.language_modeling import IGNORE_IDX
from transformers_framework.utilities.functional import index_multi_tensors, multi_get_from_dict

class BaseModelCRTD(BaseModelRTD):
    r"""
    A model that use RTS loss where the probability of swapping each token is weighted
    by the experience of previous similar switchings.
    """

    REQUIRED_POST_PROCESSORS = ['tokenizer']

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.register_buffer(
            'token_to_cluster_map', read_clusters(self.hyperparameters.clusters_filename), persistent=True
        )    # token -> clusters
        number_of_clusters = self.token_to_cluster_map.max() + 1
        self.register_buffer('counts', torch.ones(number_of_clusters, number_of_clusters, dtype=torch.float32))

        self.rts = ClusterRandomTokenSubstitution(
            self.tokenizer,
            probability=self.hyperparameters.probability,
            model=self,
            beta=self.hyperparameters.beta,
        )

    def step(self, batch: Dict) -> TokenDetectionStepOutput:
        r""" Forward step is shared between all train/val/test steps. """
        keys, input_ids, attention_mask, token_type_ids = (
            multi_get_from_dict(batch, 'keys', 'input_ids', 'attention_mask', 'token_type_ids')
        )

        # randomly replace 15% of the tokens using weighted probabilities
        replaced_ids, labels = self.rts(input_ids)

        results: TokenDetectionOutput = self(
            input_ids=replaced_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        predictions = results.token_detection_logits.argmax(dim=-1)

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

    def update_count_vector(
        self,
        originals: torch.Tensor,
        tampereds: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        r"""
        Update the vector of counts based on new predicitions.

        Args:
            originals:
                original ids of shape (batch_size, max_sequence_len)
            tampereds:
                modified ids of shape (batch_size, max_sequence_len)
            attention_mask:
                attention mask to update only on relevant positions of shape (batch_size, max_sequence_len)
            predictions:
                predictions for each modified ids of shape (batch_size, max_sequence_len)
            labels:
                gold labels of rts of shape (batch_size, max_sequence_len)

        Example:
            >>> attention_mask = torch.tensor([1, 1, 1, 1, 1, 1])
            >>> originals = torch.tensor([2, 3, 56, 1, 2, 23])
            >>> tampereds = torch.tensor([2, 33, 76, 1, 2, 28])
            >>> predictions = torch.tensor([0, 1, 0, 1, 1, 0])
            >>> labels = torch.tensor([0, 1, 1, 0, 0, 0])
            >>> updates = (predictions != labels) * 2 - 1
            torch.tensor([-1, -1, 1, 1, 1, -1])
        """

        indexes = (attention_mask > 0)
        indexes = indexes & (originals != tampereds)  # select positions where something changed

        originals = originals[indexes]
        predictions = predictions[indexes]
        tampereds = tampereds[indexes]
        labels = labels[indexes]

        originals_clusters = self.token_to_cluster_map[originals]
        tampereds_clustures = self.token_to_cluster_map[tampereds]

        updates_matrix = torch.zeros_like(self.counts)
        updates_matrix[originals_clusters, tampereds_clustures] += (predictions != labels) * 2 - 1

        updates_matrix = self.all_gather(updates_matrix).sum(dim=0)

        # works in distributed because it is registered as buffer
        self.counts += updates_matrix

    def training_step(self, batch, *args):
        r""" Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """

        step_output: TokenDetectionStepOutput = self.step(batch)

        self.update_count_vector(
            originals=batch['input_ids'],
            tampereds=step_output.replaced_ids,
            predictions=step_output.token_detection_predictions,
            labels=step_output.token_detection_labels,
            attention_mask=batch['attention_mask'],
        )
  
        self.log('loss', step_output.loss, on_epoch=True, prog_bar=True)
        self.log('rts_loss', step_output.token_detection_loss, on_epoch=True, prog_bar=True)
        self.log(
            'rts_accuracy',
            self.train_acc(step_output.token_detection_predictions, step_output.token_detection_labels),
            on_epoch=True,
        )
        self.log(
            'rts_f1_positive',
            self.train_f1(step_output.token_detection_predictions, step_output.token_detection_labels)[1],
            on_epoch=True,
        )

        return step_output.loss

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(BaseModelCRTD, BaseModelCRTD).add_argparse_args(parser)
        parser.add_argument('--beta', default=2.0, required=False, type=float)
        parser.add_argument('--clusters_filename', type=str, required=True)
