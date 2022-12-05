from argparse import ArgumentParser
from typing import Dict

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.f_beta import MulticlassF1Score
from transformers import PretrainedConfig, PreTrainedModel
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_outputs import (
    TokenDetectionOutput,
    masked_lm_adaptation,
    token_detection_adaptation,
)
from transformers_framework.pipelines.mlm.base import BaseModelMLM
from transformers_framework.utilities.arguments import add_token_detection_arguments
from transformers_framework.utilities.functional import multi_get_from_dict, sample_from_distribution
from transformers_framework.utilities.interfaces import MLMAndTokenDetectionStepOutput
from transformers_framework.utilities.models import load_model


class BaseModelMLMAndTD(BaseModelMLM):

    REQUIRED_POST_PROCESSORS = ["random_token_detection"]
    generator_model_class: PreTrainedModel

    def get_generator_config(self, config: PretrainedConfig, **kwargs):
        r""" Return config for the generator. """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.tie_weights()

        discriminator_metrics_kwargs = dict(average=None, ignore_index=IGNORE_IDX, num_classes=2)
        self.train_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.train_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)
        self.valid_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.valid_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)
        self.test_discriminator_acc = MulticlassAccuracy(**discriminator_metrics_kwargs)
        self.test_discriminator_f1 = MulticlassF1Score(**discriminator_metrics_kwargs)

    def tie_weights(self):
        r""" Do every sort of weight tying here. """

    def setup_config(self) -> PretrainedConfig:
        kwargs = dict()

        # discriminator config
        config = self.config_class.from_pretrained(self.hyperparameters.pre_trained_config, **kwargs)

        # generator config
        if self.hyperparameters.pre_trained_generator_config is not None:
            self.generator_config = self.config_class.from_pretrained(
                self.hyperparameters.pre_trained_generator_config, **kwargs
            )
        elif self.hyperparameters.pre_trained_generator_model is not None:
            rank_zero_warn(
                'Found None `pre_trained_generator_config`, setting equal to `pre_trained_generator_model`'
            )
            self.hyperparameters.pre_trained_generator_config = self.hyperparameters.pre_trained_generator_model
            self.generator_config = self.config_class.from_pretrained(
                self.hyperparameters.pre_trained_generator_config, **kwargs
            )
        else:
            rank_zero_warn(
                f"Automatically creating generator config of size {self.hyperparameters.generator_size}"
            )
            self.generator_config = self.get_generator_config(config, **kwargs)

        return config

    def setup_model(self, config: PretrainedConfig) -> PreTrainedModel:
        # generator model assigned directly to self
        self.generator = load_model(
            self.generator_model_class, self.hyperparameters.pre_trained_generator_model, config=self.generator_config
        )
        return load_model(self.model_class, self.hyperparameters.pre_trained_model, config=config)

    def forward(self, *args, **kwargs):
        return token_detection_adaptation(super(BaseModelMLM, self).forward(*args, **kwargs))

    def step(self, batch: Dict) -> MLMAndTokenDetectionStepOutput:
        r""" Forward step on the generator, labels creation and step on the discriminator. """
        keys, input_ids, masked_input_ids, attention_mask, token_type_ids, masked_labels = (
            multi_get_from_dict(
                batch, 'keys', 'input_ids', 'masked_input_ids', 'attention_mask', 'token_type_ids', 'masked_labels'
            )
        )

        generator_output = masked_lm_adaptation(
            self.generator(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=masked_labels,
                return_dict=True,
            )
        )
        
        is_mlm_applied = (masked_labels != IGNORE_IDX)

        with torch.no_grad():
            # take predictions of the generator where a mask token was placed
            generator_logits = generator_output.masked_lm_logits[is_mlm_applied]
            generator_generations = sample_from_distribution(
                generator_logits, sample_function=self.hyperparameters.sample_function
            )
            generator_predictions = generator_logits.argmax(dim=-1)
            masked_labels = masked_labels[is_mlm_applied]

            # replace mask tokens with the ones predicted by the generator
            discriminator_input_ids = input_ids.clone()
            discriminator_input_ids[is_mlm_applied] = generator_generations

            # create labels for the discriminator as the mask of the original labels (1 where a token was masked)
            # while at the same time avoiding setting a positive label when prediction where the generator was correct
            discriminator_labels = is_mlm_applied.clone()
            discriminator_labels[is_mlm_applied] = (generator_generations != masked_labels)

            discriminator_labels = discriminator_labels.to(dtype=torch.long)

        discriminator_output: TokenDetectionOutput = self(
            input_ids=discriminator_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=discriminator_labels,
        )

        loss = (
            generator_output.masked_lm_loss
            + discriminator_output.token_detection_loss * self.hyperparameters.td_weight
        )

        discriminator_predictions = (discriminator_output.token_detection_logits > 0.5).to(dtype=torch.int64)
    
        return MLMAndTokenDetectionStepOutput(
            keys=keys,
            loss=loss,
            token_detection_loss=discriminator_output.token_detection_loss,
            token_detection_predictions=discriminator_predictions,
            token_detection_labels=discriminator_labels,
            masked_lm_loss=generator_output.masked_lm_loss,
            masked_lm_predictions=generator_predictions,
            masked_lm_labels=masked_labels,
            replaced_ids=discriminator_input_ids,
        )

    def training_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output: MLMAndTokenDetectionStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=False)
        train_mlm_acc = self.train_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        train_discriminator_acc = self.train_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        train_discriminator_f1 = self.train_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('generator_loss', step_output.masked_lm_loss, **log_kwargs)
        self.log('generator_accuracy', train_mlm_acc, **log_kwargs)
        self.log('discriminator_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('discriminator_accuracy', train_discriminator_acc, **log_kwargs)
        self.log('discriminator_f1', train_discriminator_f1, **log_kwargs)

        return step_output.loss

    def validation_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output: MLMAndTokenDetectionStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        valid_mlm_acc = self.valid_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        valid_discriminator_acc = self.valid_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        valid_discriminator_f1 = self.valid_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('generator_loss', step_output.masked_lm_loss, **log_kwargs)
        self.log('generator_accuracy', valid_mlm_acc, **log_kwargs)
        self.log('discriminator_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('discriminator_accuracy', valid_discriminator_acc, **log_kwargs)
        self.log('discriminator_f1', valid_discriminator_f1, **log_kwargs)

    def test_step(self, batch, *args):
        r""" Start by masking some tokens. """
        step_output: MLMAndTokenDetectionStepOutput = self.step(batch)

        log_kwargs = dict(on_epoch=True, prog_bar=True, sync_dist=True)
        test_mlm_acc = self.test_mlm_acc(step_output.masked_lm_predictions, step_output.masked_lm_labels)
        test_discriminator_acc = self.test_discriminator_acc(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )
        test_discriminator_f1 = self.test_discriminator_f1(
            step_output.token_detection_predictions, step_output.token_detection_labels
        )

        self.log('loss', step_output.loss, **log_kwargs)
        self.log('generator_loss', step_output.masked_lm_loss, **log_kwargs)
        self.log('generator_accuracy', test_mlm_acc, **log_kwargs)
        self.log('discriminator_loss', step_output.token_detection_loss, **log_kwargs)
        self.log('discriminator_accuracy', test_discriminator_acc, **log_kwargs)
        self.log('discriminator_f1', test_discriminator_f1, **log_kwargs)

    @staticmethod
    def add_argparse_args(parser: ArgumentParser):
        super(BaseModelMLMAndTD, BaseModelMLMAndTD).add_argparse_args(parser)
        # mlm arguments already added by superclass
        add_token_detection_arguments(parser)
