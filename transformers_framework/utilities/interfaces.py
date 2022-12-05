from dataclasses import dataclass

import torch


@dataclass
class StepOutput:
    loss: torch.Tensor
    keys: torch.Tensor

    def __getitem__(self, key):
        return eval(f"self.{key}")


@dataclass
class AnswerSentenceSelectionStepOutput(StepOutput):
    seq_class_loss: torch.Tensor
    ranker_predictions: torch.Tensor
    ranker_scores: torch.Tensor
    ranker_labels: torch.Tensor


@dataclass
class MaskedLanguageModelingStepOutput(StepOutput):
    masked_lm_loss: torch.Tensor
    masked_lm_predictions: torch.Tensor
    masked_lm_labels: torch.Tensor


@dataclass
class TokenDetectionStepOutput(StepOutput):
    token_detection_loss: torch.Tensor
    token_detection_predictions: torch.Tensor
    token_detection_labels: torch.Tensor
    replaced_ids: torch.Tensor


@dataclass
class MLMAndAS2StepOutput(AnswerSentenceSelectionStepOutput, MaskedLanguageModelingStepOutput):
    pass


@dataclass
class MLMAndTokenDetectionStepOutput(MaskedLanguageModelingStepOutput, TokenDetectionStepOutput):
    token_detection_labels: torch.Tensor
