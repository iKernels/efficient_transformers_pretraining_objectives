from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import MaskedLMOutput as TransformersMaskedLMOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput


@dataclass
class MaskedLMOutput(BaseModelOutput):
    r"""
    Base class for masked language modeling outputs.

    Args:
        masked_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        masked_lm_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    masked_lm_loss: Optional[torch.FloatTensor] = None
    masked_lm_logits: Optional[torch.FloatTensor] = None


@dataclass
class TokenDetectionOutput(BaseModelOutput):
    r"""
    Base class for token detection language modeling outputs.

    Args:
        token_detection_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        token_detection_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
            Prediction scores of the language modeling head (scores for each output token before SoftMax).
    """

    token_detection_loss: Optional[torch.FloatTensor] = None
    token_detection_logits: Optional[torch.FloatTensor] = None


@dataclass
class SequenceClassificationOutput(BaseModelOutput):
    r"""
    Base class for sequence classification outputs.

    Args:
        seq_class_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`seq_class_labels` is provided):
            Sequence classification loss.
        seq_class_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Sequence classification logits (scores for each input example).
    """

    seq_class_loss: Optional[torch.FloatTensor] = None
    seq_class_logits: Optional[torch.FloatTensor] = None


@dataclass
class TokenClassificationOutput(BaseModelOutput):
    r"""
    Base class for token classification outputs.

    Args:
        token_class_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`token_class_labels` is provided):
            Token classification loss.
        tok_class_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Token classification logits (scores for each token of each each input example).
    """

    token_class_loss: Optional[torch.FloatTensor] = None
    token_class_logits: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndSequenceClassificationOutput(
    MaskedLMOutput, SequenceClassificationOutput
):
    r""" Masked language modeling + sequence classification. """


@dataclass
class MaskedLMAndTokenClassificationOutput(
    MaskedLMOutput, TokenClassificationOutput
):
    r""" Masked language modeling + token classification. """


@dataclass
class TokenDetectionAndSequenceClassificationOutput(
    TokenDetectionOutput, SequenceClassificationOutput
):
    r""" Token detection + sequence classification. """


def sequence_classification_adaptation(res: SequenceClassifierOutput) -> SequenceClassificationOutput:
    r""" Convert transformers SequenceClassifierOutput to our SequenceClassificationOutput. """
    if isinstance(res, SequenceClassificationOutput):
        return res

    return SequenceClassificationOutput(
        attentions=res.attentions,
        hidden_states=res.hidden_states,
        seq_class_loss=res.loss,
        seq_class_logits=res.logits,
    )


def masked_lm_adaptation(res: TransformersMaskedLMOutput) -> MaskedLMOutput:
    r""" Convert transformers MaskedLMOutput to our MaskedLMOutput. """
    if isinstance(res, MaskedLMOutput):
        return res

    return MaskedLMOutput(
        last_hidden_state=None,
        hidden_states=res.hidden_states,
        attentions=res.attentions,
        masked_lm_loss=res.loss,
        masked_lm_logits=res.logits,
    )


def token_detection_adaptation(res: ElectraForPreTrainingOutput) -> TokenDetectionOutput:
    r""" Convert transformers ElectraForPreTrainingOutput to our TokenDetectionOutput. """
    if isinstance(res, TokenDetectionOutput):
        return res

    return TokenDetectionOutput(
        last_hidden_state=None,
        hidden_states=res.hidden_states,
        attentions=res.attentions,
        token_detection_loss=res.loss,
        token_detection_logits=res.logits,
    )
