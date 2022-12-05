import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, logger
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_outputs import TokenDetectionOutput


class RobertaTokenDetectionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_state, **kwargs):
        hidden_state = self.dense(hidden_state)
        hidden_state = gelu(hidden_state)
        hidden_state = self.layer_norm(hidden_state)

        # project back to size of vocabulary with bias
        hidden_state = self.decoder(hidden_state)
        return hidden_state


class RobertaOnlyTokenDetectionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaTokenDetectionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RobertaForRandomTokenDetection(RobertaPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForRandomTS` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.cls = RobertaOnlyTokenDetectionHead(config)

        self.loss_fct = CrossEntropyLoss(ignore_index=IGNORE_IDX)    # -100 index = padding token

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the random token substitution loss.
            Indices should be in ``[-100, 0, 1]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored, the loss is only computed for the tokens with labels
            in ``[0, 1]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        rtd_loss = None
        if labels is not None:
            # do not compute loss when labels are set to -100
            rtd_loss = self.loss_fct(prediction_scores.view(-1, 2), labels.view(-1))

        return TokenDetectionOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_detection_loss=rtd_loss,
            token_detection_logits=prediction_scores
        )
