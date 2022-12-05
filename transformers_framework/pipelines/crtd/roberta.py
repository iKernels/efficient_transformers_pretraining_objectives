from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.modeling_roberta import RobertaForRandomTokenDetection
from transformers_framework.pipelines.crtd.base import BaseModelCRTD


class RobertaCRTD(BaseModelCRTD):
    r""" Roberta with a simple RTS binary classification head on top. Uses clusters to improve replacement. """

    config_class = RobertaConfig
    model_class = RobertaForRandomTokenDetection
    tokenizer_class = RobertaTokenizerFast
