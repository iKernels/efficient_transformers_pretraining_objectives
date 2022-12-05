from transformers.models.roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.modeling_roberta import RobertaForRandomTokenDetection
from transformers_framework.pipelines.rtd.base import BaseModelRTD


class RobertaRTD(BaseModelRTD):
    r""" Roberta with a simple RTD binary classification head on top. """

    config_class = RobertaConfig
    model_class = RobertaForRandomTokenDetection
    tokenizer_class = RobertaTokenizerFast
