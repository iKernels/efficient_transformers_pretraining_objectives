from transformers import BertConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.architectures.bert.modeling_bert import BertForRandomTokenDetection
from transformers_framework.pipelines.rtd.base import BaseModelRTD


class BertRTD(BaseModelRTD):
    r""" Bert with a simple RTD binary classification head on top. """

    config_class = BertConfig
    model_class = BertForRandomTokenDetection
    tokenizer_class = BertTokenizerFast
