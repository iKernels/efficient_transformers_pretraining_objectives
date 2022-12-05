from transformers import BertConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.architectures.bert.modeling_bert import BertForRandomTokenDetection
from transformers_framework.pipelines.crtd.base import BaseModelCRTD


class BertCRTD(BaseModelCRTD):
    r""" Bert with a simple RTS binary classification head on top. Uses clusters to improve replacement. """

    config_class = BertConfig
    model_class = BertForRandomTokenDetection
    tokenizer_class = BertTokenizerFast
