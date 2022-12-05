from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.mlm.base import BaseModelMLM


class BertMLM(BaseModelMLM):

    config_class = BertConfig
    model_class = BertForMaskedLM
    tokenizer_class = BertTokenizerFast
