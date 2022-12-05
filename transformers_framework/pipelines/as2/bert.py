from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.as2.base import BaseModelAS2


class BertAS2(BaseModelAS2):

    config_class = BertConfig
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizerFast
