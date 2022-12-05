from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.pipelines.as2.base import BaseModelAS2


class RobertaAS2(BaseModelAS2):

    config_class = RobertaConfig
    model_class = RobertaForSequenceClassification
    tokenizer_class = RobertaTokenizerFast
