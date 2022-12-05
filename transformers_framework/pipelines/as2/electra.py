from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForSequenceClassification
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.as2.base import BaseModelAS2


class ElectraAS2(BaseModelAS2):

    config_class = ElectraConfig
    model_class = ElectraForSequenceClassification
    tokenizer_class = ElectraTokenizerFast
