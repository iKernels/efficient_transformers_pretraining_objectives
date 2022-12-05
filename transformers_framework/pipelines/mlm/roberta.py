from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast

from transformers_framework.pipelines.mlm.base import BaseModelMLM


class RobertaMLM(BaseModelMLM):

    config_class = RobertaConfig
    model_class = RobertaForMaskedLM
    tokenizer_class = RobertaTokenizerFast
