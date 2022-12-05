from transformers import ElectraForMaskedLM, ElectraForPreTraining, ElectraTokenizerFast, PretrainedConfig
from transformers.models.electra.modeling_electra import ElectraConfig

from transformers_framework.pipelines.td_mlm.base import BaseModelMLMAndTD
from transformers_framework.utilities.models import get_electra_reduced_generator_config, tie_weights_electra


class ElectraMLMAndTD(BaseModelMLMAndTD):

    config_class = ElectraConfig
    model_class = ElectraForPreTraining
    generator_model_class = ElectraForMaskedLM
    tokenizer_class = ElectraTokenizerFast

    def get_generator_config(self, config: PretrainedConfig, **kwargs):
        return get_electra_reduced_generator_config(
            config, factor=self.hyperparameters.generator_size, **kwargs
        )

    def tie_weights(self):
        tie_weights_electra(
            self.generator,
            self.model,
            tie_generator_discriminator_embeddings=self.hyperparameters.tie_generator_discriminator_embeddings,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )
