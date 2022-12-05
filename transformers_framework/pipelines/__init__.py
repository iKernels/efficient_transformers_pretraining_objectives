from transformers_framework.pipelines.as2 import models as as2_models
from transformers_framework.pipelines.mlm import models as mlm_models
from transformers_framework.pipelines.rtd import models as rtd_models
from transformers_framework.pipelines.td_mlm import models as td_mlm_models
from transformers_framework.pipelines.crtd import models as crtd_models


pipelines = dict(
    answer_sentence_selection=as2_models,
    masked_language_modeling=mlm_models,
    random_token_detection=rtd_models,
    cluster_random_token_detection=crtd_models,
    token_detection_and_masked_language_modeling=td_mlm_models,
)
