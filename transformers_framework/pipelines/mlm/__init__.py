from transformers_framework.pipelines.mlm.bert import BertMLM
from transformers_framework.pipelines.mlm.roberta import RobertaMLM


models = dict(
    bert=BertMLM,
    roberta=RobertaMLM,
)
