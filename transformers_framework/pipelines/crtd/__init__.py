from transformers_framework.pipelines.crtd.bert import BertCRTD
from transformers_framework.pipelines.crtd.roberta import RobertaCRTD


models = dict(
    bert=BertCRTD,
    roberta=RobertaCRTD,
)
