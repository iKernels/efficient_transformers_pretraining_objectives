from transformers_framework.pipelines.rtd.bert import BertRTD
from transformers_framework.pipelines.rtd.roberta import RobertaRTD


models = dict(
    bert=BertRTD,
    roberta=RobertaRTD,
)
