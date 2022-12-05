from transformers_framework.pipelines.as2.bert import BertAS2
from transformers_framework.pipelines.as2.electra import ElectraAS2
from transformers_framework.pipelines.as2.roberta import RobertaAS2


models = dict(
    bert=BertAS2,
    roberta=RobertaAS2,
    electra=ElectraAS2,
)
