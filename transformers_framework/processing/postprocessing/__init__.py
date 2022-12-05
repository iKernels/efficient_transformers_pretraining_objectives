from transformers_framework.processing.postprocessing.language_modeling.masked_language_modeling import (
    MaskedLanguageModelingPostProcessor,
)
from transformers_framework.processing.postprocessing.language_modeling.random_token_detection import (
    RandomTokenDetectionPostProcessor,
)
from transformers_framework.processing.postprocessing.parser import ParserPostProcessor
from transformers_framework.processing.postprocessing.tokenization import TokenizerPostProcessor
from transformers_framework.utilities.processors import create_dict_from_processors


post_processors = create_dict_from_processors(
    ParserPostProcessor,
    TokenizerPostProcessor,
    MaskedLanguageModelingPostProcessor,
    RandomTokenDetectionPostProcessor,
)
