from abc import abstractmethod
from argparse import Namespace
from typing import Dict, List

from datasets import Features
from transformers.tokenization_utils import PreTrainedTokenizerBase

from transformers_framework.processing.processor import Processor


class PostProcessor(Processor):
    r""" Super class of all post-processors. Post-processors prepare data samples before providing them
    to the models. Post-processing is a 1-1 operation, so the dataset size cannot change.
    The post-processing is performed inside the pytorch DataLoader in parallel on many workers.
    """

    REQUIRED_PREVIOUS_POST_PROCESSORS: List[str] = []

    def __init__(
        self,
        hyperparameters: Namespace,
        features: Features,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        stage_name: str,
    ):
        super().__init__(hyperparameters, features, tokenizer, seed, stage_name)
        self.check_previous_processors()

    def check_previous_processors(self):
        r""" Check every processor in `REQUIRED_PREVIOUS_POST_PROCESSORS` is present before this. """
        self_position = self.hyperparameters.post_processors.index(self.name)
        assert self_position >= 0
        for name in self.REQUIRED_PREVIOUS_POST_PROCESSORS:
            if name not in self.hyperparameters.post_processors[:self_position]:
                raise ValueError(f"post_processor {self.name} requires post_processor {name} before it")

    @abstractmethod
    def __call__(self, sample: Dict) -> Dict:
        r""" Process a single sample, for example by tokenizing text or converting lists to tensors. """
