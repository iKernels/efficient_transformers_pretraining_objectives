import logging
import math
import random
from argparse import ArgumentParser, Namespace
from typing import Dict, List

from blingfire import text_to_sentences

from process_datasets.strategies.strategy import Strategy


class ElectraPretrainingStrategy(Strategy):
    r"""
    ELECTRA introduced a slightly modified way to create examples. In ELECTRA sentences
    are usually paired with some consecutive amount of text that makes sense. Sometimes, the example contains
    only a single sentence and sometimes it is trimmed to short lengths to introduce some padding.
    """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        assert self.hparams.probability_random_length >= 0 and self.hparams.probability_random_length <= 1, (
            "`--probability_random_length` must be None or a positive integer"
        )

        assert self.hparams.probability_single_sentence >= 0 and self.hparams.probability_single_sentence <= 1, (
            "`--probability_single_sentence` must be None or a positive integer"
        )

        assert (self.hparams.probability_first_segment_over_length >= 0) and (
            self.hparams.probability_first_segment_over_length <= 1
        ), (
            "`--probability_first_segment_over_length` must be None or a positive integer"
        )

        assert self.hparams.max_sequence_length is not None or self.hparams.max_sequence_length >= 1, (
            "`max-sequence-length` must be None or a positive integer"
        )

        self.current_sentences = []
        self.current_lengths = []
        self.target_length = self.hparams.max_sequence_length

        # this will check if args.tokenizer is available as path or pre_trained tokenizer
        tokenizer_class = getattr(globals()['transformers'], hparams.tokenizer_class or 'AutoTokenizer')
        self.tokenizer = tokenizer_class.from_pretrained(hparams.tokenizer)
        logging.info(f"Loaded tokenizer. Is it fast? {self.tokenizer.is_fast}")

    @property
    def current_length(self):
        return sum(self.current_lengths)

    def is_sentence_ok(self, line):
        r""" Filter sentence if not long enough. """
        return len(line) >= self.hparams.min_sentence_length

    def clean(self, line):
        r""" () is remainder after links were filtered out. """
        return " ".join(line.replace("()", "").split())

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        r""" Process a batch of inputs. """
        new_examples = []

        # for every doc
        for entry in batch:
            doc = entry[self.hparams.field]

            if doc is not None:
                # for every sentence
                for sentence in text_to_sentences(doc).split("\n"):

                    # continue if empty or too short
                    if not self.is_sentence_ok(sentence):
                        continue

                    example = self.add_line(sentence)
                    if example:
                        new_examples.append(example)

                if self.current_length != 0:
                    example = self.create_example()
                    new_examples.append(example)

        return new_examples

    def get_encoded_length(self, sentence):
        r""" Get number of expected tokens in this sentence. """
        return len(self.tokenizer.tokenize(sentence))

    def add_line(self, line):
        r"""Adds a line of text to the current example being built."""
        line = self.clean(line)

        # retrieve line length preview (no special tokens)
        length = self.get_encoded_length(line)

        self.current_sentences.append(line)
        self.current_lengths.append(length)

        if self.current_length >= self.target_length:
            return self.create_example()
        return None

    def create_example(self):
        r"""
        Creates a pre-training example from the current list of sentences.

        First give a small chance to only have one segment as in classification tasks.
        Then, the sentence goes to the first segment if (1) the first segment is
        empty, (2) the sentence doesn't put the first segment over length or
        (3) 50% of the time when it does put the first segment over length
        """

        if random.random() < self.hparams.probability_single_sentence:
            first_segment_target_length = math.inf
        else:
            first_segment_target_length = (self.target_length - 3) // 2

        first_segment, second_segment = "", ""
        first_segment_length, second_segment_length = 0, 0

        # sentence is a string, sentence_len is the corresponding tokenizer length
        for sentence, sentence_len in zip(self.current_sentences, self.current_lengths):
            if (
                (first_segment_length == 0)
                or (first_segment_length + sentence_len < first_segment_target_length)
                or (
                    second_segment_length == 0
                    and first_segment_length < first_segment_target_length
                    and random.random() < self.hparams.probability_first_segment_over_length
                )
            ):
                first_segment += sentence
                first_segment_length += sentence_len
            else:
                second_segment += sentence
                second_segment_length += sentence_len

        # prepare to start building the next example
        self.current_sentences = []
        self.current_lengths = []

        # small chance for random-length instead of max_length-length example
        if random.random() < self.hparams.probability_random_length:
            self.target_length = random.randint(5, self.hparams.max_sequence_length)
        else:
            self.target_length = self.hparams.max_sequence_length

        return dict(first=first_segment, second=second_segment if second_segment else None)

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(ElectraPretrainingStrategy, ElectraPretrainingStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--tokenizer', required=True, type=str,
                            help="Name of the huggingface pre-trained tokenizer to use to tokenize the text.")
        parser.add_argument('--tokenizer_class', required=False, default=None, type=str,
                            help="Name of the huggingface pre-trained tokenizer class to use to load the tokenizer.")
        parser.add_argument('--max_sequence_length', type=int, required=True,
                            help="Max sequence length to fill sentence.")
        parser.add_argument(
            '--probability_random_length',
            default=0.05,
            required=False,
            type=float,
            help="Probability of creating a sample with a random length between 5 and `max_sequence_length`."
        )
        parser.add_argument('--probability_single_sentence', default=0.1, required=False, type=float,
                            help="Probability of creating a sentence with a single sentence.")
        parser.add_argument('--probability_first_segment_over_length', default=0.5, required=False, type=float,
                            help="Probability of creating a longer first sequence.")
        parser.add_argument('--min_sentence_length', type=int, default=6, required=False,
                            help="Minimum line length to consider (in characters)")
