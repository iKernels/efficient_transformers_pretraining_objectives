import logging
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import transformers  # noqa: F401
from blingfire import text_to_sentences

from process_datasets.strategies.strategy import Strategy


class SplitPretrainingStrategy(Strategy):
    r""" Just split document such that average sequence length is about `max_sequence_length`. """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        assert self.hparams.max_sequence_length is not None and self.hparams.max_sequence_length >= 1, (
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

        text = " ".join(self.current_sentences)

        # prepare to start building the next example
        self.current_sentences = []
        self.current_lengths = []

        return dict(text=text)

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        super(SplitPretrainingStrategy, SplitPretrainingStrategy).add_arguments_to_argparse(parser)
        parser.add_argument('--tokenizer', required=True, type=str,
                            help="Name of the huggingface pre-trained tokenizer to use to tokenize the text.")
        parser.add_argument('--tokenizer_class', required=False, default=None, type=str,
                            help="Name of the huggingface pre-trained tokenizer class to use to load the tokenizer.")
        parser.add_argument('--max_sequence_length', type=int, required=True,
                            help="Max sequence length to fill example (in tokens).")
        parser.add_argument('--min_sentence_length', type=int, default=6, required=False,
                            help="Minimum line length to consider (in characters)")
