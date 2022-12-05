from argparse import Action, ArgumentError, ArgumentParser
from typing import List

from pytorch_lightning.utilities.rank_zero import rank_zero_warn


class FlexibleArgumentParser(ArgumentParser):

    def add_argument(self, *args, exist_ok: bool = True, replace: bool = False, **kwargs) -> Action:
        if exist_ok:
            try:
                return super().add_argument(*args, **kwargs)
            except ArgumentError as e:
                rank_zero_warn(f"Argument {e.argument_name} was define twice, make sure this is intended...")
                if replace:
                    self.remove_options(args)
                    return super().add_argument(*args, **kwargs)
        else:
            return super().add_argument(*args, **kwargs)

    def remove_options(self, options: List[str]):
        for option in options:
            for action in self._actions:
                if vars(action)['option_strings'][0] == option:
                    self._handle_conflict_resolve(None, [(option, action)])
                    break


def is_already_defined_in_argparse(parser: ArgumentParser, name: str) -> bool:
    r""" Check if argument `name` has already been defined in parser. """
    for action in parser._actions:
        if name == action.dest:
            return True
    return False


def add_answer_sentence_selection_arguments(parser: ArgumentParser):
    r""" Add default AS2 arguments. """
    parser.add_argument(
        '--train_metrics_empty_target_action',
        choices=('skip', 'neg', 'pos', 'error'),
        default='skip',
        required=False,
        help="Empty target action for train metrics",
    )
    parser.add_argument(
        '--val_metrics_empty_target_action',
        choices=('skip', 'neg', 'pos', 'error'),
        default='skip',
        required=False,
        help="Empty target action for validation metrics",
    )
    parser.add_argument(
        '--test_metrics_empty_target_action',
        choices=('skip', 'neg', 'pos', 'error'),
        default='skip',
        required=False,
        help="Empty target action for test metrics",
    )


def add_machine_reading_arguments(parser: ArgumentParser):
    r""" Add default MR arguments. """


def add_token_detection_arguments(parser: ArgumentParser):
    r""" Add default token detection arguments. """
    parser.add_argument('--td_weight', type=float, default=50.0)


def add_masked_language_modeling_and_token_detection_arguments(parser: ArgumentParser):
    r""" Add MLM and TD arguments. """
    parser.add_argument('--sample_function', type=str, default='gumbel', choices=['gumbel', 'multinomial'])
    parser.add_argument('--pre_trained_generator_config', type=str, default=None)
    parser.add_argument('--pre_trained_generator_model', type=str, default=None)
    parser.add_argument('--tie_generator_discriminator_embeddings', action="store_true")
    parser.add_argument('--generator_size', type=float, default=1 / 2)
