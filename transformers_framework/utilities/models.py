import torch
from typing import Dict, List, Type

from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from transformers import ElectraConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


def read_clusters(filename):
    with open(filename, "r") as fi:
        return torch.tensor([int(line) for line in fi.readlines() if line], dtype=torch.int64)


def tie_or_clone_weights(output_embeddings, input_embeddings):
    r"""Tie or clone module weights and optionally biases if they are present. """
    if output_embeddings.weight.shape != input_embeddings.weight.shape:
        raise ValueError(
            f"Cannot tie weights, size mismatch: {output_embeddings.weight.shape} vs {input_embeddings.weight.shape}"
        )
    output_embeddings.weight = input_embeddings.weight

    if hasattr(output_embeddings, "bias") and hasattr(input_embeddings, "bias"):
        output_embeddings.bias = input_embeddings.bias

    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


def tie_weights_electra(
    generator, discriminator, tie_generator_discriminator_embeddings: bool, tie_word_embeddings: bool
):
    r"""
    Tie the weights between the generator and the discriminator embeddings.
    Electra paper says to link both the token and the positional embeddings, see Section 3.2
    """
    if tie_generator_discriminator_embeddings:
        # token embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.word_embeddings,
            generator.electra.embeddings.word_embeddings
        )

        # positional embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.position_embeddings,
            generator.electra.embeddings.position_embeddings
        )

        # token type embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.token_type_embeddings,
            generator.electra.embeddings.token_type_embeddings
        )

        # layernorm weights
        tie_or_clone_weights(
            discriminator.electra.embeddings.LayerNorm, generator.electra.embeddings.LayerNorm
        )

    # assert all weights are tied
    if tie_word_embeddings:
        assert generator.generator_lm_head.weight is generator.electra.embeddings.word_embeddings.weight

    if tie_generator_discriminator_embeddings:
        assert generator.generator_lm_head.weight is (
            discriminator.electra.embeddings.word_embeddings.weight
        )
        assert generator.electra.embeddings.word_embeddings.weight is (
            discriminator.electra.embeddings.word_embeddings.weight
        )
        assert generator.electra.embeddings.position_embeddings.weight is (
            discriminator.electra.embeddings.position_embeddings.weight
        )
        assert generator.electra.embeddings.token_type_embeddings.weight is (
            discriminator.electra.embeddings.token_type_embeddings.weight
        )
        assert generator.electra.embeddings.LayerNorm.weight is (
            discriminator.electra.embeddings.LayerNorm.weight
        )


def tie_weights_deberta(generator, discriminator, tie_generator_discriminator_embeddings: bool):
    r""" DeBERTa does not directly tie the embedding weights of the generator and the discriminator. """

    # token embeddings
    if tie_generator_discriminator_embeddings:
        tie_or_clone_weights(
            discriminator.deberta.embeddings.word_embeddings,
            generator.deberta.embeddings.word_embeddings
        )

        assert generator.get_output_embeddings().weight is (
            discriminator.deberta.embeddings.word_embeddings.weight
        )
        assert generator.deberta.embeddings.word_embeddings.weight is (
            discriminator.deberta.embeddings.word_embeddings.weight
        )

        # positional embeddings
        if (
            discriminator.deberta.embeddings.position_embeddings is not None
            and generator.deberta.embeddings.position_embeddings is not None
        ):
            tie_or_clone_weights(
                discriminator.deberta.embeddings.position_embeddings,
                generator.deberta.embeddings.position_embeddings
            )

            assert generator.deberta.embeddings.position_embeddings.weight is (
                discriminator.deberta.embeddings.position_embeddings.weight
            )
            
        if (
            hasattr(discriminator.deberta.embeddings, "token_type_embeddings")
            and hasattr(generator.deberta.embeddings, "token_type_embeddings")
        ):
            # token type embeddings
            tie_or_clone_weights(
                discriminator.deberta.embeddings.token_type_embeddings,
                generator.deberta.embeddings.token_type_embeddings
            )

            assert generator.deberta.embeddings.token_type_embeddings.weight is (
                discriminator.deberta.embeddings.token_type_embeddings.weight
            )

        # layernorm weights
        tie_or_clone_weights(
            discriminator.deberta.embeddings.LayerNorm,
            generator.deberta.embeddings.LayerNorm
        )

        assert generator.deberta.embeddings.LayerNorm.weight is (
            discriminator.deberta.embeddings.LayerNorm.weight
        )

        if (
            hasattr(discriminator.deberta.embeddings, "embed_proj")
            and hasattr(generator.deberta.embeddings, "embed_proj")
        ):
            # proj weights
            tie_or_clone_weights(
                discriminator.deberta.embeddings.embed_proj,
                generator.deberta.embeddings.embed_proj
            )
        
            assert generator.deberta.embeddings.embed_proj.weight is (
                discriminator.deberta.embeddings.embed_proj.weight
            )


def get_electra_reduced_generator_config(discriminator_config: ElectraConfig, factor: float = 1 / 3, **kwargs):
    r""" Created reduced configuration for electra generator. """
    params = {
        **vars(discriminator_config),
        'hidden_size': int(discriminator_config.hidden_size * factor),
        'num_attention_heads': int(discriminator_config.num_attention_heads * factor),
        'intermediate_size': int(discriminator_config.intermediate_size * factor),
        **kwargs,
    }
    return ElectraConfig(**params)


def init_mismatched_keys(model: PreTrainedModel, info: Dict[str, List]):
    r"""
    Initialize the modules that were not initialized while loading the ckpt because of wrong shape.
    Args:
        model: transformers model that was initialized from checkpoint
        info: dict returned by `model.from_pretrained(..., output_loading_info=True)`
    """

    loaded_keys = [k[0] for k in info['mismatched_keys']]

    expected_keys = list(model.state_dict().keys())
    prefix = model.base_model_prefix

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    modules = model.retrieve_modules_from_names(
        loaded_keys,
        add_prefix=add_prefix_to_model,
        remove_prefix=remove_prefix_from_model
    )

    for (key, old_size, new_size), module in zip(info['mismatched_keys'], modules, ):
        rank_zero_warn(f'Initializing mismatched weights {key}, size {old_size} -> {new_size}')
        model._init_weights(module)


def load_pretrained_safe(
    model_class: Type, name_or_path: str, config: PretrainedConfig = None, **kwargs
) -> PreTrainedModel:
    r""" Load a model in safe environment, taking care of initializing eventual parameters
    that were not loaded from the checkpoint. """

    model, info = model_class.from_pretrained(
        name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **kwargs,
    )
    init_mismatched_keys(model, info)
    return model


def load_model(model_class: Type, name_or_path: str = None, config: PretrainedConfig = None, **kwargs):
    r""" Load a model in safe environment, taking care of initializing eventual parameters
    that were not loaded from the checkpoint. If the `name_or_path` is None, the model will be
    created safely from scratch. """

    if name_or_path is not None:
        model = load_pretrained_safe(model_class, name_or_path, config=config, **kwargs)
    else:
        rank_zero_warn(f"Model {model_class.__name__} loaded from scratch and not from a pretrained model.")
        model = model_class(config, **kwargs)

    return model
