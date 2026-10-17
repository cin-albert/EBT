from types import SimpleNamespace

import pytest

from model.model_utils import model_sizes, setup_ebt, setup_transformer

PAPER_XXS_NON_EMBEDDING = 6.18e6  # Table D.1


def xxs_hparams(**overrides):
    size = model_sizes["xxs"]
    hparams = dict(
        embedding_dim=size["embedding_dim"],
        num_transformer_blocks=size["num_transformer_blocks"],
        multiheaded_attention_heads=size["multiheaded_attention_heads"],
        batch_size_per_device=32,
        context_length=256,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        ffn_dim_multiplier=1.0,
        ebt_type="time_embed",
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_num_steps=2,
    )
    hparams.update(overrides)
    return SimpleNamespace(**hparams)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("build", [setup_ebt, setup_transformer], ids=["ebt_time_embed", "transformer_pp"])
def test_xxs_backbone_matches_paper(build):
    n = n_params(build(xxs_hparams()))
    print(f"{build.__name__}: {n:,}")
    assert abs(n - PAPER_XXS_NON_EMBEDDING) / PAPER_XXS_NON_EMBEDDING < 0.01


def test_repo_default_ffn_multiplier_does_not_match_paper():
    n = n_params(setup_ebt(xxs_hparams(ffn_dim_multiplier=4.0)))
    assert n > 14e6
