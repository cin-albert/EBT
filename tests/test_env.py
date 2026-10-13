import importlib.metadata as md

import pytest

PINS = {
    "torch": "2.4.0",
    "pytorch-lightning": "2.4.0",
    "transformers": "4.44.2",
    "tokenizers": "0.19.1",
    "datasets": "3.0.0",
    "huggingface-hub": "0.36.2",
    "trackio": "0.20.2",
}


@pytest.mark.parametrize("dist,version", PINS.items())
def test_pinned_versions(dist, version):
    assert md.version(dist).startswith(version)  # torch reports e.g. 2.4.0+cu121


def test_training_stack_imports():
    import diffusers  # noqa: F401  (model/model_utils.py imports AutoencoderKL)
    import pytorch_lightning  # noqa: F401
    import trackio  # noqa: F401
    import transformers  # noqa: F401
    import base_model_trainer  # noqa: F401  (whole import graph used by train_model.py)


def test_cuda_available():
    import torch
    assert torch.cuda.is_available()
