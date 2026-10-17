"""Minimal PyTorch Lightning logger backed by trackio (replaces WandbLogger in train_model.py)."""
import argparse
import inspect
import json

import torch
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


def _to_config(params):
    if isinstance(params, argparse.Namespace):
        params = vars(params)
    config = {}
    for key, value in dict(params).items():
        try:
            json.dumps(value)
            config[key] = value
        except (TypeError, ValueError):
            config[key] = str(value)
    return config


class _Experiment:
    """What base_model_trainer expects from logger.experiment: a .log(dict) method."""

    def __init__(self, trackio_module):
        self._trackio = trackio_module

    def log(self, metrics, step=None):
        self._trackio.log(dict(metrics), step=step)


class TrackioLogger(Logger):
    def __init__(self, project, name, trackio_module=None):
        super().__init__()
        self._project = project
        self._run_name = name
        self._trackio = trackio_module
        self._run = None

    def _module(self):
        if self._trackio is None:
            import trackio
            self._trackio = trackio
        return self._trackio

    def _ensure_run(self, config=None):
        if self._run is None:
            module = self._module()
            kwargs = {"project": self._project, "name": self._run_name}
            if config is not None:
                kwargs["config"] = config
            if "resume" in inspect.signature(module.init).parameters:
                kwargs["resume"] = "allow"  # resuming from last.ckpt keeps writing to the same run
            self._run = module.init(**kwargs)
        return self._run

    @property
    def name(self):
        return self._project

    @property
    def version(self):
        return self._run_name

    @property
    @rank_zero_experiment
    def experiment(self):
        self._ensure_run()
        return _Experiment(self._module())

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        self._ensure_run(_to_config(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self._ensure_run()
        clean = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        self._module().log(clean, step=step)

    @rank_zero_only
    def finalize(self, status):
        if self._run is not None:
            self._module().finish()
            self._run = None
