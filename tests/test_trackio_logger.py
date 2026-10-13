import argparse

import torch

from utils.trackio_logger import TrackioLogger


class FakeTrackio:
    def __init__(self):
        self.inits, self.logs, self.finished = [], [], 0

    def init(self, project, name, config=None, resume=None):
        self.inits.append({"project": project, "name": name, "config": config, "resume": resume})
        return object()

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))

    def finish(self):
        self.finished += 1


def make_logger():
    fake = FakeTrackio()
    return TrackioLogger(project="p", name="run-1", trackio_module=fake), fake


def test_hyperparams_start_run_once_with_serializable_config():
    logger, fake = make_logger()
    params = argparse.Namespace(lr=0.0012, model_size="xxs", strategy=object())
    logger.log_hyperparams(params)
    logger.log_metrics({"train_loss": 1.0}, step=1)
    assert len(fake.inits) == 1
    config = fake.inits[0]["config"]
    assert config["lr"] == 0.0012 and config["model_size"] == "xxs"
    assert isinstance(config["strategy"], str)  # non-JSON values are stringified
    assert fake.inits[0]["resume"] == "allow"


def test_log_metrics_converts_tensors_and_keeps_step():
    logger, fake = make_logger()
    logger.log_metrics({"valid_loss": torch.tensor(2.5), "Global_LR": 1e-3}, step=15000)
    assert fake.logs == [({"valid_loss": 2.5, "Global_LR": 1e-3}, 15000)]


def test_experiment_log_is_forwarded():
    logger, fake = make_logger()
    logger.experiment.log({"x": 1})
    assert fake.logs == [({"x": 1}, None)]


def test_finalize_finishes_once():
    logger, fake = make_logger()
    logger.log_metrics({"a": 1.0}, step=0)
    logger.finalize("success")
    logger.finalize("success")
    assert fake.finished == 1


def test_name_and_version():
    logger, _ = make_logger()
    assert logger.name == "p" and logger.version == "run-1"
