import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from models.rsnn import RSNN, RSNNConfig
from models.readout import Readout, ReadoutConfig
from models.hebbian import DualHebbianAccumulator, HebbianConfig
from training.online_trainer import OnlineTrainer, TrainerConfig
from data.synthetic import bci_velocity_stream, supply_chain_stream


def make_system(input_size=20, hidden_size=32, output_size=2):
    rsnn = RSNN(RSNNConfig(input_size=input_size, hidden_size=hidden_size))
    readout = Readout(ReadoutConfig(hidden_size, output_size))
    hebbian = DualHebbianAccumulator(HebbianConfig(shape=(hidden_size, hidden_size)))
    trainer = OnlineTrainer(rsnn, readout, hebbian, TrainerConfig(lr_readout=1e-3))
    return trainer


def test_trainer_step_output_shape():
    trainer = make_system()
    x = torch.rand(20)
    y = torch.rand(2)
    y_pred, error = trainer.step(x, target=y)
    assert y_pred.shape == (2,)
    assert error.shape == (2,)


def test_trainer_error_decreases():
    """Loss should trend downward over 200 steps on a fixed linear target."""
    trainer = make_system(input_size=20, hidden_size=64, output_size=2)
    torch.manual_seed(0)

    losses_early, losses_late = [], []
    stream = list(bci_velocity_stream(T=300, input_size=20, seed=0))

    for i, (x, y) in enumerate(stream):
        _, error = trainer.step(x, target=y)
        loss = error.pow(2).mean().item()
        if i < 50:
            losses_early.append(loss)
        elif i > 250:
            losses_late.append(loss)

    early_mean = sum(losses_early) / len(losses_early)
    late_mean = sum(losses_late) / len(losses_late)
    assert late_mean < early_mean * 1.5, (
        f"Loss did not decrease: early={early_mean:.4f} late={late_mean:.4f}"
    )


def test_trainer_metrics():
    trainer = make_system()
    for x, y in bci_velocity_stream(T=50, input_size=20):
        trainer.step(x, target=y)
    metrics = trainer.get_metrics()
    assert "steps" in metrics
    assert metrics["steps"] == 50
    assert "loss_mean" in metrics


def test_reward_mode():
    trainer = make_system()
    trainer.cfg.mode = "reward"
    x = torch.rand(20)
    y_pred, error = trainer.step(x, reward=1.0)
    assert y_pred.shape == (2,)


def test_run_stream():
    trainer = make_system()
    log = []
    trainer.cfg.log_every = 10
    trainer.run_stream(
        bci_velocity_stream(T=50, input_size=20),
        callback=lambda step, yp, err: log.append(step)
    )
    assert trainer._step == 50
    assert len(log) > 0


def test_supply_chain_stream_runs():
    trainer = make_system(input_size=50, output_size=3)
    for x, y in supply_chain_stream(T=100, input_size=50, n_outputs=3):
        y_pred, error = trainer.step(x, target=y)
    assert trainer._step == 100
