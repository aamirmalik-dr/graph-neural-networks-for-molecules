"""Tests for the config system, the ablation loop, and checkpoint round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from gnn_molecules.config import (
    AblationConfig,
    ExperimentConfig,
    load_ablation,
    load_config,
)
from gnn_molecules.experiment import build_model, run_ablation, run_experiment
from gnn_molecules.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from gnn_molecules.models import MPNN
from gnn_molecules.train import MetricsLogger, Trainer


def _synthetic_cfg(model: str = "mpnn") -> ExperimentConfig:
    return ExperimentConfig(
        name=f"test-{model}",
        model=model,
        dataset="synthetic",
        target="logp",
        epochs=5,
        batch_size=8,
        hidden_dim=16,
        num_layers=2,
        n_bits=256,
        patience=0,
        seed=0,
    )


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model: gcn\ndataset: synthetic\nnum_layers: 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model == "gcn"
    assert cfg.num_layers == 2
    assert cfg.dataset == "synthetic"


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"model": "mpnn", "bogus": 1})


def test_config_rejects_bad_model():
    with pytest.raises(ValueError, match="model must be one of"):
        ExperimentConfig(model="transformer")


def test_config_rejects_bad_split():
    with pytest.raises(ValueError, match="split"):
        ExperimentConfig(split=(0.5, 0.4, 0.4))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_ablation_expands_layers():
    base = _synthetic_cfg("mpnn")
    ablation = AblationConfig(base=base, num_layers=[1, 2, 3])
    cfgs = ablation.configs()
    assert [c.num_layers for c in cfgs] == [1, 2, 3]
    assert cfgs[0].name == "mpnn-L1"
    # The base config must be unchanged.
    assert base.num_layers == 2


def test_load_ablation_requires_num_layers(tmp_path):
    p = tmp_path / "ab.yaml"
    p.write_text("model: mpnn\ndataset: synthetic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="num_layers"):
        load_ablation(p)


def test_load_ablation_reads_sweep(tmp_path):
    p = tmp_path / "ab.yaml"
    p.write_text(
        "model: mpnn\ndataset: synthetic\nnum_layers: [1, 2]\nepochs: 3\n",
        encoding="utf-8",
    )
    ablation = load_ablation(p)
    assert ablation.num_layers == [1, 2]
    assert ablation.base.epochs == 3


def test_run_experiment_returns_metrics():
    result = run_experiment(_synthetic_cfg("mpnn"))
    assert set(result.metrics) == {"rmse", "mae", "r2"}
    assert result.y_true.shape == result.y_pred.shape
    assert len(result.train_curve) == 5


def test_run_experiment_fingerprint_has_no_trainer():
    result = run_experiment(_synthetic_cfg("fingerprint"))
    assert result.trainer is None
    assert result.metrics["rmse"] >= 0.0


def test_run_ablation_runs_each_point():
    base = _synthetic_cfg("mpnn")
    results = run_ablation(AblationConfig(base=base, num_layers=[1, 2]))
    assert len(results) == 2
    assert [r.config.num_layers for r in results] == [1, 2]


def test_checkpoint_roundtrip_preserves_predictions():
    from gnn_molecules.data import random_split, rdkit_property_dataset

    dataset = rdkit_property_dataset(target="logp")
    train_set, val_set, test_set = random_split(dataset, seed=0)
    model = MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2)
    trainer = Trainer(model)
    trainer.fit(train_set, val_set, epochs=5, batch_size=8, verbose=False)
    before = trainer.predict(test_set)

    ckpt = "checkpoint.pt"
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ckpt
        trainer.save(path)
        assert path.exists()
        fresh = Trainer(MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2))
        fresh.load(path)
        after = fresh.predict(test_set)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


def test_early_stopping_restores_best_and_stops_early():
    from gnn_molecules.data import random_split, rdkit_property_dataset

    dataset = rdkit_property_dataset(target="logp")
    train_set, val_set, _ = random_split(dataset, seed=0)
    trainer = Trainer(
        MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2),
        patience=3,
    )
    trainer.fit(train_set, val_set, epochs=200, batch_size=8, verbose=False)
    # Early stopping should trigger well before the 200 epoch cap.
    assert len(trainer.history["train"]) < 200
    assert trainer.best_epoch >= 1


def test_metrics_logger_curves_and_json(tmp_path):
    logger = MetricsLogger()
    logger.log(1, 0.5, 1.0)
    logger.log(2, 0.3, 0.8)
    assert logger.train_curve == [0.5, 0.3]
    assert logger.val_curve == [1.0, 0.8]
    out = tmp_path / "log.json"
    logger.to_json(out)
    assert out.exists()
    assert '"epoch"' in out.read_text(encoding="utf-8")


def test_build_model_dispatch():
    assert isinstance(build_model(_synthetic_cfg("mpnn")), MPNN)


def test_run_experiment_is_reproducible():
    # Two runs of the same config must produce identical metrics. This guards
    # the fixed-seed contract: a seeded shuffle generator and single-threaded
    # reductions make the benchmark bit-for-bit reproducible.
    first = run_experiment(_synthetic_cfg("mpnn"))
    second = run_experiment(_synthetic_cfg("mpnn"))
    assert first.metrics == second.metrics
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
