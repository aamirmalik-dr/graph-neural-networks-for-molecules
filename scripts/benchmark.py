"""Run one configured model and report test-set metrics.

The run is described entirely by a YAML config (see ``configs/``). This is the
single-model entry point; ``scripts/run_all.py`` drives the full benchmark
suite and the depth ablation.

Usage:
    # Train the MPNN on the committed ESOL sample (offline):
    python scripts/benchmark.py --config configs/mpnn.yaml

    # Train and save a checkpoint for later inference:
    python scripts/benchmark.py --config configs/mpnn.yaml --save results/mpnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gnn_molecules.config import load_config
from gnn_molecules.experiment import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an experiment YAML")
    parser.add_argument("--out", default="results", help="directory for the metrics log")
    parser.add_argument("--save", default=None, help="optional checkpoint path (.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {cfg.name} ({cfg.model} on {cfg.dataset})")

    result = run_experiment(cfg)
    m = result.metrics
    print(
        f"Test metrics  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}"
        + (f"  (best epoch {result.best_epoch})" if result.best_epoch else "")
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from gnn_molecules.train import MetricsLogger

    logger = MetricsLogger()
    for i, (tr, va) in enumerate(zip(result.train_curve, result.val_curve, strict=False)):
        logger.log(i + 1, tr, va)
    logger.to_json(out_dir / f"training_log_{cfg.model}.json")

    if args.save and result.trainer is not None:
        result.trainer.save(args.save)
        print(f"Saved checkpoint to {args.save}")
    elif args.save:
        print("The fingerprint baseline has no graph checkpoint to save; skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
