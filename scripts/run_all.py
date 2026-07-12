"""Run the full benchmark suite and the depth ablation, then write artifacts.

Trains the MPNN, GCN, and fingerprint baseline from their configs on the same
seeded split, runs a message-passing-depth ablation for the MPNN, and writes:

* ``results/metrics.json``      the benchmark table
* ``results/parity.png``        MPNN predicted vs measured logS on the test set
* ``results/training_curves.png`` training loss for the three models
* ``results/ablation.png``      test RMSE vs number of message-passing layers
* ``results/mpnn.pt``           the trained MPNN checkpoint
* ``RESULTS.md``                the full table plus the ablation

Usage:
    python scripts/run_all.py                 # uses configs/*.yaml, writes results/
    python scripts/run_all.py --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_molecules.config import load_ablation, load_config
from gnn_molecules.experiment import ExperimentResult, run_ablation, run_experiment

MODEL_CONFIGS = ["configs/mpnn.yaml", "configs/gcn.yaml", "configs/fingerprint.yaml"]
ABLATION_CONFIG = "configs/ablation_depth.yaml"


def _plot_training_curves(results: dict[str, ExperimentResult], path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for name, res in results.items():
        plt.plot(res.train_curve, label=name)
    plt.xlabel("epoch")
    plt.ylabel("training loss (standardized MSE)")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_parity(res: ExperimentResult, path: Path) -> None:
    y_true, y_pred = res.y_true, res.y_pred
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    plt.xlabel("measured logS")
    plt.ylabel("predicted logS")
    r2 = res.metrics["r2"]
    rmse = res.metrics["rmse"]
    plt.title(f"MPNN parity (test)  R2={r2:.3f}  RMSE={rmse:.3f}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_ablation(results: list[ExperimentResult], path: Path) -> None:
    layers = [r.config.num_layers for r in results]
    rmses = [r.metrics["rmse"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(layers, rmses, marker="o")
    plt.xticks(layers)
    plt.xlabel("message-passing layers")
    plt.ylabel("test RMSE")
    plt.title("MPNN depth ablation")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _write_results_md(
    results: dict[str, ExperimentResult],
    ablation: list[ExperimentResult],
    sample_size: int,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Results")
    lines.append("")
    lines.append(
        f"All numbers below were produced by `scripts/run_all.py` on the "
        f"committed ESOL sample ({sample_size} molecules, seed 0, "
        f"80/10/10 split, early stopping on validation RMSE)."
    )
    lines.append("")
    lines.append("## Benchmark")
    lines.append("")
    lines.append("| Model | RMSE | MAE | R2 | Best epoch |")
    lines.append("|-------|-----:|----:|---:|-----------:|")
    for name, res in results.items():
        m = res.metrics
        be = res.best_epoch or len(res.train_curve)
        lines.append(
            f"| {name} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {be} |"
        )
    lines.append("")
    lines.append("![MPNN parity](results/parity.png)")
    lines.append("")
    lines.append("## Depth ablation (MPNN)")
    lines.append("")
    lines.append("Same MPNN, only the number of message-passing layers changes.")
    lines.append("")
    lines.append("| Layers | RMSE | MAE | R2 |")
    lines.append("|-------:|-----:|----:|---:|")
    for res in ablation:
        m = res.metrics
        lines.append(
            f"| {res.config.num_layers} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} |"
        )
    lines.append("")
    lines.append("![Depth ablation](results/ablation.png)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, ExperimentResult] = {}
    for cfg_path in MODEL_CONFIGS:
        cfg = load_config(cfg_path)
        print(f"Running {cfg.name} ...")
        res = run_experiment(cfg)
        results[cfg.name] = res
        m = res.metrics
        print(f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    from gnn_molecules.experiment import build_dataset

    sample_size = len(build_dataset(load_config("configs/mpnn.yaml")))

    metrics_out = {name: res.metrics for name, res in results.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    _plot_training_curves(results, out_dir / "training_curves.png")
    _plot_parity(results["MPNN"], out_dir / "parity.png")

    # Per-epoch training log for the headline model.
    from gnn_molecules.train import MetricsLogger

    mpnn_log = MetricsLogger()
    for i, (tr, va) in enumerate(
        zip(results["MPNN"].train_curve, results["MPNN"].val_curve, strict=False)
    ):
        mpnn_log.log(i + 1, tr, va)
    mpnn_log.to_json(out_dir / "training_log_mpnn.json")

    # Save the trained MPNN for instant inference.
    mpnn_res = results["MPNN"]
    if mpnn_res.trainer is not None:
        mpnn_res.trainer.save(out_dir / "mpnn.pt")
        print(f"Saved MPNN checkpoint to {out_dir / 'mpnn.pt'}")

    print("Running depth ablation ...")
    ablation = run_ablation(load_ablation(ABLATION_CONFIG))
    for res in ablation:
        print(f"  layers={res.config.num_layers}  RMSE={res.metrics['rmse']:.4f}")
    _plot_ablation(ablation, out_dir / "ablation.png")

    _write_results_md(results, ablation, sample_size, Path("RESULTS.md"))
    print(f"Wrote metrics, figures, checkpoint to {out_dir}/ and RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
