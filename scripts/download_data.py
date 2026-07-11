"""Download the public ESOL (Delaney) aqueous-solubility dataset.

The dataset is the widely mirrored ``delaney-processed.csv`` from MoleculeNet.
If the download fails (no network, mirror down), the script falls back to
writing a small offline CSV whose target is an RDKit-computed logP for a set of
literal SMILES, so the rest of the pipeline still runs. The choice is printed so
it is never silent.

Usage:
    python scripts/download_data.py --out data/esol.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests
from rdkit import Chem

from gnn_molecules.data import SAMPLE_SMILES, TARGET_FUNCTIONS

ESOL_URLS = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
]


def download_esol(out_path: Path) -> bool:
    """Try each mirror in turn; return True on the first success."""
    for url in ESOL_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any network error triggers fallback
            print(f"  mirror failed ({url}): {exc}")
            continue
        out_path.write_bytes(resp.content)
        print(f"Downloaded ESOL from {url} -> {out_path}")
        return True
    return False


def write_offline_csv(out_path: Path) -> None:
    """Write an offline CSV of SMILES with an RDKit logP target."""
    logp = TARGET_FUNCTIONS["logp"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "measured log solubility in mols per litre"])
        for smi in SAMPLE_SMILES:
            mol = Chem.MolFromSmiles(smi)
            # Use negative logP as a rough solubility proxy for the offline set.
            writer.writerow([smi, f"{-float(logp(mol)):.4f}"])
    print(
        f"Wrote offline fallback dataset ({len(SAMPLE_SMILES)} molecules, "
        f"RDKit logP proxy target) -> {out_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/esol.csv", help="output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not download_esol(out_path):
        print("All ESOL mirrors failed; using offline RDKit-computed fallback.")
        write_offline_csv(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
