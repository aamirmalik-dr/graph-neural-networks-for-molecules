# Data

This directory is gitignored. No datasets are committed to the repository.

## ESOL (Delaney) aqueous solubility

The benchmark uses the public ESOL dataset (1128 molecules with measured log
aqueous solubility), distributed as `delaney-processed.csv` by MoleculeNet.

Fetch it with:

```bash
python scripts/download_data.py --out data/esol.csv
```

The script tries two public mirrors. If both are unreachable it writes a small
offline CSV instead: a set of literal SMILES strings with an RDKit-computed
logP proxy target, so the pipeline still runs without network access. The script
prints which source was used.

## Offline / synthetic target

The unit tests and the `--dataset synthetic` benchmark path need no download at
all. They build a dataset on the fly from a built-in list of valid SMILES and
compute an RDKit descriptor (logP, TPSA, or molecular weight) as the regression
target. This keeps continuous integration fully self-contained.
