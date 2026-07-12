# Data

## esol_sample.csv (committed)

`esol_sample.csv` is a 300-molecule subset carved from the public ESOL
(Delaney) aqueous-solubility dataset. It holds two columns, `smiles` and
`measured log solubility in mols per litre`. The rows are a seeded random
sample (seed 0) of the full set. ESOL is public-domain reference data, so a
small subset is fine to commit. It exists so the quickstart and the benchmark
suite run offline with no download.

## Full ESOL (not committed)

The full dataset is 1128 molecules, distributed as `delaney-processed.csv` by
MoleculeNet. It is not committed. Fetch it with:

```bash
python scripts/download_data.py --out data/esol.csv
```

The script tries two public mirrors. If both are unreachable it writes a small
offline CSV instead (literal SMILES with an RDKit-computed logP proxy target),
so the pipeline still runs without network access. The script prints which
source was used.

To benchmark on the full set, point a config's `csv` field at `data/esol.csv`
and run `python scripts/benchmark.py --config <that config>`.

## Offline synthetic target

The unit tests and the `dataset: synthetic` config path need no download at
all. They build a dataset on the fly from a built-in list of valid SMILES and
compute an RDKit descriptor (logP, TPSA, or molecular weight) as the regression
target, which keeps continuous integration fully self-contained.
