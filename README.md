# Graph neural networks for molecules

Message passing neural networks for molecular property prediction, implemented
from first principles in PyTorch. The library builds molecular graphs from
SMILES with RDKit and provides three models under one training interface:

- **MPNN**: edge-conditioned message passing with a hand-written scatter
  aggregation and a GRU node update.
- **GCN**: a graph convolutional network with symmetric adjacency normalization
  and added self loops, computed directly on the edge list.
- **FingerprintMLP**: a Morgan-fingerprint multilayer perceptron, included as a
  non-graph baseline.

There is no dependency on a graph-learning framework such as PyTorch Geometric.
The scatter reductions, the message-passing convolution, and the graph
convolution are all written out in `src/gnn_molecules/layers.py` so the
mechanics are explicit and unit tested.

## What it does

- Featurizes atoms and bonds into numeric graphs (`featurize.py`).
- Batches variable-size graphs into one disjoint graph with a `batch` index
  vector, the standard trick for graph-level pooling (`data.py`).
- Trains with standardized targets and reports RMSE, MAE, and R^2 (`train.py`).
- Benchmarks the three models on the public ESOL solubility set or on a fully
  offline RDKit-descriptor target (`scripts/benchmark.py`).

## What it does not do

- It is not a general graph-learning framework; only molecular graph regression
  is implemented.
- The models are small and CPU-friendly. They are not tuned for
  state-of-the-art accuracy, and no pretrained weights are shipped.
- Only regression targets are supported out of the box.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Run

Fully offline (no download; target is an RDKit descriptor on a built-in SMILES
set):

```bash
python scripts/benchmark.py --dataset synthetic --epochs 40
```

On the public ESOL (Delaney) aqueous-solubility dataset:

```bash
python scripts/download_data.py --out data/esol.csv
python scripts/benchmark.py --dataset esol --csv data/esol.csv --epochs 40
```

`notebooks/demo.ipynb` is a short, executed walkthrough of the same workflow.

## Results

Measured on the public ESOL dataset (1128 molecules, random 80/10/10 split,
40 epochs, single CPU, seed 0). These numbers were produced by
`scripts/benchmark.py` in this repository, not copied from any prior source.

| Model          |   RMSE |    MAE |    R^2 |
|----------------|-------:|-------:|-------:|
| MPNN           | 0.7194 | 0.5654 | 0.8910 |
| GCN            | 0.8324 | 0.6915 | 0.8541 |
| FingerprintMLP | 1.1240 | 0.8709 | 0.7340 |

The message passing network, which uses bond features and learned node updates,
outperforms both the plain graph convolution and the fingerprint baseline on
solubility, as expected for this task. Training curves and a parity plot are
written to `results/` when you run the benchmark.

## Layout

```
src/gnn_molecules/   library: featurize, data, layers, models, train
scripts/             download_data.py, benchmark.py
notebooks/           demo.ipynb (executed)
tests/               pytest suite for featurization, scatter ops, models
data/                gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
