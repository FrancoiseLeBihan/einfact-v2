# Reproduction checks for ICML 2026 paper #29482

These are **verification scripts**, not a claim that a result has been reproduced
until the required raw data, exact experimental split, and specified hardware are
used. Every script emits a machine-readable JSON report and exits nonzero when its
pre-registered checks fail.  They deliberately avoid materialising a dense
train/validation mask: a seeded hash predicate is evaluated on-device in chunks.

## Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Run from repository root. Use the paper revision named in the request (arXiv
2602.02759v2); record `git rev-parse HEAD`, PyTorch, CUDA, GPU, and input SHA-256
in each report. `--help` lists all controls. Results are stochastic: use the
stated seeds and do not relax a tolerance after viewing a result.

## Exact hardware to use

| Script | Required hardware | Why |
|---|---|---|
| `claim1_theorem.py` | CPU: AMD EPYC 7543P (32 cores), 128 GB RAM | Tiny synthetic tensors; GPU adds no evidential value. |
| `claim2_uber_table1.py` | **2× NVIDIA Tesla T4 (16 GB each)**, 32+ CPU cores, 128 GB RAM, CUDA 12.x | Exact i-mode sharding holds only 50/100 origin cells per GPU; update statistics are aggregated before each factor update. |
| `claim3_speed.py` | **1× NVIDIA A100 80 GB PCIe**, 32+ CPU cores, 128 GB RAM, CUDA 12.x | Fair wall-clock comparison on six divergences; synchronised GPU timing. |
| `claim4_models.py` | **1× NVIDIA A100 80 GB PCIe**, 32+ CPU cores, **256 GB RAM**, CUDA 12.x | ICEWS/Uber/WITS sweeps and model contractions. |
| `claim5_interpretability.py` | **1× NVIDIA A100 80 GB PCIe**, 32+ CPU cores, 128 GB RAM, CUDA 12.x | Fits/extracts Uber spatial-temporal components. |

No GPU is available in this agent environment, so the GPU scripts are supplied
but not executed here. A100 is specified rather than a consumer GPU to make the
claimed timing comparison meaningful and to provide a safe memory margin.

## Data contract

Place each raw tensor as a `.npy` float32 C-contiguous array, respectively:
`DATA/uber.npy` shape `(27,7,24,100,100)`, `DATA/icews.npy` shape
`(249,249,20,228)`, and `DATA/wits.npy` shape `(196,196,96,29)`. Values must be
nonnegative. The paper/repository does not include these raw data or the original
heldout indices; scripts use a documented deterministic 5% hash split unless an
`--heldout` boolean `.npy` is supplied. To reproduce exact paper numbers, supply
the authors' split with `--heldout`.

For Claim 2 specifically, use the authors' bundled `data/Y.npz` artifact with
the Table-1 runner. It contains the canonical Uber tensor from `demo.ipynb`;
the locally rebuilt `DATA/uber.npy` is not byte-equivalent after spatial binning.
The Table-1 model is `R=10, K=6` (12,580 parameters) and its comparable CP
baseline is rank 49 (12,642 parameters). The 48,580-parameter count refers to
the separate 400×400 qualitative figure, not the 100×100 Table-1 experiment.

Run the full Table-1 check with:

```bash
python reproduce/claim2_uber_table1.py --data data/Y.npz
```

The published 48,580-parameter architecture/ranks are not encoded in the public
quick-start source. Claim 5 therefore requires `--custom-model`, `--shapes`, and
ranks from the paper authors' experiment configuration; the script independently
counts and asserts exactly 48,580 parameters rather than guessing them.
