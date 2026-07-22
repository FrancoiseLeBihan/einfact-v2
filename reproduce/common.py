"""Shared, deliberately small utilities for paper-29482 verification."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from einfact import NNEinFact

def add_common(p):
    p.add_argument('--data', required=True, help='float32 .npy tensor')
    p.add_argument('--heldout', help='boolean .npy; required for exact paper split')
    p.add_argument('--seed', type=int, default=29482)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--out', default='reproduce/report.json')

def load(args, expected=None):
    y=np.load(args.data, mmap_mode='r'); assert y.dtype==np.float32 and y.flags.c_contiguous
    if expected: assert y.shape == expected, (y.shape, expected)
    assert np.isfinite(y).all() and (y >= 0).all()
    if args.heldout:
        h=np.load(args.heldout, mmap_mode='r'); assert h.dtype==np.bool_ and h.shape==y.shape
    else:
        # Deterministic 5% split without a 5.8-GB uint64 index tensor.
        # Exact experimental heldout indices must override this fallback.
        flat=np.empty(y.size, dtype=np.bool_); block=10_000_000
        multiplier=np.uint64(11400714819323198485)
        for start in range(0, y.size, block):
            ind=np.arange(start, min(start+block,y.size), dtype=np.uint64)
            flat[start:start+ind.size]=((ind*multiplier + np.uint64(args.seed)) % 100) < 5
        h=flat.reshape(y.shape)
    return np.asarray(y), np.asarray(h)

def fit(y, heldout, model, shapes, alpha, beta, args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    m=NNEinFact(model, shapes, device='cuda', alpha=alpha, beta=beta)
    # NNEinFact's mask convention is observed=True.
    hist=m.fit(y, mask=~heldout, max_iter=args.iterations, verbose=False)
    return m, hist

def mse(y, pred, mask): return float(np.mean((y[mask]-pred[mask])**2))
def metadata(data):
    p=Path(data)
    return {'input_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'torch':torch.__version__,
      'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0), 'python':platform.python_version()}
def report(args, payload):
    payload['environment']=metadata(args.data); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2)); print(json.dumps(payload, indent=2))
