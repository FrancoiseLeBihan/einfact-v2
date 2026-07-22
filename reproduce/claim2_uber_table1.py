#!/usr/bin/env python3
"""Reproduce the Uber alpha=1.0 row of Table 1.

Table 1 uses the 27 x 7 x 24 x 100 x 100 Uber tensor.  Its custom model is
``wr,hr,dr,ikr,jkr->whdij`` with R=10 temporal classes and K=6 spatial
components per temporal class (12,580 parameters).  A rank-49 CP model has a
comparable 12,642 parameters.  The 48,580-parameter R=10, K=6 configuration
described later in the paper is *only* for the qualitative 400 x 400 grid; it
must not be projected back to a 100 x 100 grid by changing K to 24.

The paper reports means over ten independent 90/10 train--heldout splits,
with 5% of each training split reserved for validation stopping.  The paper
does not publish its masks, so the default uses ten deterministic splits from
``--seed`` and labels that fact in the JSON report.  Supplying ``--heldout``
instead runs one explicitly supplied split (use ``--splits 1``); it is useful
for diagnostics but is not a Table-1 mean.

The i (origin-cell) mode is split across two CUDA GPUs.  For every factor,
the numerator and denominator are aggregated over both shards before updating
the factor, exactly matching the full-batch multiplicative update apart from
floating-point summation order.
"""
import argparse
import concurrent.futures
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import add_common, report
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from einfact import swap


UBER_SHAPE = (27, 7, 24, 100, 100)
CUSTOM_RANK = 10
CUSTOM_SPATIAL_RANK = 6
CP_RANK = 49
HELDOUT_FRACTION = 0.10
VALIDATION_FRACTION = 0.05


def parameter_count(model, shapes):
    return sum(int(np.prod([shapes[index] for index in term]))
               for term in model.split('->', 1)[0].split(','))


def load_uber(path):
    """Load either the paper's bundled ``Y.npz`` or a matching ``.npy`` tensor."""
    loaded = np.load(path, mmap_mode='r')
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if 'Y' not in loaded.files:
            raise ValueError(f'{path} is missing the paper tensor named Y')
        # The authors' demo stores the same tensor as (week, hour, day, i, j).
        # Table 1 names its axes (week, day, hour, i, j), which is the convention
        # used by this runner and by the paper's Section 6.1.
        y = np.ascontiguousarray(loaded['Y'].transpose(0, 2, 1, 3, 4), dtype=np.float32)
    else:
        y = loaded
        if y.dtype != np.float32 or not y.flags.c_contiguous:
            raise ValueError('--data must be a C-contiguous float32 .npy array')
    if y.shape != UBER_SHAPE:
        raise ValueError(f'expected Uber shape {UBER_SHAPE}, got {y.shape}')
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Uber counts must be finite and nonnegative')
    return np.asarray(y)


def load_heldout(path, shape):
    heldout = np.load(path, mmap_mode='r')
    if heldout.dtype != np.bool_ or heldout.shape != shape:
        raise ValueError('--heldout must be a boolean .npy array matching --data')
    return np.asarray(heldout)


def random_mask(shape, fraction, rng):
    """Make a Boolean mask without materialising a multi-gigabyte float array."""
    mask = np.empty(shape, dtype=np.bool_)
    flat = mask.reshape(-1)
    block = 10_000_000
    for start in range(0, flat.size, block):
        stop = min(start + block, flat.size)
        flat[start:stop] = rng.random_sample(stop - start) < fraction
    return mask


def generated_split(shape, seed):
    """A 90/10 split and an independent 5% validation draw from its training set."""
    rng = np.random.RandomState(seed)
    heldout = random_mask(shape, HELDOUT_FRACTION, rng)
    validation = random_mask(shape, VALIDATION_FRACTION, rng)
    validation &= ~heldout
    return heldout, validation


def provided_split(heldout, seed):
    """Pair a supplied heldout mask with the paper's independent validation draw."""
    validation = random_mask(heldout.shape, VALIDATION_FRACTION,
                             np.random.RandomState(seed))
    validation &= ~heldout
    return heldout, validation


class TwoT4MU:
    """KL multiplicative updates with exact two-way sharding over the i mode."""

    def __init__(self, y, heldout, validation, model, shapes, seed):
        if torch.cuda.device_count() < 2:
            raise RuntimeError('Claim 2 requires two CUDA GPUs.')
        self.model = model
        self.terms = model.split('->', 1)[0].split(',')
        self.out = model.split('->', 1)[1]
        self.devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        self.i_axis = self.out.index('i')
        if y.shape[self.i_axis] % len(self.devices):
            raise ValueError('the i mode must split evenly across GPUs')

        rng = np.random.RandomState(seed)
        self.cpu = [rng.uniform(0.0, 1.0, [shapes[c] for c in term]).astype(np.float32)
                    for term in self.terms]
        width = y.shape[self.i_axis] // len(self.devices)
        self.slices = [slice(n * width, (n + 1) * width) for n in range(len(self.devices))]
        self.y, self.train, self.validation, self.heldout = [], [], [], []
        for device, shard_slice in zip(self.devices, self.slices):
            key = [slice(None)] * y.ndim
            key[self.i_axis] = shard_slice
            key = tuple(key)
            train = ~(heldout[key] | validation[key])
            self.y.append(torch.as_tensor(np.ascontiguousarray(y[key]), device=device))
            self.train.append(torch.as_tensor(np.ascontiguousarray(train), device=device))
            self.validation.append(torch.as_tensor(
                np.ascontiguousarray(validation[key]), device=device))
            self.heldout.append(torch.as_tensor(
                np.ascontiguousarray(heldout[key]), device=device))
        self.equations = [swap(model, n) for n in range(len(self.terms))]
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _parameters(self, shard):
        params = [torch.as_tensor(x, device=self.devices[shard]) for x in self.cpu]
        # Both Table-1 models place the i factor fourth.
        params[3] = params[3][self.slices[shard]]
        return params

    def _update_shard(self, shard, which):
        device = self.devices[shard]
        with torch.cuda.device(device):
            params = self._parameters(shard)
            prediction = torch.einsum(self.model, *params).clamp_min_(1e-10)
            # alpha=1, beta=0: a(x, y)=x/y and b(x, y)=1.
            numerator_data = (self.y[shard] / prediction) * self.train[shard]
            denominator_data = self.train[shard].float()
            other_params = params[:which] + params[which + 1:]
            numerator = torch.einsum(self.equations[which], *other_params, numerator_data)
            denominator = torch.einsum(
                self.equations[which], *other_params, denominator_data).clamp_min_(1e-10)
            torch.cuda.synchronize(device)
            return numerator.cpu(), denominator.cpu()

    def step(self):
        for factor in range(len(self.cpu)):
            shards = list(self.pool.map(
                lambda shard: self._update_shard(shard, factor), range(len(self.devices))))
            if factor == 3:  # i is disjoint across the two shards.
                numerator = torch.cat([item[0] for item in shards], dim=0)
                denominator = torch.cat([item[1] for item in shards], dim=0)
            else:
                numerator = sum(item[0] for item in shards)
                denominator = sum(item[1] for item in shards)
            ratio = np.clip((numerator / denominator).numpy(), 1e-10, 1e10)
            self.cpu[factor] *= ratio

    @staticmethod
    def _kl_sum(y, prediction):
        positive = y > 0
        return (prediction.sum() - y.sum()
                + (y[positive] * torch.log(y[positive] / prediction[positive])).sum())

    def divergence(self, masks):
        def one(shard):
            device = self.devices[shard]
            with torch.cuda.device(device):
                prediction = torch.einsum(
                    self.model, *self._parameters(shard)).clamp_min_(1e-10)
                mask = masks[shard]
                divergence = self._kl_sum(self.y[shard][mask], prediction[mask])
                count = mask.sum()
                torch.cuda.synchronize(device)
                return divergence.cpu(), count.cpu()

        values = list(self.pool.map(one, range(len(self.devices))))
        return float(sum(item[0] for item in values) / sum(item[1] for item in values))

    def fit(self, max_iterations):
        start = time.perf_counter()
        train_history, validation_history, heldout_history = [], [], []
        validation_increases = 0
        stop_reason = 'max_iterations'
        try:
            for iteration in range(max_iterations):
                self.step()
                train = self.divergence(self.train)
                validation = self.divergence(self.validation)
                heldout = self.divergence(self.heldout)
                train_history.append(train)
                validation_history.append(validation)
                heldout_history.append(heldout)

                if len(validation_history) > 1 and validation > validation_history[-2]:
                    validation_increases += 1
                else:
                    validation_increases = 0
                if validation_increases >= 5:
                    stop_reason = 'five_validation_increases'
                    break
                # Match the upstream implementation: only consider a small
                # training-loss change after its initial 100-iteration phase.
                if len(train_history) > 100 and abs(train - train_history[-2]) < 1e-6:
                    stop_reason = 'training_delta_below_1e-6'
                    break
        finally:
            self.pool.shutdown(wait=True)
        return {
            'heldout': heldout_history[-1],
            'iterations': len(heldout_history),
            'seconds': time.perf_counter() - start,
            'stop_reason': stop_reason,
        }


def run_model(y, heldout, validation, model, shapes, seed, max_iterations):
    return TwoT4MU(y, heldout, validation, model, shapes, seed).fit(max_iterations)


def mean_and_se(values):
    values = np.asarray(values, dtype=np.float64)
    se = 0.0 if len(values) == 1 else float(values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean()), se


def main():
    parser = argparse.ArgumentParser()
    add_common(parser)
    parser.set_defaults(iterations=5000)
    parser.add_argument('--splits', type=int, default=10,
                        help='number of deterministic 90/10 splits (Table 1 uses 10)')
    parser.add_argument('--custom-rank', type=int, default=CUSTOM_RANK)
    parser.add_argument('--spatial-rank', type=int, default=CUSTOM_SPATIAL_RANK)
    parser.add_argument('--cp-rank', type=int, default=CP_RANK)
    args = parser.parse_args()
    if args.splits < 1:
        parser.error('--splits must be positive')
    if args.heldout and args.splits != 1:
        parser.error('--heldout supplies one split; use --splits 1, or omit --heldout for Table 1')

    y = load_uber(args.data)
    shapes = dict(w=27, h=7, d=24, i=100, j=100,
                  r=args.custom_rank, k=args.spatial_rank)
    custom_model = 'wr,hr,dr,ikr,jkr->whdij'
    cp_model = 'wr,hr,dr,ir,jr->whdij'
    custom_parameters = parameter_count(custom_model, shapes)
    cp_shapes = {**shapes, 'r': args.cp_rank}
    cp_parameters = parameter_count(cp_model, cp_shapes)

    supplied = load_heldout(args.heldout, y.shape) if args.heldout else None
    custom_runs, cp_runs, heldout_fractions, validation_fractions = [], [], [], []
    for split in range(args.splits):
        split_seed = args.seed + split
        if supplied is None:
            heldout, validation = generated_split(y.shape, split_seed)
        else:
            heldout, validation = provided_split(supplied, split_seed)
        heldout_fractions.append(float(heldout.mean()))
        validation_fractions.append(float(validation.mean()))
        custom_runs.append(run_model(
            y, heldout, validation, custom_model, shapes, split_seed, args.iterations))
        cp_runs.append(run_model(
            y, heldout, validation, cp_model, cp_shapes, split_seed, args.iterations))
        del heldout, validation

    custom_mean, custom_se = mean_and_se([run['heldout'] for run in custom_runs])
    cp_mean, cp_se = mean_and_se([run['heldout'] for run in cp_runs])
    is_table_protocol = (supplied is None and args.splits == 10
                         and str(args.data).endswith('.npz'))
    report(args, {
        'claim': 2,
        'implementation': 'exact two-GPU i-sharded full-batch multiplicative updates',
        'divergence': 'alpha=1 beta=0 (KL/Poisson)',
        'split': ('ten deterministic 90/10 splits plus 5% validation' if supplied is None
                  else 'single provided split plus 5% validation'),
        'input': ('authors bundled demo tensor' if str(args.data).endswith('.npz')
                  else 'user-provided .npy tensor'),
        'table_1_protocol': is_table_protocol,
        'heldout_fractions': heldout_fractions,
        'validation_fractions': validation_fractions,
        'custom_model': {'rank': args.custom_rank, 'spatial_rank': args.spatial_rank,
                         'parameters': custom_parameters},
        'cp_model': {'rank': args.cp_rank, 'parameters': cp_parameters},
        # Keep scalar fields for compatibility with the prior report format.
        'custom_parameters': custom_parameters,
        'cp_parameters': cp_parameters,
        'custom_heldout_div': custom_mean,
        'cp_heldout_div': cp_mean,
        'custom_heldout_div_se': custom_se,
        'cp_heldout_div_se': cp_se,
        'runs': {'custom': custom_runs, 'cp': cp_runs},
        'targets': {'custom': 0.0101, 'cp': 0.0104},
        'pass': (is_table_protocol
                 and abs(custom_mean - 0.0101) <= 0.0005
                 and abs(cp_mean - 0.0104) <= 0.0005),
    })


if __name__ == '__main__':
    main()
