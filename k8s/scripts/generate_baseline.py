#!/usr/bin/env python
"""
Generate baseline inversion result for test validation.

Runs a clean 2D acoustic FWI (no fault tolerance, no drops) using local
mosaic mode, then saves:
  - k8s/tests/baselines/vp_100x100_4iter.npy  — final vp.data array
  - misc/plots/baseline_vp_100x100_4iter.png  — true model vs inversion plot

The .npy file is committed to the repo and baked into the Docker image.
Test workflows compare their result against it via L2 relative error.

Usage:
    python k8s/scripts/generate_baseline.py
"""

import os
import sys

import numpy as np

# Ensure k8s/scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from stride import forward
from stride.utils import wavelets

import mosaic

from scripts.utils import (
    N, SHAPE, F_CENTRE, N_CYCLES,
    create_space_time, create_forward_problem,
    create_inversion_problem, run_inversion, save_comparison_plot,
    log_observed_checksums, log_vp_extended,
)

NUM_ITERS = 4

BASELINE_PATH = os.path.join(os.path.dirname(__file__), '..', 'tests', 'baselines',
                             'vp_%dx%d_%diter.npy' % (SHAPE[0], SHAPE[1], NUM_ITERS))
PLOT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'misc', 'plots',
                         'baseline_vp_%dx%d_%diter.png' % (SHAPE[0], SHAPE[1], NUM_ITERS))


async def main(runtime, exp_name='baseline'):
    np.random.seed(42)

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    space, time = create_space_time()

    # -- Phase 1: Forward pass (true model) --
    print('Baseline: Phase 1 — forward pass', flush=True)

    problem_fwd, vp_true = create_forward_problem(exp_name, exp_dir, space, time)

    from stride import IsoAcousticDevito
    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid,
                                       len=runtime.num_workers)
    await forward(problem_fwd, pde_fwd, vp_true, dump=False)

    print('Baseline: Phase 1 complete', flush=True)

    # -- Phase 2: Inversion (starting model) --
    print('Baseline: Phase 2 — inversion (%d iters)' % NUM_ITERS, flush=True)

    problem, vp = create_inversion_problem(exp_name, exp_dir, space, time)

    for shot_inv, shot_fwd in zip(problem.acquisitions.shots,
                                  problem_fwd.acquisitions.shots):
        shot_inv.observed.data[:] = shot_fwd.observed.data
        shot_inv.wavelets.data[:] = shot_fwd.wavelets.data

    log_observed_checksums(problem, prefix='Baseline')
    log_vp_extended(vp, 'pre-inversion', prefix='Baseline')

    await run_inversion(problem, vp, runtime, NUM_ITERS,
                        log_prefix='Baseline')

    print('Baseline: inversion complete — range [%.1f, %.1f] m/s'
          % (vp.data.min(), vp.data.max()), flush=True)

    # -- Save baseline .npy --
    os.makedirs(os.path.dirname(os.path.abspath(BASELINE_PATH)), exist_ok=True)
    np.save(BASELINE_PATH, vp.data)
    print('Baseline: saved %s (shape=%s, dtype=%s)'
          % (BASELINE_PATH, vp.data.shape, vp.data.dtype), flush=True)

    # -- Save comparison plot --
    saved = save_comparison_plot(vp_true.data, vp.data, NUM_ITERS, PLOT_PATH,
                                title='Baseline Inversion (%d iter)' % NUM_ITERS)
    if saved:
        print('Baseline: saved plot %s' % saved, flush=True)


if __name__ == '__main__':
    num_workers = int(os.environ.get('NUM_WORKERS', '8'))
    mosaic.run(main, num_workers=num_workers)
