#!/usr/bin/env python
"""
Unified test inversion script.

Runs a 2D acoustic FWI using the same model and physics settings as
generate_baseline.py (via shared utils), but with configurable
fault-tolerance parameters for K8s test workflows.

Model:   circular disk inclusion (1600 m/s) in 1500 m/s background
Grid:    100x100, 1mm spacing
Shots:   16 elliptical transducer locations
Inversion: multi-frequency progression to avoid cycle skipping

Environment variables:
    NUM_ITERS           Total adjoint iterations (default: 4)
    DROP_THRESHOLD      Worker drop threshold (default: unset = None)
    BASELINE_PATH       Path to baseline .npy for validation (optional)
    BASELINE_TOLERANCE  Max L2 relative error (default: 0.05)
    ARTIFACT_ENDPOINT   MinIO endpoint
    ARTIFACT_ACCESS_KEY / ARTIFACT_SECRET_KEY / ARTIFACT_BUCKET
"""

import os
import time
from typing import Optional

import numpy as np

from stride import forward, IsoAcousticDevito

import mosaic
from mosaic.runtime.head import Head

from scripts.utils import (
    create_space_time, create_forward_problem,
    create_inversion_problem, run_inversion, save_comparison_plot,
    log_observed_checksums, log_vp_extended,
)

NUM_ITERS: int = int(os.environ.get('NUM_ITERS', '4'))
_dt_env: str = os.environ.get('DROP_THRESHOLD', '')
DROP_THRESHOLD: Optional[float] = float(_dt_env) if _dt_env != '' else None


async def main(runtime: Head, exp_name: str = 'test') -> None:
    np.random.seed(42)

    exp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    space, time_ = create_space_time()

    # ========================================================================
    # Phase 1: Forward pass — generate observed data, upload to artifact store
    # ========================================================================
    print('TEST: Phase 1 — forward pass starting', flush=True)
    t0 = time.time()

    problem_fwd, vp_true = create_forward_problem(exp_name, exp_dir, space, time_)

    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid,
                                       len=runtime.num_workers)
    await forward(problem_fwd, pde_fwd, vp_true)

    print('TEST: Phase 1 complete in %.1fs — traces uploaded to artifact store'
          % (time.time() - t0), flush=True)

    # ========================================================================
    # Phase 2: Inversion — multi-frequency progression
    # ========================================================================
    print('TEST: Phase 2 — inversion starting (drop_threshold=%s)' % DROP_THRESHOLD, flush=True)
    t_inv = time.time()

    problem, vp = create_inversion_problem(exp_name, exp_dir, space, time_)

    problem.acquisitions.load_artifacts()

    print('TEST: loaded %d shots via artifact store'
          % len(problem.acquisitions.shot_ids), flush=True)

    log_observed_checksums(problem, prefix='TEST')
    log_vp_extended(vp, 'pre-inversion', prefix='TEST')

    await run_inversion(problem, vp, runtime, NUM_ITERS,
                        drop_threshold=DROP_THRESHOLD,
                        log_prefix='TEST')

    print('\nTEST: inversion complete in %.1fs' % (time.time() - t_inv), flush=True)
    print('Final model range: [%.1f, %.1f] m/s' % (vp.data.min(), vp.data.max()), flush=True)

    # -- Mesh integrity check -------------------------------------------------
    worker_uids = sorted(w.uid for w in runtime.workers)
    node_uids = sorted(runtime._nodes.keys()) if hasattr(runtime, '_nodes') else []
    print('TEST: mesh check — workers=%s nodes=%s'
          % (worker_uids, node_uids), flush=True)

    # -- Validate against baseline --------------------------------------------
    baseline_path = os.environ.get('BASELINE_PATH')
    if baseline_path and os.path.exists(baseline_path):
        baseline = np.load(baseline_path)
        rel_error = np.linalg.norm(vp.data - baseline) / np.linalg.norm(baseline)
        tolerance = float(os.environ.get('BASELINE_TOLERANCE', '0.05'))
        print('VALIDATION: relative L2 error = %.6f (tolerance = %.2f)' % (rel_error, tolerance))
        if rel_error > tolerance:
            print('VALIDATION: FAIL — result diverged too far from baseline')
            raise RuntimeError('Baseline validation failed: rel_error=%.6f > tolerance=%.2f'
                               % (rel_error, tolerance))
        else:
            print('VALIDATION: PASS')

    # -- Save comparison plot --------------------------------------------------
    run_id = os.environ.get('ARTIFACT_RUN_ID', exp_name)
    plot_name = '%s_%diter' % (run_id, NUM_ITERS)
    plot_path = os.path.join(exp_dir, '%s.png' % plot_name)
    saved = save_comparison_plot(vp_true.data, vp.data, NUM_ITERS, plot_path)
    if saved:
        print('TEST: saved plot %s' % saved, flush=True)

        # Upload to MinIO so it can be retrieved after the pod dies
        art_warehouse = mosaic.get_artifact_warehouse()
        if art_warehouse is not None:
            plot_key = 'plots/%s.png' % plot_name
            with open(saved, 'rb') as f:
                plot_bytes = f.read()
            from io import BytesIO
            art_warehouse._client.put_object(
                art_warehouse._bucket, plot_key,
                BytesIO(plot_bytes), len(plot_bytes),
                content_type='image/png',
            )
            print('TEST: uploaded plot to s3://%s/%s' % (art_warehouse._bucket, plot_key),
                  flush=True)


if __name__ == '__main__':
    pod_ip = os.environ.get('POD_IP')
    if pod_ip:
        mosaic.run(main, address=pod_ip)
    else:
        mosaic.run(main)
