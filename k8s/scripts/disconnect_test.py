#!/usr/bin/env python
"""
Disconnect Test Script

Same artifact-backed inversion as simple_inverse_artifacts.py, but with:
  - drop_threshold=None  — inversion is never cancelled/restarted on a drop;
                           it continues with whatever workers are still alive.
  - Timing prints at each block boundary so disconnect latency is visible
    when log timestamps are compared across pods.
  - Mesh integrity check at the end: prints runtime.workers / _nodes so you
    can confirm the dropped node left no stale entries.

Environment variables (same as simple_inverse_artifacts.py):
    NUM_WORKERS         — how many workers to wait for before starting (default: 2)
    NUM_ITERS           — total adjoint iterations (default: 4)
    ARTIFACT_ENDPOINT   — MinIO endpoint
    ARTIFACT_ACCESS_KEY / ARTIFACT_SECRET_KEY / ARTIFACT_BUCKET
"""

import os
import time

import numpy as np

from stride import *
from stride.utils import wavelets

import mosaic
from mosaic.runtime.head import Head

N: int = 2
NUM_ITERS: int = int(os.environ.get('NUM_ITERS', '4'))


async def main(runtime: Head, exp_name: str = 'disconnect-test') -> None:
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # -- Shared grid / time --------------------------------------------------
    shape = (100, 100)
    extra = (10, 10)
    absorbing = (10, 10)
    spacing = (1.0e-3, 1.0e-3)

    space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)

    start = 0.
    step = 0.1e-6
    num = 1000

    time_ = Time(start=start, step=step, num=num)

    f_centre = 0.50e6
    n_cycles = 3

    # ========================================================================
    # Phase 1: Forward pass — generate observed data, upload to artifact store
    # ========================================================================
    print('DISCONNECT-TEST: Phase 1 — forward pass starting', flush=True)
    t0 = time.time()

    problem_fwd = Problem(name=exp_name, space=space, time=time_,
                          output_folder=exp_dir)

    vp_true = ScalarField(name='vp', grid=problem_fwd.grid)
    vp_true.fill(1500.)
    vp_true.data[shape[0] // 2:, :] = 1600.
    problem_fwd.medium.add(vp_true)

    problem_fwd.transducers.default()
    problem_fwd.geometry.default('elliptical', N)
    problem_fwd.acquisitions.default()

    for shot in problem_fwd.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time_.num, time_.step)

    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid,
                                       len=runtime.num_workers)
    await forward(problem_fwd, pde_fwd, vp_true)

    print('DISCONNECT-TEST: Phase 1 complete in %.1fs — traces uploaded to artifact store'
          % (time.time() - t0), flush=True)

    # ========================================================================
    # Phase 2: Inversion — workers fetch observed data from artifact store
    # ========================================================================
    # drop_threshold=None: the inversion never cancels/restarts on a worker
    # drop.  adjoint() will simply dispatch the remaining shots to whatever
    # workers are still alive.  This lets us observe:
    #   (a) whether the mesh cleans up correctly after the drop, and
    #   (b) whether the inversion runs to completion on a single worker.
    print('DISCONNECT-TEST: Phase 2 — inversion starting (drop_threshold=None)', flush=True)
    t_inv = time.time()

    problem = Problem(name=exp_name, space=space, time=time_,
                      output_folder=exp_dir)

    vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
    vp.fill(1500.)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default('elliptical', N)
    problem.acquisitions.default()

    problem.acquisitions.load_artifacts()

    print('DISCONNECT-TEST: loaded %d shots via artifact store'
          % len(problem.acquisitions.shot_ids), flush=True)

    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    optimiser = GradientDescent(
        vp,
        step_size=5,
        process_grad=ProcessGlobalGradient(),
        process_model=ProcessModelIteration(min=1400., max=1700.),
    )

    optimisation_loop = OptimisationLoop()

    for block in optimisation_loop.blocks(NUM_ITERS):
        t_block = time.time()
        print('DISCONNECT-TEST: block %d starting — workers_alive=%d t=%.1fs'
              % (block.id, len(runtime.workers), time.time() - t_inv), flush=True)
        try:
            await adjoint(problem, pde, loss,
                          optimisation_loop, optimiser, vp,
                          num_iters=1,
                          select_shots=dict(num=N),
                          drop_threshold=None)
            print('DISCONNECT-TEST: block %d complete — workers_alive=%d elapsed=%.1fs t=%.1fs'
                  % (block.id, len(runtime.workers),
                     time.time() - t_block, time.time() - t_inv), flush=True)
        except Exception as e:
            import traceback
            print('DISCONNECT-TEST: ERROR in block %d at t=%.1fs: %s'
                  % (block.id, time.time() - t_inv, e), flush=True)
            traceback.print_exc()
            raise

    print('\nDISCONNECT-TEST: inversion complete in %.1fs' % (time.time() - t_inv), flush=True)
    print('Final model range: [%.1f, %.1f] m/s' % (vp.data.min(), vp.data.max()), flush=True)

    # ── Mesh integrity check ─────────────────────────────────────────────────
    # After the drop, these sets should contain only the surviving worker(s).
    # Any entry from the dropped node here means a stale reference leaked.
    worker_uids = sorted(w.uid for w in runtime.workers)
    node_uids = sorted(runtime._nodes.keys()) if hasattr(runtime, '_nodes') else []
    print('DISCONNECT-TEST: mesh check — workers=%s nodes=%s'
          % (worker_uids, node_uids), flush=True)


if __name__ == '__main__':
    pod_ip = os.environ.get('POD_IP')
    if pod_ip:
        mosaic.run(main, address=pod_ip)
    else:
        mosaic.run(main)
