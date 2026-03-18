#!/usr/bin/env python
"""
Artifact-backed inversion script.

Runs the same 2D acoustic FWI as simple_inverse.py, but uses an artifact
store (MinIO) for all shot data transfer and gradient accumulation:

  Phase 1 — Forward pass
    Observed traces and wavelets are computed and uploaded to MinIO as .npy
    objects (one file per shot per field):
        shots/{N}/observed.npy   — recorded receiver traces
        shots/{N}/wavelets.npy   — source drive signals

    Each shot's observed field is replaced with an ArtifactTraces reference
    (metadata only — no data held in the head's memory).

  Phase 2 — Inversion
    acquisitions.load_artifacts() populates each shot:
        wavelets  — downloaded eagerly on the head (small; needed before
                    the PDE runs on the worker)
        observed  — wired as an ArtifactTraces reference; workers fetch
                    their shot's data from MinIO on demand (lazy loading)

    After each iteration the external gradient-accumulator service sums the
    per-worker gradient files and writes:
        gradients/iter_{N}/final.pkl

    The head's variable.pull(attr='grad') polls for this file and loads it.

Environment variables (see ArtifactWarehouse.from_env):
    ARTIFACT_ENDPOINT   MinIO endpoint, e.g. minio.argo.svc.cluster.local:9000
    ARTIFACT_ACCESS_KEY Access key
    ARTIFACT_SECRET_KEY Secret key
    ARTIFACT_BUCKET     Bucket name (default: stride-data)
"""

import os
import numpy as np

from stride import *
from stride.utils import wavelets

import mosaic

N = 2
NUM_ITERS = int(os.environ.get('NUM_ITERS', '4'))


async def main(runtime, exp_name: str = 'simple') -> None:
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # The ArtifactWarehouse is already configured globally by mrun via the
    # ARTIFACT_ENDPOINT env var — no need to construct it here.

    # -- Shared grid / time --------------------------------------------------
    shape = (100, 100)
    extra = (10, 10)
    absorbing = (10, 10)
    spacing = (1.0e-3, 1.0e-3)

    space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)

    start = 0.
    step = 0.1e-6
    num = 1000

    time = Time(start=start, step=step, num=num)

    f_centre = 0.50e6
    n_cycles = 3

    # ========================================================================
    # Phase 1: Forward pass — generate observed data, upload to artifact store
    # ========================================================================
    print('Phase 1: Running forward pass...', flush=True)

    problem_fwd = Problem(name=exp_name, space=space, time=time,
                          output_folder=exp_dir)

    # True model — layered: 1500 m/s top, 1600 m/s bottom
    vp_true = ScalarField(name='vp', grid=problem_fwd.grid)
    vp_true.fill(1500.)
    vp_true.data[shape[0] // 2:, :] = 1600.
    problem_fwd.medium.add(vp_true)

    problem_fwd.transducers.default()
    problem_fwd.geometry.default('elliptical', N)
    problem_fwd.acquisitions.default()

    for shot in problem_fwd.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid,
                                       len=runtime.num_workers)

    # forward() uploads each shot's observed.npy and wavelets.npy to MinIO
    # and replaces shot.observed with an ArtifactTraces reference.
    await forward(problem_fwd, pde_fwd, vp_true)

    print('Phase 1 complete. Observed traces uploaded to artifact store.', flush=True)

    # ========================================================================
    # Phase 2: Inversion — workers fetch observed data from artifact store
    # ========================================================================
    print('Phase 2: Starting inversion...', flush=True)

    problem = Problem(name=exp_name, space=space, time=time,
                      output_folder=exp_dir)

    # Initial model — homogeneous 1500 m/s
    vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
    vp.fill(1500.)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default('elliptical', N)
    problem.acquisitions.default()

    # Set up ArtifactTraces for each shot — no data downloaded here.
    # Workers will fetch from MinIO lazily when they run their shot.
    problem.acquisitions.load_artifacts()

    print('Loaded %d shots via artifact store.' % len(problem.acquisitions.shot_ids),
          flush=True)

    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    optimiser = GradientDescent(
        vp,
        step_size=5,
        process_grad=ProcessGlobalGradient(),
        process_model=ProcessModelIteration(min=1400., max=1700.),
    )

    optimisation_loop = OptimisationLoop()

    print('Beginning optimisation loop.', flush=True)

    for block in optimisation_loop.blocks(NUM_ITERS):
        try:
            await adjoint(problem, pde, loss,
                          optimisation_loop, optimiser, vp,
                          num_iters=1,
                          select_shots=dict(num=N))
            print('Block complete.', flush=True)
        except Exception as e:
            import traceback
            print('ERROR in adjoint: %s' % e, flush=True)
            traceback.print_exc()
            raise

    print('\nInversion complete.')
    print('Final model range: [%.1f, %.1f] m/s' % (vp.data.min(), vp.data.max()))


if __name__ == '__main__':
    mosaic.run(main)
