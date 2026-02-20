#!/usr/bin/env python
"""
S3-Enabled Inversion Script (self-contained)

Same as simple_inverse.py but with S3/MinIO persistence for models and gradients.
Generates observed data via a forward pass before running the inversion.
Reads MinIO configuration from environment variables.

Environment variables:
    MINIO_ENDPOINT   — MinIO endpoint (default: localhost:9000)
    MINIO_ACCESS_KEY — Access key (default: minioadmin)
    MINIO_SECRET_KEY — Secret key (default: minioadmin)
    MINIO_BUCKET     — Bucket name (default: stride-data)
    MINIO_SECURE     — Use HTTPS (default: false)

Must be run via the mosaic ``mrun`` launcher:

    mrun -nw 2 python scripts/simple_inverse_s3.py
"""

import os
import numpy as np

from stride import *
from stride.utils import wavelets
from stride.utils.s3 import S3Config, get_s3_client, ensure_bucket, stage_shots_to_s3

import mosaic

N = 2


async def main(runtime, exp_name='simple'):
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # S3 configuration from environment
    s3_config = S3Config.from_env()

    print('S3 Configuration:')
    print('  Endpoint: %s' % s3_config.endpoint)
    print('  Bucket:   %s' % s3_config.bucket)
    print('  Secure:   %s' % s3_config.secure)

    # Verify S3 connectivity
    s3_client = get_s3_client(s3_config)
    ensure_bucket(s3_client, s3_config.bucket)
    print('  Connected to S3 successfully.')

    # -- Shared grid/time setup ------------------------------------------
    shape = (100, 100)
    extra = (10, 10)
    absorbing = (10, 10)
    spacing = (1.0e-3, 1.0e-3)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)

    start = 0.
    step = 0.1e-6
    num = 1000

    time = Time(start=start,
                step=step,
                num=num)

    f_centre = 0.50e6
    n_cycles = 3

    # ====================================================================
    # Phase 1: Forward pass — generate observed data with the TRUE model
    # ====================================================================
    print('Phase 1: Running forward pass to generate observed data...', flush=True)

    problem_fwd = Problem(name=exp_name,
                          space=space, time=time,
                          output_folder=exp_dir)

    # True model — layered: 1500 m/s top, 1600 m/s bottom
    vp_true = ScalarField(name='vp', grid=problem_fwd.grid)
    vp_true.fill(1500.)
    vp_true.data[shape[0] // 2:, :] = 1600.
    problem_fwd.medium.add(vp_true)

    # Transducers + geometry
    problem_fwd.transducers.default()
    problem_fwd.geometry.default('elliptical', N)

    # Acquisitions + wavelets
    problem_fwd.acquisitions.default()
    for shot in problem_fwd.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid, len=runtime.num_workers)
    await forward(problem_fwd, pde_fwd, vp_true)
    print('Forward complete. Observed data saved to: %s' % exp_dir, flush=True)

    # ====================================================================
    # Phase 2: Inversion — recover model from homogeneous initial guess
    # ====================================================================
    print('Phase 2: Starting inversion...', flush=True)

    problem = Problem(name=exp_name,
                      space=space, time=time,
                      output_folder=exp_dir)

    # Initial model — homogeneous 1500 m/s
    vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    vp.fill(1500.)
    problem.medium.add(vp)

    # Transducers + geometry (must match forward)
    problem.transducers.default()
    problem.geometry.default('elliptical', N)

    # Load observed data from forward run
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Stage shot data to S3
    stage_shots_to_s3(s3_client, s3_config, problem)

    # Create PDE for inversion (separate from forward PDE)
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Optimiser — with s3_config for gradient persistence
    step_size = 5
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=1700.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model,
                                s3_config=s3_config)

    # Optimisation loop — 2 blocks x 2 iterations, with S3 persistence
    optimisation_loop = OptimisationLoop()

    num_blocks = 2
    num_iters = 2

    for block in optimisation_loop.blocks(num_blocks):
        await adjoint(problem, pde, loss,
                      optimisation_loop, optimiser, vp,
                      num_iters=num_iters,
                      select_shots=dict(num=N),
                      s3_config=s3_config)

    print('\nS3-enabled inversion complete.')
    print('Final model range: [%.1f, %.1f] m/s' % (vp.data.min(), vp.data.max()))
    print('Model and gradient data persisted to s3://%s/' % s3_config.bucket)


if __name__ == '__main__':
    mosaic.run(main)
