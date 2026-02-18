#!/usr/bin/env python
"""
Simple Inversion Script (self-contained)

Runs a basic 2D acoustic full-waveform inversion:
- Generates observed data via a forward pass (layered true model)
- Initial homogeneous model at 1500 m/s
- L2 distance loss
- Gradient descent optimiser, step_size=5, bounds [1400, 1700]
- 2 blocks x 2 iterations, all shots per iteration

Must be run via the mosaic ``mrun`` launcher:

    mrun -nw 2 python scripts/simple_inverse.py
"""

import os
import numpy as np

from stride import *
from stride.utils import wavelets

import mosaic

N = 2


async def main(runtime, exp_name='simple'):
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

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
    print('Loading observed data from %s (project=%s)...' %
          (problem.output_folder, problem.name), flush=True)
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)
    print('Loaded %d shots.' % len(problem.acquisitions.shot_ids), flush=True)

    # Create PDE for inversion (separate from forward PDE)
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Optimiser
    step_size = 5
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=1700.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model)

    # Optimisation loop — 2 blocks x 2 iterations
    optimisation_loop = OptimisationLoop()

    num_blocks = 2
    num_iters = 2

    print('Beginning optimisation loop.', flush=True)

    for block in optimisation_loop.blocks(num_blocks):
        try:
            await adjoint(problem, pde, loss,
                          optimisation_loop, optimiser, vp,
                          num_iters=num_iters,
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
