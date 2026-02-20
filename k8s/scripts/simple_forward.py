#!/usr/bin/env python
"""
Simple Forward Modeling Script

Runs a basic 2D acoustic forward simulation:
- 100x100 grid at 1mm spacing
- Layered velocity model (1500 m/s top, 1600 m/s bottom)
- 8 transducers in elliptical arrangement
- 500 kHz tone burst wavelet, 3 cycles

Must be run via the mosaic ``mrun`` launcher, which spawns worker processes:

    mrun -nw 2 python scripts/simple_forward.py

To see all mrun options:

    mrun --help
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

    # Create the grid
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

    # Create problem
    problem = Problem(name=exp_name,
                      space=space, time=time,
                      output_folder=exp_dir)

    # Create medium — layered velocity model
    vp = ScalarField(name='vp', grid=problem.grid)
    vp.fill(1500.)
    # Bottom half: 1600 m/s
    vp.data[shape[0] // 2:, :] = 1600.

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry — 8 transducers in elliptical arrangement
    num_locations = N
    problem.geometry.default('elliptical', num_locations)

    # Create acquisitions
    problem.acquisitions.default()

    # Create wavelets — 500 kHz tone burst, 3 cycles
    f_centre = 0.50e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run forward
    await forward(problem, pde, vp)

    print('\nForward run complete.')
    print('Output saved to: %s' % problem.output_folder)


if __name__ == '__main__':
    mosaic.run(main)
