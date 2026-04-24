#!/usr/bin/env python
"""
Forward modelling test script.

Runs a 2D acoustic forward simulation using the same model and physics
settings as generate_baseline.py / inverse_test.py (via shared utils).

Model:   circular disk inclusion (1600 m/s) in 1500 m/s background
Grid:    100x100, 1mm spacing
Shots:   16 elliptical transducer locations
Wavelet: 0.30 MHz tone burst, 3 cycles

Environment variables:
    EXP_NAME            Experiment directory name (default: forward-test)
    ARTIFACT_ENDPOINT   MinIO endpoint (enables artifact warehouse upload)
    ARTIFACT_ACCESS_KEY / ARTIFACT_SECRET_KEY / ARTIFACT_BUCKET
"""

import os
import time

import numpy as np

from stride import forward, IsoAcousticDevito

import mosaic
from mosaic.runtime.head import Head

from scripts.utils import N, create_space_time, create_forward_problem


async def main(runtime: Head, exp_name: str = 'forward-test') -> None:
    exp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    space, time_ = create_space_time()

    print('FORWARD-TEST: starting (N=%d, workers=%d)' % (N, len(runtime.workers)),
          flush=True)
    t0 = time.time()

    problem, vp = create_forward_problem(exp_name, exp_dir, space, time_)

    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    await forward(problem, pde, vp)

    elapsed = time.time() - t0
    print('FORWARD-TEST: complete in %.1fs — %d shots' % (elapsed, N), flush=True)


if __name__ == '__main__':
    pod_ip = os.environ.get('POD_IP')
    if pod_ip:
        mosaic.run(main, address=pod_ip)
    else:
        mosaic.run(main)
