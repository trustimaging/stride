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

────────────────────────────────────────────────────────────────────────────────
RUNTIME ARCHITECTURE
────────────────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  HEAD pod  (runs this script via mosaic.run)                │
  │  - Owns the optimisation loop and model parameters          │
  │  - Dispatches shots to workers via async_for                │
  │  - Collects losses; polls MinIO for the summed gradient     │
  │  - Updates the model (optimiser.step)                       │
  └──────────────────┬──────────────────────────────────────────┘
                     │  ZMQ (pub/sub + request/reply)
  ┌──────────────────▼──────────────────────────────────────────┐
  │  MOSAIC MONITOR  (subprocess inside the head pod)           │
  │  - Spawned by the head at startup (head.py:init_monitor)   │
  │  - Listens on ports 3000/3001; workers reach it via the    │
  │    K8s monitor-service (a Service, not a pod) which routes │
  │    DNS traffic into the head pod                           │
  │  - Maintains a registry of all connected workers           │
  │  - Routes ZMQ messages between head and workers            │
  │  - Sends heartbeats; fires disconnect on failure           │
  └──────────┬───────────────────────────┬──────────────────────┘
             │                           │
  ┌──────────▼──────────┐   ┌────────────▼────────────┐
  │  WORKER 0           │   │  WORKER 1               │
  │  - Holds a replica  │   │  - Holds a replica      │
  │    of pde / loss    │   │    of pde / loss        │
  │  - Runs state eq.   │   │  - Runs state eq.       │
  │  - Runs adjoint eq. │   │  - Runs adjoint eq.     │
  │  - Uploads          │   │  - Uploads              │
  │    shot_N.pkl →MinIO│   │    shot_N.pkl →MinIO    │
  └─────────────────────┘   └─────────────────────────┘
                                       ▲
  ┌────────────────────────────────────┴────────────────────────┐
  │  GRADIENT ACCUMULATOR  (separate pod, no ZMQ connection)    │
  │  - Polls MinIO for shot_*.pkl files each iteration          │
  │  - Sums them and writes gradients/iter_N/final.pkl          │
  └─────────────────────────────────────────────────────────────┘

TESSERA (distributed objects)
  Objects created with .remote() are "tessera" — the head holds a lightweight
  stub and each worker holds a full replica. A method call on the stub is
  serialised and forwarded by the monitor to every worker replica in parallel.
  Return values are gathered back to the head.

  pde  = IsoAcousticDevito.remote(len=2)   →  one replica on each of 2 workers
  loss = L2DistanceLoss.remote(len=2)      →  same

  vp   = ScalarField.parameter(...)        →  a tessera for the model parameter;
                                              vp.pull(attr='grad') fetches the
                                              gradient that was written to MinIO
                                              by the accumulator pod.
"""

import os
from typing import Optional

import numpy as np

from stride import *
from stride.utils import wavelets

import mosaic
from mosaic.runtime.head import Head

N: int = 16
NUM_ITERS: int = int(os.environ.get('NUM_ITERS', '4'))
_dt_env: str = os.environ.get('DROP_THRESHOLD', '')
DROP_THRESHOLD: Optional[float] = float(_dt_env) if _dt_env != '' else None


async def main(runtime: Head, exp_name: str = 'simple') -> None:
    # runtime  — the Head runtime object.  Gives access to the worker pool,
    #            async_for dispatch, and the global event loop.  mosaic.run()
    #            starts the head, waits for NUM_WORKERS workers to phone home
    #            via the monitor, then calls this coroutine.

    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'exps', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # The ArtifactWarehouse is configured globally before main() is called:
    # mosaic.init() reads ARTIFACT_ENDPOINT and calls
    # mosaic.set_artifact_warehouse(ArtifactWarehouse.from_env()).
    # All subsequent forward/adjoint calls read it via mosaic.get_artifact_warehouse().

    # -- Shared grid / time --------------------------------------------------
    # These objects are constructed on the head only.  They are passed as
    # arguments to .remote() calls and serialised over ZMQ to every worker.
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
    # This phase runs the TRUE model to produce synthetic "field recordings"
    # that the inversion will try to match.  All of this runs on the HEAD
    # except the actual PDE solves which are dispatched to workers.
    print('Phase 1: Running forward pass...', flush=True)

    problem_fwd = Problem(name=exp_name, space=space, time=time,
                          output_folder=exp_dir)

    # True model — circular inclusion: 1500 m/s background, 1600 m/s disk
    # ScalarField (not .parameter) — plain numpy array, lives only on the head.
    vp_true = ScalarField(name='vp', grid=problem_fwd.grid)
    vp_true.fill(1500.)
    cx, cz = shape[0] // 2, shape[1] // 2
    radius = 25
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    mask = (xx - cx)**2 + (yy - cz)**2 <= radius**2
    vp_true.data[mask] = 1600.
    problem_fwd.medium.add(vp_true)

    # Transducers: physical source/receiver elements (geometry metadata only).
    # Geometry: where transducers are placed around the domain (head-side).
    # Acquisitions: collection of Shot objects — each shot is one source firing.
    problem_fwd.transducers.default()
    problem_fwd.geometry.default('elliptical', N)
    problem_fwd.acquisitions.default()

    for shot in problem_fwd.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # IsoAcousticDevito.remote(...) creates a tessera: the head gets a stub,
    # and each worker receives a full IsoAcousticDevito replica via the monitor.
    # len=runtime.num_workers means one replica per worker.
    pde_fwd = IsoAcousticDevito.remote(grid=problem_fwd.grid,
                                       len=runtime.num_workers)

    # forward() dispatches shots to workers via runtime.async_for:
    #   For each shot:
    #     HEAD  — serialises vp_true and the shot's wavelets, sends to a worker
    #     WORKER — runs the state (forward) PDE equation, records traces at receivers
    #     HEAD  — receives the modelled traces back
    #     HEAD  — uploads observed.npy + wavelets.npy to MinIO via ArtifactWarehouse
    #     HEAD  — replaces shot.observed with an ArtifactTraces(key=...) reference
    #             so the raw array is no longer held in head memory
    await forward(problem_fwd, pde_fwd, vp_true)

    print('Phase 1 complete. Observed traces uploaded to artifact store.', flush=True)

    # ========================================================================
    # Phase 2: Inversion — workers fetch observed data from artifact store
    # ========================================================================
    print('Phase 2: Starting inversion...', flush=True)

    problem = Problem(name=exp_name, space=space, time=time,
                      output_folder=exp_dir)

    # ScalarField.parameter(...) creates a tessera for the model parameter:
    #   - The head holds the current values and accumulates the gradient.
    #   - Workers receive the current vp values before each PDE solve via
    #     runtime.put(vp) inside adjoint(), which broadcasts a snapshot of
    #     the current values to all workers without keeping a permanent replica.
    #   - needs_grad=True: the tessera tracks a gradient buffer that is
    #     populated by vp.pull(attr='grad') after each iteration.
    vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
    vp.fill(1500.)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default('elliptical', N)
    problem.acquisitions.default()

    # load_artifacts() replaces each shot's observed field with an
    # ArtifactTraces(key='shots/{id}/observed.npy') reference.
    # Workers will download their shot's data from MinIO on demand
    # during the adjoint PDE solve — the head never holds the raw arrays.
    # Wavelets are downloaded eagerly here (small; needed by the head to
    # pre-process before dispatching to workers).
    problem.acquisitions.load_artifacts()

    print('Loaded %d shots via artifact store.' % len(problem.acquisitions.shot_ids),
          flush=True)

    # Create tessera replicas of the PDE solver and loss function on each worker.
    # The monitor forwards initialisation messages to all workers in parallel.
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # GradientDescent lives entirely on the head — it reads vp.grad (a local
    # numpy array populated by vp.pull) and updates vp.data in-place.
    # ProcessGlobalGradient: normalises the gradient across all shots.
    # ProcessModelIteration: clips vp values to [1400, 1700] m/s after each step.
    optimiser = GradientDescent(
        vp,
        step_size=5,
        process_grad=ProcessGlobalGradient(),
        process_model=ProcessModelIteration(min=1400., max=1700.),
    )

    # OptimisationLoop tracks block/iteration counters and loss history.
    # It runs entirely on the head.
    optimisation_loop = OptimisationLoop()

    print('Beginning optimisation loop.', flush=True)

    for block in optimisation_loop.blocks(NUM_ITERS):
        # Each block is one gradient-descent step.  With num_iters=1 there is
        # exactly one adjoint iteration per block, using all N shots.
        try:
            # adjoint() orchestrates one full iteration across head and workers:
            #
            #   HEAD
            #     1. Broadcasts the current vp snapshot to all workers via put().
            #     2. Writes shots.json to MinIO listing expected shot IDs.
            #     3. Dispatches shots to workers via runtime.async_for (round-robin).
            #        If drop_threshold is set, starts _watch_workers concurrently.
            #
            #   WORKER (per shot, in parallel across workers)
            #     4. Downloads its shot's observed.npy from MinIO (lazy, on demand).
            #     5. Runs the state (forward) equation, saving the wavefield.
            #     6. Computes the residual against observed traces (loss function).
            #     7. Runs the adjoint equation using the saved wavefield.
            #     8. Computes the gradient contribution for this shot.
            #     9. Uploads shot_{id}.pkl (gradient array) to MinIO.
            #     10. Returns the loss value to the head.
            #
            #   HEAD
            #     11. Collects loss values from all workers.
            #     12. Calls vp.pull(attr='grad'), which polls MinIO for
            #         gradients/iter_N/final.pkl (written by the accumulator pod
            #         once it has summed all shot_*.pkl files).
            #     13. Runs optimiser.step(): applies the gradient to update vp.data.
            #
            #   GRADIENT ACCUMULATOR POD (independent, no ZMQ)
            #     - Polls MinIO for shots.json, then waits for each shot_{id}.pkl.
            #     - Sums all gradient arrays as they arrive.
            #     - Writes final.pkl, unblocking step 12 above.
            #
            #   FAULT TOLERANCE (if drop_threshold is not None)
            #     - _watch_workers monitors the live worker UID set.
            #     - If the fraction of original workers that have dropped exceeds
            #       drop_threshold, the async_for loop is cancelled and retried
            #       with the surviving workers covering all shots.
            await adjoint(problem, pde, loss,
                          optimisation_loop, optimiser, vp,
                          num_iters=1,
                          select_shots=dict(num=N),
                          drop_threshold=DROP_THRESHOLD)
            print('Block complete.', flush=True)
        except Exception as e:
            import traceback
            print('ERROR in adjoint: %s' % e, flush=True)
            traceback.print_exc()
            raise

    print('\nInversion complete.')
    print('Final model range: [%.1f, %.1f] m/s' % (vp.data.min(), vp.data.max()))

    # ── Validate against baseline ────────────────────────────────────────────
    baseline_path = os.environ.get('BASELINE_PATH')
    if baseline_path and os.path.exists(baseline_path):
        baseline = np.load(baseline_path)
        rel_error = np.linalg.norm(vp.data - baseline) / np.linalg.norm(baseline)
        tolerance = float(os.environ.get('BASELINE_TOLERANCE', '0.15'))
        print('VALIDATION: relative L2 error = %.6f (tolerance = %.2f)' % (rel_error, tolerance))
        if rel_error > tolerance:
            print('VALIDATION: FAIL — result diverged too far from baseline')
            raise RuntimeError('Baseline validation failed: rel_error=%.6f > tolerance=%.2f'
                               % (rel_error, tolerance))
        else:
            print('VALIDATION: PASS')

        # Save comparison plot if requested
        plot_path = os.environ.get('PLOT_PATH')
        if plot_path:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(16, 5))

                vmin = min(baseline.min(), vp.data.min())
                vmax = max(baseline.max(), vp.data.max())

                axes[0].imshow(baseline.T, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
                axes[0].set_title('Baseline')

                axes[1].imshow(vp.data.T, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
                axes[1].set_title('Test Result')

                diff = vp.data - baseline
                im2 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r')
                axes[2].set_title('Difference (L2=%.4f)' % rel_error)
                fig.colorbar(im2, ax=axes[2], shrink=0.8)

                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                print('VALIDATION: saved plot to %s' % plot_path)
            except ImportError:
                print('VALIDATION: matplotlib not available, skipping plot')


if __name__ == '__main__':
    # mosaic.run() does the following before calling main():
    #   1. Starts the HEAD runtime and the MONITOR (both in this process).
    #   2. Reads the monitor.key file written by the Kubernetes monitor service
    #      to get the monitor's ZMQ address and ports.
    #   3. Waits for NUM_WORKERS worker pods to phone home and register with
    #      the monitor (dynamic mode).
    #   4. Calls main(runtime) in the head's async event loop.
    #   5. On return, shuts down the runtime and closes all ZMQ connections.
    mosaic.run(main)
