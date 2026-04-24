"""Shared constants, model setup, inversion runner, and plotting for test scripts."""

import os
import hashlib

# Pin Devito thread count so local and K8s runs produce identical PDE numerics.
# Must be set before Devito/OpenMP initialises (i.e. before any stride import).
os.environ.setdefault('OMP_NUM_THREADS', '8')

import numpy as np


def _md5(arr):
    """Return short md5 hex of a numpy array's raw bytes."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def log_observed_checksums(problem, prefix=''):
    """Print per-shot observed data checksums for comparison."""
    for shot in problem.acquisitions.shots:
        obs = shot.observed
        # ArtifactTraces has no data in memory — load it
        if hasattr(obs, '_artifact_key') and obs._artifact_key is not None:
            obs = obs.load()
        d = obs.data
        print('DIAG %s: shot %2d observed  md5=%s  sum=%.8e  shape=%s dtype=%s'
              % (prefix, shot.id, _md5(d), float(np.sum(d)), d.shape, d.dtype),
              flush=True)


def log_vp_extended(vp, label, prefix=''):
    """Print vp inner vs extended domain checksums."""
    vd = vp.data
    vx = vp.extended_data
    print('DIAG %s: %s  vp.data         shape=%s  min=%.6f  max=%.6f  sum=%.8e'
          % (prefix, label, vd.shape, float(np.min(vd)), float(np.max(vd)),
             float(np.sum(vd))), flush=True)
    print('DIAG %s: %s  vp.extended     shape=%s  min=%.6f  max=%.6f  sum=%.8e'
          % (prefix, label, vx.shape, float(np.min(vx)), float(np.max(vx)),
             float(np.sum(vx))), flush=True)


def log_grad_checksum(vp, iteration_label, prefix=''):
    """Print gradient and model checksums after an adjoint block."""
    g = vp.grad
    if g is not None and g.data is not None:
        gd = g.data
        gx = g.extended_data
        print('DIAG %s: %s  grad.data       md5=%s  sum=%.8e  min=%.8e  max=%.8e  shape=%s'
              % (prefix, iteration_label, _md5(gd), float(np.sum(gd)),
                 float(np.min(gd)), float(np.max(gd)), gd.shape), flush=True)
        print('DIAG %s: %s  grad.ext_data   md5=%s  sum=%.8e  shape=%s'
              % (prefix, iteration_label, _md5(gx), float(np.sum(gx)), gx.shape),
              flush=True)
    log_vp_extended(vp, iteration_label, prefix)


from stride import (
    Space, Time, Problem, ScalarField, IsoAcousticDevito,
    L2DistanceLoss, GradientDescent, ProcessGlobalGradient,
    ProcessModelIteration, OptimisationLoop, forward, adjoint,
)
from stride.utils import wavelets


# ── Physical constants ───────────────────────────────────────────────────────

N = 16
SHAPE = (100, 100)
EXTRA = (10, 10)
ABSORBING = (10, 10)
SPACING = (1.0e-3, 1.0e-3)

TIME_START = 0.
TIME_STEP = 0.1e-6
TIME_NUM = 1000

F_CENTRE = 0.30e6
N_CYCLES = 3

VP_BACKGROUND = 1500.
VP_INCLUSION = 1600.
INCLUSION_RADIUS = 25

MAX_FREQS = [0.5e7, 0.1e6, 0.3e6, 0.6e6]

STEP_SIZE = 5
VP_MIN = 1400.
VP_MAX = 1700.


# ── Grid / time ─────────────────────────────────────────────────────────────

def create_space_time():
    """Return (space, time) with the standard test grid."""
    space = Space(shape=SHAPE, extra=EXTRA, absorbing=ABSORBING, spacing=SPACING)
    time = Time(start=TIME_START, step=TIME_STEP, num=TIME_NUM)
    return space, time


# ── True model ───────────────────────────────────────────────────────────────

def create_true_model(grid):
    """Return a ScalarField with the circular-inclusion velocity model."""
    vp_true = ScalarField(name='vp', grid=grid)
    vp_true.fill(VP_BACKGROUND)
    cx, cz = SHAPE[0] // 2, SHAPE[1] // 2
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    mask = (xx - cx)**2 + (yy - cz)**2 <= INCLUSION_RADIUS**2
    vp_true.data[mask] = VP_INCLUSION
    return vp_true


# ── Forward problem ──────────────────────────────────────────────────────────

def create_forward_problem(exp_name, exp_dir, space, time):
    """Set up a Problem with the true model, transducers, geometry, and wavelets.

    Returns (problem, vp_true).
    """
    problem = Problem(name=exp_name, space=space, time=time,
                      output_folder=exp_dir)

    vp_true = create_true_model(problem.grid)
    problem.medium.add(vp_true)

    problem.transducers.default()
    problem.geometry.default('elliptical', N)
    problem.acquisitions.default()

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            F_CENTRE, N_CYCLES, time.num, time.step)

    return problem, vp_true


# ── Inversion problem ───────────────────────────────────────────────────────

def create_inversion_problem(exp_name, exp_dir, space, time):
    """Set up a Problem with a homogeneous starting model for inversion.

    Returns (problem, vp).
    """
    problem = Problem(name=exp_name, space=space, time=time,
                      output_folder=exp_dir)

    vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
    vp.fill(VP_BACKGROUND)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default('elliptical', N)
    problem.acquisitions.default()

    return problem, vp


# ── Run inversion ────────────────────────────────────────────────────────────

async def run_inversion(problem, vp, runtime, num_iters,
                        drop_threshold=None, log_prefix=''):
    """Run the multi-frequency inversion loop.

    Parameters
    ----------
    problem : Problem
        Problem with observed data already loaded (via direct copy or load_artifacts).
    vp : ScalarField
        Starting velocity model (needs_grad=True).
    runtime
        Mosaic runtime (Head or local).
    num_iters : int
        Total iterations across all frequency blocks.
    drop_threshold : float or None
        Worker-drop threshold for fault tolerance.
    log_prefix : str
        Prefix for log messages (e.g. 'Baseline' or 'TEST').
    """
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    optimiser = GradientDescent(
        vp,
        step_size=STEP_SIZE,
        process_grad=ProcessGlobalGradient(),
        process_model=ProcessModelIteration(min=VP_MIN, max=VP_MAX),
    )

    optimisation_loop = OptimisationLoop()

    num_blocks = len(MAX_FREQS)
    num_iters_per_block = num_iters // num_blocks

    for block, freq in optimisation_loop.blocks(num_blocks, MAX_FREQS):
        if log_prefix:
            print('%s: block %d (f_max=%.1f kHz) starting'
                  % (log_prefix, block.id, freq / 1e3), flush=True)

        await adjoint(problem, pde, loss,
                      optimisation_loop, optimiser, vp,
                      num_iters=num_iters_per_block,
                      select_shots=dict(num=N),
                      f_max=freq, max_freqs=MAX_FREQS,
                      drop_threshold=drop_threshold,
                      global_prec=True,
                      )

        if log_prefix:
            print('%s: block %d complete' % (log_prefix, block.id), flush=True)
            log_grad_checksum(vp, 'block_%d' % block.id, prefix=log_prefix)


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_comparison_plot(vp_true_data, vp_data, num_iters, plot_path, title=None):
    """Save a side-by-side comparison of true vs inverted velocity model.

    Parameters
    ----------
    vp_true_data : ndarray
        True velocity model.
    vp_data : ndarray
        Inverted velocity model.
    num_iters : int
        Number of iterations (used in title).
    plot_path : str
        Output file path for the PNG.
    title : str, optional
        Right panel title. Defaults to 'Inversion (N iter)'.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot', flush=True)
        return None

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    vmin = min(vp_true_data.min(), vp_data.min())
    vmax = max(vp_true_data.max(), vp_data.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].imshow(vp_true_data.T, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('True Vp')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')

    im1 = axes[1].imshow(vp_data.T, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title(title or 'Inversion (%d iter)' % num_iters)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')

    fig.colorbar(im1, ax=axes.ravel().tolist(), label='Vp (m/s)', shrink=0.8)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path
