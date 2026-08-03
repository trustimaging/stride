# Stride Forward & Inverse Scripts — Object & Argument Reference

A practical reference for the objects most used in Stride forward-modelling and
inversion (FWI) scripts, centred on **the arguments accepted by `forward()` and
`adjoint()`** and by the PDE operators they drive.

Line references point at the public repo
(`/Users/andreidanila/code/sonalis/stride`). A final section documents the
extra functionality and advanced examples in the private fork
(`/Users/andreidanila/code/sonalis/stride-private`).

---

## 1. The two canonical script shapes

Every script is an `async def main(runtime)` launched with `mosaic.run(main)`
(run on a cluster with `mrun -nw <workers> python script.py`).

**Forward script** ([`breast2D/01_script_forward.py`](stride_examples/examples/breast2D/01_script_forward.py)):

```python
space = Space(shape=(356, 385), extra=(50, 50), absorbing=(40, 40), spacing=0.5e-3)
time  = Time(start=0., step=0.08e-6, num=2500)
problem = Problem(name='anastasio2D', space=space, time=time)

vp = ScalarField(name='vp', grid=problem.grid); vp.load('...TrueModel.h5')
problem.medium.add(vp)
problem.transducers.default()
problem.geometry.default('elliptical', 128)
problem.acquisitions.default()
for shot in problem.acquisitions.shots:
    shot.wavelets.data[0, :] = wavelets.tone_burst(0.5e6, 3, time.num, time.step)

pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
await forward(problem, pde, vp)          # <-- entry point 1
```

**Inverse script** ([`breast2D/02_script_inverse.py`](stride_examples/examples/breast2D/02_script_inverse.py)):

```python
vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
vp.fill(1500.)
problem.medium.add(vp)
problem.acquisitions.load(path=problem.output_folder, project_name=problem.name, version=0)

pde  = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
loss = L2DistanceLoss.remote(len=runtime.num_workers)
optimiser = GradientDescent(vp, step_size=10,
                            process_grad=ProcessGlobalGradient(),
                            process_model=ProcessModelIteration(min=1400., max=1700.))
optimisation_loop = OptimisationLoop()

for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
    await adjoint(problem, pde, loss, optimisation_loop, optimiser, vp,   # <-- entry point 2
                  num_iters=8, select_shots=dict(num=16, randomly=True),
                  f_max=freq, max_freqs=max_freqs)
```

Two key rules that explain most of the API:

- **`.remote(...)`** turns an `Operator`/PDE class into a distributed *tessera*
  spread across `len=runtime.num_workers` workers. Construction kwargs
  (`grid`, `space`, `time`, `name`, `cached_operator`, …) go here.
- **`needs_grad=True`** on a variable (via `ScalarField.parameter(...)`) is what
  switches the whole pipeline from pure forward to gradient-tracking adjoint
  mode. `forward()` will automatically save the wavefield when it sees a
  `needs_grad` input; `adjoint()` builds the adjoint graph from it.

---

## 2. `forward(problem, pde, *args, **kwargs)`

Defined in [`stride/__init__.py:53`](stride/__init__.py#L53). Runs `pde` once
per shot, stores the result in `shot.observed`, and (by default) appends it to
the observed HDF5 file. `*args` (e.g. `vp`, `rho`, `alpha`) and all extra
`kwargs` are **forwarded to the PDE**.

| kwarg | type | default | meaning |
|---|---|---|---|
| `dump` | bool | `True` | Write forward result to disk (`observed` file). When `True` it also pre-loads any existing observed (version 0) so already-run shots are skipped. |
| `shot_ids` | list / int | `None` | Specific shots to run. `None` → all *remaining* shots (those with no observed yet). If none remain it logs a warning and returns. |
| `deallocate` | bool | `False` | Free `shot.observed` right after each shot (memory saving). |
| `safe` | bool | `True` | Discard workers that fail mid-execution instead of aborting. |
| `platform` | str | `'cpu'` | `'cpu'`, `'gpu'`, `'nvidia-acc'`, `'nvidia-cuda'`. Triggers GPU device round-robin across workers. |
| `devices` | list | `None` | Explicit GPU device ids; `None` → auto (`gpu_count()`). |
| `*args` | — | — | Positional PDE inputs (medium fields): `vp`, or `vp, vs, rho` (elastic), etc. Published to all workers. |
| `**kwargs` | — | — | Everything else is passed straight to the PDE `forward` (see §4). |

Notable behaviour: the loop hands each `shot_id` to a worker, builds a
`sub_problem = problem.sub_problem(shot_id)`, calls
`pde(wavelets, *args, problem=sub_problem, runtime=worker, **kwargs).result()`,
writes `shot.observed.data[:] = traces.data`, and raises if NaN/Inf appear.

---

## 3. `adjoint(problem, pde, loss, optimisation_loop, optimiser, *args, **kwargs)`

Defined in [`stride/__init__.py:172`](stride/__init__.py#L172). This is the FWI
iteration driver. Per iteration it: selects shots → pre-processes
wavelets/observed → runs the PDE forward (saving the wavefield) →
post-processes modelled+observed traces → evaluates `loss` → runs the adjoint
(`fun.adjoint(...)`) to accumulate the gradient → takes an optimiser step →
dumps the updated variable.

| kwarg | type | default | meaning |
|---|---|---|---|
| `num_iters` | int | `1` | Iterations to run **within the current block**. |
| `select_shots` | dict | `{}` | Shot-selection rules per iteration, forwarded to `Acquisitions.select_shot_ids` — e.g. `dict(num=16, randomly=True)`, or `dict(start=, end=, every=)`. |
| `lazy_loading` | bool | `False` | Load shot data each iteration and deallocate after, to save memory. |
| `dump` | bool | `True` | Save the updated optimiser variable after each iteration (enables restart). |
| `safe` | bool | `True` | Discard failing workers. |
| `f_min` | float | `None` | High-pass corner for filtering wavelets/traces. `None` → no high-pass. |
| `f_max` | float | `None` | Low-pass corner (frequency continuation). `None` → no low-pass. Usually set per block. |
| `filter_traces` | bool | `True` | Whether the trace-processing pipeline filters modelled/observed. |
| `filter_wavelets` | bool | `= filter_traces` | Whether wavelets/observed are band-limited before the PDE. |
| `filter_wavelets_relaxation` | float | `0.75` | Filter roll-off relaxation for wavelets. |
| `filter_traces_relaxation` | float | `0.75` (or `1.0` if no wavelet filtering) | Filter roll-off relaxation for traces. |
| `step_size` | float / `LineSearch` | `optimiser.step_size` | Step length; a `LineSearch` instance triggers the residual-keeping test-step loop. |
| `platform` / `devices` | str / list | `'cpu'` / `None` | GPU control, same as `forward()`. |
| `*args` | — | — | The medium variable(s) being inverted (also the `wrt` of the gradient), e.g. `vp` (must be a `.parameter(..., needs_grad=True)`). |
| `**kwargs` | — | — | Passed to the four processing pipelines, the PDE, the loss, and `optimiser.step`. Includes everything in §4 (`kernel`, `interpolation_type`, `boundary_type`, `devito_config`, …) and pipeline knobs (§7). |

The `max_freqs=[...]` kwarg seen in examples is not consumed by `adjoint()`
itself; it is passed through so the pipelines/optimiser see the full schedule.

**Frequency continuation** is driven outside `adjoint()` by the loop:
```python
for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
    await adjoint(..., f_max=freq, max_freqs=max_freqs)
```

---

## 4. The PDE operator — `IsoAcousticDevito`

The single most important object. It is the second-order isotropic **acoustic**
wave equation on Devito. Defined in
[`iso_acoustic/devito.py:24`](stride/physics/iso_acoustic/devito.py#L24).

### 4.1 Construction (`.remote(...)`)

```python
pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
# or: IsoAcousticDevito.remote(space=space, time=time)
```

| construction kwarg | meaning |
|---|---|
| `grid` | Existing `Grid` (preferred). Alternatively pass `space=`/`time=`/`slow_time=`. |
| `len` | Number of worker replicas (`runtime.num_workers`). |
| `name` | Optional PDE name. |
| `cached_operator` | Reuse a compiled operator/grid stored in the worker warehouse across PDE instances. |
| `dev_grid` | Supply an existing `GridDevito` (advanced/shared setups). |

Class-level: `space_order = 10`, `time_order = 2`.

### 4.2 Forward-call arguments (the crux)

Whether via `forward(problem, pde, vp, **kwargs)` or a direct
`await pde(wavelets, vp, problem=sub_problem, **kwargs).result()`, these are the
inputs and options consumed by `before_forward` / `run_forward` / `after_forward`
([`iso_acoustic/devito.py:220`](stride/physics/iso_acoustic/devito.py#L220)):

**Positional / field inputs**

| arg | type | default | meaning |
|---|---|---|---|
| `wavelets` | `Traces` | required | Source wavelets (`shot.wavelets`). Supplied automatically by `forward()`/`adjoint()`. |
| `vp` | `ScalarField` | required | Compressional speed of sound, m/s. |
| `rho` | `ScalarField` | `None` | Density, kg/m³. `None` → homogeneous. |
| `alpha` | `ScalarField` | `None` | Attenuation, dB/cm. `None` → lossless. |
| `problem` | `Problem`/`SubProblem` | required | The sub-problem (one shot). Injected by the driver. |

**Physics / discretisation options**

| kwarg | type | default | meaning |
|---|---|---|---|
| `kernel` | str | auto (`'OT2'`/`'OT4'`) | Time-stepping order: `'OT2'` (2nd) or `'OT4'` (4th). Auto-chosen from `dt` if unset. |
| `boundary_type` | str | `'sponge_boundary_2'` | Absorbing boundary. Also `'complex_frequency_shift_PML_2'` (lower OT4 stability). |
| `interpolation_type` | str | `'linear'` | Source/receiver interpolation: `'linear'` (bi/tri-linear) or `'hicks'` (sinc). |
| `attenuation_power` | int / None | `0` | Power of the attenuation law when `alpha` given (`0`, `2`, or `None`). |
| `drp` | bool | `False` | Dispersion-relation-preserving coefficients (build-dependent). |
| `diff_source` | bool | `False` | Inject the source as its 1st time-derivative instead of as-is. |
| `adaptive_boxes` | bool | `False` | Adaptive computational boxes (DevitoPRO). |
| `local_prec` | bool | `True` | Local preconditioning (build-dependent). |

**Wavefield saving / gradient options**

| kwarg | type | default | meaning |
|---|---|---|---|
| `save_wavefield` | bool | auto | Save forward wavefield for the gradient. Auto-`True` when any input `needs_grad`. |
| `time_bounds` | (int,int) | `(0, time.extended_num)` | Timestep window over which the wavefield is saved. |
| `save_undersampling` | int | auto (bandwidth) | Temporal undersampling factor when saving the wavefield. |
| `save_compression` | str | `None` (2D) / `'bitcomp'` (3D) | Wavefield compression (DevitoPRO/GPU only). |
| `save_interpolation` | bool | build-dependent | Cubic-spline interpolation of the saved wavefield. |
| `stream_wavefield` | bool / str | `True` | Streaming layer strategy: `True`/`False` or `'disk'`, `'host'`, `'device'`, `'disk-host'`, `'host-device'`, `'disk-host-device'`, `'no-layers'`. |
| `spill_wavefield` | bool | `False` | Spill the wavefield to disk/host. |
| `cache_forward` | bool | `False` | Cache the forward wavefield (in memory or `cache_location`) for reuse in the adjoint. |
| `cache_location` | str | `None` | Directory to cache the wavefield to disk. |
| `nbits_compression` | int | `9` | Bits for compressed streaming (maps to `devito_args['nbits']`). |

**Wavefield dumping (debug / imaging)**

| kwarg | type | default | meaning |
|---|---|---|---|
| `dump_forward_wavefield` | bool / int | `False` | Dump forward wavefield. `True` → every `save_undersampling`; int → every N steps. |
| `dump_adjoint_wavefield` | bool / int | `False` | Same for the adjoint wavefield. |
| `dump_wavefield_id` | int | shot id | Only dump this shot's wavefields. |

**Platform / Devito plumbing**

| kwarg | type | default | meaning |
|---|---|---|---|
| `platform` | str | `None`/`'cpu'` | `None`/`'cpu'` or `'nvidia-acc'` (OpenACC) / `'nvidia-cuda'`. |
| `devito_config` | dict | `{}` | Devito config applied **before** operator generation, e.g. `{'opt': ('advanced', {'index-mode': 'int64'})}`, `{'compiler':'pgcc','language':'openacc','platform':'nvidiaX'}`. `platform='nvidia-acc'` is shorthand for the latter. |
| `devito_args` | dict | `{}` | Args passed when **calling** the compiled operator, e.g. `{'autotune': 'off'}`, `{'deviceid': N}`, `{'nbits': 9}`. |
| `deallocate` | bool | `False` | Free Devito buffers (boundary, `p`, `src`, `rec`, `vp`, …) after the run. |

### 4.3 Adjoint-call arguments

The adjoint side (`before_adjoint`/`run_adjoint`/`after_adjoint`,
[`devito.py:689`](stride/physics/iso_acoustic/devito.py#L689)) is invoked by
`fun.adjoint(**kwargs)` inside the `adjoint()` driver — you rarely call it
directly. Its inputs are `(adjoint_source, wavelets, vp, rho=None, alpha=None,
**kwargs)` where `adjoint_source` is produced by the loss. It honours
`dump_adjoint_wavefield`, `dump_wavefield_id`, `cache_forward`, `time_bounds`,
`platform`, `deallocate`, and the same physics kwargs, then returns the
gradients via the `get_grad_*` methods.

Gradients are produced per variable through the naming convention
(`prepare_grad_vp` / `init_grad_vp` / `get_grad_vp`), so `vp`, `rho`, and
`wavelets` can each be inverted for when flagged `needs_grad`
(see [`problem_type.py:202`](stride/physics/problem_type.py#L202)).

### 4.4 Calling the PDE directly

`forward()`/`adjoint()` are conveniences. You can call the tessera yourself,
which is how the homogeneous-medium test sweeps work
([`homogeneous_acoustic/forward_2D.py:109`](stride_examples/examples/homogeneous_acoustic/forward_2D.py#L109)):

```python
sub_problem = problem.sub_problem(shot.id)
traces = await pde(sub_problem.shot.wavelets, vp,
                   problem=sub_problem, diff_source=True,
                   kernel='OT4', interpolation_type='hicks',
                   boundary_type='complex_frequency_shift_PML_2',
                   rho=rho, alpha=alpha, attenuation_power=2).result()
data = traces.data                 # numpy array, shape (num_receivers, time.num)
await pde.clear_operators()        # force recompile when config changes
```

`forward()` returns nothing (it writes `shot.observed`); a direct call returns a
`Traces` object whose `.data` is the modelled gather.

### 4.5 Other PDE operators (same call convention)

- **`IsoElasticDevito`** ([`iso_elastic/devito.py:16`](stride/physics/iso_elastic/devito.py#L16)) —
  stress-strain elastic wave equation. Forward inputs are
  `(wavelets, vp, vs, rho, problem=...)`; `space_order=10`, `time_order=1`,
  `boundary_type='sponge_boundary_1'`. Multi-parameter: pass `vp, vs, rho=rho`.
- **`MarmottantDevito`** ([`marmottant/devito.py`](stride/physics/marmottant/devito.py)) —
  microbubble model.

---

## 5. Problem-definition objects

All in `stride/problem/`. These build the `problem` you hand to
`forward()`/`adjoint()`.

### `Space` — [`domain.py:9`](stride/problem/domain.py#L9)
Spatial grid = inner domain + padding.
`Space(shape, spacing, extra, absorbing)`.
- `shape` (tuple) inner grid points; `spacing` (tuple or scalar float, m);
  `extra` padding points per axis; `absorbing` portion of `extra` used for the
  boundary. Exposes `.dim`, `.limit` (physical size), `.extended_shape`,
  `.inner`, `.resample(...)`.

### `Time` — [`domain.py:263`](stride/problem/domain.py#L263)
`Time(start, step, num, stop)` — give any three. `num` must be `int`. Exposes
`.start/.step/.num/.stop`, `.extended_num`, `.extend(...)`, `.resample(...)`.

### `SlowTime` — [`domain.py:417`](stride/problem/domain.py#L417)
Frame/acquisition sampling for multi-frame data:
`SlowTime(frame_rate|frame_step, acq_rate|acq_step, num_frame, num_acq)`.

### `Grid` — [`domain.py:528`](stride/problem/domain.py#L528)
`Grid(space, time, slow_time)` — a bundle. `problem.grid` is usually what you
pass to `.remote(grid=...)` and to fields.

### `Problem` — [`problem.py:13`](stride/problem/problem.py#L13)
`Problem(name, space=, time=)` (or `grid=`). Top-level container. Attributes
auto-created: `.medium`, `.transducers`, `.geometry`, `.acquisitions`, `.grid`.
- `.sub_problem(shot_id)` → a `SubProblem` with `.shot`, `.shot_id` (built per
  shot inside the drivers).
- `.plot()`, `.load(...)`, `.dump(...)`, `.output_folder`/`.input_folder`
  (default `cwd`).
- `.space_resample(new_spacing)`, `.time_resample(new_step, new_num)` for
  multi-resolution FWI.

### `ScalarField` — [`data.py:790`](stride/problem/data.py#L790)
The medium-field type (`vp`, `rho`, `alpha`). Two construction paths:
- Forward / known model: `ScalarField(name='vp', grid=problem.grid)` then
  `.load(path)` or `.fill(value)`.
- Inversion variable: `ScalarField.parameter(name='vp', grid=problem.grid,
  needs_grad=True)`. `.parameter()` is injected by `@mosaic.tessera` and returns
  the optimisable/distributed variant; `needs_grad=True` enables gradients.
- `time_dependent=`/`slow_time_dependent=` prepend time axes.
- Members: `.data` (inner view), `.extended_data` (with padding), `.fill(v)`,
  `.plot()`, `.load()`/`.dump()`, `.needs_grad`, `.grad`, `.clear_grad()`.

### `Traces` — [`data.py:1383`](stride/problem/data.py#L1383)
Time traces indexed by transducer id — the type of `shot.wavelets` and
`shot.observed`. Write with `shot.wavelets.data[i, :] = ...`. `.data` shape is
`(num_transducers, time.num)`. `.plot(plot_type='gather'|'spectrum')`,
`.get(id)`, `.alike(...)`. `DiskTraces` is the lazy on-disk variant.

### `Medium` — [`medium.py:8`](stride/problem/medium.py#L8)
Named-field container. `problem.medium.add(vp)`; access `medium.vp` /
`medium['vp']`. `.load()/.dump()/.plot()` iterate all fields.

### `Transducers` — [`transducers.py:11`](stride/problem/transducers.py#L11)
Registry of transducer devices. `problem.transducers.default()` creates one
`PointTransducer(0)`.

### `Geometry` — [`geometry.py:106`](stride/problem/geometry.py#L106)
Transducer *locations*. `problem.geometry.default('elliptical', num_locations)`
(2D) auto-computes radius/centre from `space.limit`;
`'ellipsoidal', num_locations, radius, centre, theta=, threshold=` (3D).
`.coordinates`, `.locations`, `.num_locations`, `.plot()`.

### `Acquisitions` — [`acquisitions.py:733`](stride/problem/acquisitions.py#L733)
The set of shots (the data).
- `.default()` — one single-source shot per location, all locations as receivers.
- `.load(path=, project_name=, version=0, shot_ids=None, fast=False)`.
- `.select_shot_ids(num=, start=, end=, every=1, randomly=False)` — stateful
  per-iteration selection (this is what `select_shots` in `adjoint()` feeds).
- `.remaining_shot_ids` — shots with no observed yet (what `forward()` runs).
- `.shots`, `.shot_ids`, `.num_shots`, `.plot()`, `.reset_selection()`.

### `Shot` — [`acquisitions.py:57`](stride/problem/acquisitions.py#L57)
One acquisition event: `.wavelets` (per source), `.observed` (per receiver),
`.delays`, `.source_coordinates`, `.receiver_coordinates`,
`.num_sources`/`.num_receivers`.

### Wavelet helpers — [`utils/wavelets.py`](stride/utils/wavelets.py)
- `tone_burst(centre_freq, n_cycles, n_samples, dt, envelope='gaussian')`
- `ricker(centre_freq, n_samples, dt)`
- `continuous_wave(centre_freq, n_samples, dt, ramp_length=4, phase=0)`

### Asset fetch — [`utils/fetch.py`](stride/utils/fetch.py)
`fetch('anastasio2D', dest='data/...h5')` downloads a known release asset once.

---

## 6. Optimisation objects (inverse scripts)

### `L2DistanceLoss` — [`loss/l2_distance.py:14`](stride/optimisation/loss/l2_distance.py#L14)
`f = ½‖modelled − observed‖²`. `L2DistanceLoss.remote(len=runtime.num_workers)`.
- Ctor kwarg `d_sample=4` (downsampling of the residual passed to the functional).
- `forward(modelled, observed, **kwargs)` → `FunctionalValue`; consumes
  `problem`/`shot_id`, forwards `keep_residual`.
- `adjoint(d_fun, modelled, observed)` → `(grad_modelled, grad_observed)` — the
  adjoint source.

### `GradientDescent` / `LocalOptimiser` — [`optimisers/`](stride/optimisation/optimisers/)
> There is **no `Adam`** in the public repo — only `GradientDescent` (and the
> `LocalOptimiser` base). Adam/NAdam/RAdam/SGD/AdaBelief/AGD live in the private
> fork (§8).

`GradientDescent(variable, step_size=1., process_grad=..., process_model=...)`.
Base `LocalOptimiser` ctor kwargs:

| kwarg | default | meaning |
|---|---|---|
| `variable` | — | Variable to optimise (must be `needs_grad`). |
| `step_size` | `1.` | Float or `LineSearch`. |
| `test_step_size` | `1.` | Multiplier on the processed gradient. |
| `force_step` | `False` | Skip step clipping/capping. |
| `max_step` | `None` | Cap on step magnitude. |
| `process_grad` | `ProcessGlobalGradient(**kwargs)` | Gradient pre-processing pipeline. |
| `process_model` | `ProcessModelIteration(**kwargs)` | Model post-processing pipeline. |
| `reset_block` | `False` | Reset optimiser state each block (flag, not method). |
| `reset_iteration` | `False` | Reset optimiser state each iteration. |
| `dump_grad` / `dump_prec` | `False` | Debug dumps. |

Methods: `await step(step_size=None, grad=None, step_loop=..., **kwargs)`,
`clear_grad()`, `reset()`, `dump()/load()`.

### `OptimisationLoop` / `Block` / `Iteration` — [`optimisation_loop.py`](stride/optimisation/optimisation_loop.py)
- `OptimisationLoop(name='optimisation_loop')`. Properties `.num_blocks`,
  `.current_block`; `.blocks(num, *iters, restart=False, restart_id=-1)` is the
  generator you loop over (zips extra sequences like `max_freqs`).
- `Block.iterations(num, *iters, ...)` yields `Iteration`s; `.total_loss`, `.id`.
- `Iteration`: `.add_loss(fun)`, `.add_submitted/.add_completed(shot)`,
  `.next_run()` (line-search test steps), `.id`, `.abs_id`, `.total_loss`,
  `.prev_run`.

### Processing pipelines — [`pipelines/default_pipelines.py`](stride/optimisation/pipelines/default_pipelines.py)
`.remote(...)` operators the `adjoint()` driver builds automatically; you tune
them by passing kwargs through `adjoint()`.

| pipeline | role | default steps (kwarg → default) |
|---|---|---|
| `ProcessWavelets` | pre-process source wavelets | `check_traces`(T), `filter_traces`(T), `shift_traces`, `resonance_filter`(F) |
| `ProcessObserved` | pre-process observed | `check_traces`, `filter_traces` |
| `ProcessWaveletsObserved` | joint step | `differentiate_traces`(T) |
| `ProcessTraces` | modelled+observed before the loss | `check_traces`(T), `filter_offsets`(F), `mute_first_arrival`(T), `mute_traces`(T), `filter_traces`(T), `agc`(F), `norm_per_shot`(T)/`norm_per_trace`(F), `scale_per_*`(F), `time_tweaking`(T), `time_weighting`(T) |
| `ProcessGlobalGradient` | default `process_grad` | `mask_field`(`mask_grad`=T), `smooth_field`(`smooth_grad`=T), `norm_field`(`norm_grad`=T) |
| `ProcessModelIteration` | default `process_model` | `clip` (uses `min=`/`max=`) |

> Several `ProcessTraces` steps (`resonance_filter`, `differentiate_traces`,
> `filter_offsets`, `mute_first_arrival`, `agc`, `time_tweaking`,
> `time_weighting`) are added *non-raising*: they are no-ops in the public repo
> unless the step is registered — and those step classes ship in the **private
> fork** (§8).

### Individual steps — [`pipelines/steps/`](stride/optimisation/pipelines/steps/)
Registered (public): `filter_traces`, `norm_per_shot`, `norm_per_trace`,
`scale_per_shot`, `scale_per_trace`, `norm_field`, `smooth_field`, `mask_field`,
`mute_traces`, `clip`, `check_traces`, `dump`, `shift_traces`. Common kwargs:

| step | key kwargs |
|---|---|
| `FilterTraces` | `f_min`, `f_max`, `filter_type` (`'cos'`/`'butterworth'`/`'fir'`), `filter_relaxation` |
| `MuteTraces` | `f_max`, `filter_relaxation` |
| `Clip` | `min`, `max` |
| `SmoothField` | `smooth_sigma` (default 0.25 = 25% of a cell) |
| `NormField` | `global_norm` (F), `norm_guess_change` (0.5) |
| `MaskField` | `mask`, `mask_rampoff` (10) |
| `ScalePer*` | `scale_to`, `relative_scale` (T) |
| `CheckTraces` | `raise_incorrect` (T), `filter_incorrect` (F) |
| `ShiftTraces` | `f_max`, `filter_relaxation` |

### `LineSearch` — [`step_length/line_search.py:8`](stride/optimisation/step_length/line_search.py#L8)
Abstract step-length base (`init_search`, `next_step`). Passed as
`step_size=LineSearch(...)`; the optimiser drives it with a `step_loop` callable.
The public repo ships only the abstract base — concrete searches are in the
private fork.

---

## 7. Quick kwarg cheat-sheet

Passing these through `forward(problem, pde, vp, ...)` /
`adjoint(problem, pde, loss, loop, opt, vp, ...)` reaches the right layer:

```python
# --- performance / hardware ---
platform='nvidia-cuda'                 # or 'nvidia-acc', 'gpu', 'cpu'
devices=[0,1]                          # explicit GPUs
devito_config={'opt': ('advanced', {'index-mode': 'int64'})}
devito_args={'autotune': 'off'}
deallocate=True                        # free device buffers each shot

# --- physics ---
kernel='OT4'                           # or 'OT2'
interpolation_type='hicks'             # or 'linear', 'sinc'
boundary_type='complex_frequency_shift_PML_2'
rho=rho_field, alpha=alpha_field, attenuation_power=2
diff_source=True

# --- forward wavefield / imaging ---
save_wavefield=True, time_bounds=(0, N), save_undersampling=4
dump_forward_wavefield=True, dump_adjoint_wavefield=8, dump_wavefield_id=0
cache_forward=True, cache_location='/scratch'

# --- inversion control (adjoint only) ---
num_iters=8
select_shots=dict(num=16, randomly=True)
f_min=0.1e6, f_max=0.5e6, max_freqs=[0.3e6, 0.4e6, 0.5e6, 0.6e6]
lazy_loading=True
step_size=10                           # or a LineSearch instance

# --- pipeline toggles (adjoint only) ---
filter_traces=True, filter_wavelets=True
mask_grad=True, smooth_grad=True, norm_grad=True
smooth_sigma=0.5, norm_guess_change=0.25
```

---

## 8. Private fork (`stride-private`) — extensions & advanced examples

The private repo forks an **older** public baseline and adds specialised
operators, losses, optimisers and pipeline steps. (Its acoustic operator is
older/smaller than the current public one; where the two diverge, the private
defaults differ — e.g. `drp=True` and
`boundary_type='hybrid_interpolating_boundary_2'` by default.)

### 8.1 Advanced invocation patterns (copy-paste worthy)

**GPU multi-block acoustic FWI** — `examples/alpha2D/inverse.py:74`:
```python
await adjoint(problem, pde, loss, optimisation_loop, optimiser, vp,
              num_iters=num_iters,
              select_shots=dict(num=12, randomly=True),
              f_max=freq, max_freqs=max_freqs,
              kernel='OT4', fw3d_mode=True,
              interpolation_type='hicks', platform='nvidia-cuda')
```
matching forward `alpha2D/forward.py:62`:
```python
await forward(problem, pde, vp, kernel='OT4', fw3d_mode=True,
              interpolation_type='hicks', platform='nvidia-cuda')
```

**Elastic forward, multi-parameter + Devito opt** — `examples/alpha2D_elastic/forward.py:90`:
```python
await forward(problem, pde, vp, vs, rho=rho, shot_ids=[0],
              interpolation_type='sinc', dump=False, deallocate=False,
              dump_forward_wavefield=False,
              devito_config={'opt': ('advanced', {'index-mode': 'int64'})})
```

**3D elastic, autotune off** — `examples/alpha3D_elastic/forward.py:113`:
```python
await forward(problem, pde, vp, rho=rho, shot_ids=[0],
              interpolation_type='sinc', kernel='OT4',
              dump=False, deallocate=False, devito_args={'autotune': 'off'})
```

**Optimiser with line search** — `examples/alpha2D/inverse.py:58`:
```python
optimiser = GradientDescent(vp, step_size=LineSearch(),
                            process_grad=ProcessGlobalGradient(),
                            process_model=ProcessModelIteration(min=1450., max=3000.))
```

**Custom driver for source/receiver-IR inversion (SRI)** —
`examples/SRI/SRI_transmit_optim_wavelet.py` replaces the library `adjoint()`
with its own `adjoint_finite(problem, pde, loss, optimisation_loop, optimiser,
*args, **kwargs)` that builds `ProcessWavelets.remote(f_min=, f_max=)` /
`ProcessTraces.remote(...)`, convolves the wavelet with each transducer's
`transmit_ir` via a `Convolution.remote(...)` operator, runs the PDE, and calls
`await fun.adjoint(**kwargs)`. Manual per-worker GPU assignment:
```python
devito_args = kwargs.get('devito_args', {})
devito_args['deviceid'] = devices[worker.indices[1] % num_gpus]
kwargs['devito_args'] = devito_args
```

**Multi-resolution FWI** — `examples/alpha2D_resample/` uses
`problem.space_resample()` / `problem.time_resample()` between blocks.

### 8.2 Added physics operators
- `IsoAcousticAnalyticNumba` (analytic Green's-function acoustic, Numba
  CPU/CUDA) — extra kwargs `compute_forward`, `vp_constant`, `gradient_crop`,
  `threads_per_block`, `blocks_per_grid`, `diff_source`, `save_undersampling`.
- `TransportDevito` / `SLTransportNumpy` (semi-Lagrangian transport) — kwargs
  `interpolating`, `velocity_interpolation`, `taylor_accuracy`, `parallel_time`,
  `num_procs`, `fill_holes`; forward inputs `(sigma_0, u, ...)`.
- `FlowInjectionDevito` — kwargs `interpolation_type`, `t_i`.
- New boundaries `HybridHigdonBoundary2`, `HybridInterpolatingBoundary2`.

### 8.3 Added loss functions (`stride_private/optimisation/loss/`)
| loss | key kwargs |
|---|---|
| `AdaptiveWaveformLoss` (AWI) | `pad`, `augment`, `gamma`, `eta`, `mode='reverse'`, `type='standard'`, `d_sample=4` |
| `OptimalTransportLoss` (GSOT) | `p=2`, `mode='taot'`, `e_start/e_end/e_fac`, `taot_algorithm='dtw'`, `reg`, `max_iter`, `tol`, `d_sample` |
| `FrequencyControllableEnvelopeLoss` (FCEI) | `pad`, `p`, `filter_type='butterworth'`, `filter_order=8`, `num_modify`, `weight_l2`, `weight_fcei` |
| `ReverseTimeMigration` (RTM) | imaging condition |
| `DoubleDifferenceLoss` | double-difference misfit |
| `MultiLoss` | `weight_l2`, `weight_ot`, `weight_fcei`, `fcei_mode`, `mute_traces`, `mute_threshold`, `d_sample` |

### 8.4 Added optimisers & steps
- Optimisers: `Adam`, `NAdam`, `RAdam` (`betas=(0.9,0.999)`, `eps=1e-8`, `t`),
  `SGD`, `AdaBelief`, `AGD` — all on a `LocalOptimiserSaved` base.
- Pipeline steps (these are the ones the public default pipelines reference but
  don't ship): `AGC`, `DifferentiateTraces`, `FilterOffsets`, `MuteFirstArrival`,
  `ResonanceFilter`, `TimeTweaking`, `TimeWeighting`.
- Constraint `DivergenceFree`; helper operators `Convolution`, `Split`,
  `Replicate`, `Unflatten`, `SVDFilter`, `DiffFilter`, `FreqWindowingFilter`.
- Fullwave3D interop (`.ttr/.pgy/.vtr`) in `utils/fullwave.py`.

---

## 9. Mental model (how the pieces connect)

```
Space + Time ─► Grid ─► Problem ─┬─ medium (ScalarField vp/rho/alpha)
                                 ├─ transducers (.default)
                                 ├─ geometry (.default 'elliptical'/'ellipsoidal')
                                 └─ acquisitions (Shots: wavelets + observed)

pde = IsoAcousticDevito.remote(grid=..., len=workers)      # the physics

FORWARD:  forward(problem, pde, vp, **pde_kwargs)          # writes shot.observed

INVERSE:  loss = L2DistanceLoss.remote()
          optimiser = GradientDescent(vp.parameter(needs_grad=True),
                                       process_grad=ProcessGlobalGradient(),
                                       process_model=ProcessModelIteration(min,max))
          loop = OptimisationLoop()
          for block, freq in loop.blocks(num_blocks, max_freqs):
              adjoint(problem, pde, loss, loop, optimiser, vp,
                      num_iters=, select_shots=, f_max=freq, **pde_kwargs)
                 └─ per iter: ProcessWavelets/Observed → pde.forward (saves wavefield)
                              → ProcessTraces → loss.forward → fun.adjoint (grad)
                              → optimiser.step → dump updated vp
```

*Generated as a code reference for Stride forward/inverse scripting.*
