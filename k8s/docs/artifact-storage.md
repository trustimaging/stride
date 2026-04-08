# Artifact-Backed Distributed Inversion

This document describes the cloud-native data plane on the `dev-cloud` branch.
It covers the `ArtifactWarehouse`, lazy shot-data loading, per-iteration gradient
upload, the external accumulator service, and how to run the end-to-end
artifact-backed inversion on Minikube.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [End-to-End Flow](#end-to-end-flow)
4. [Code Changes](#code-changes)
5. [Object Key Layout](#object-key-layout)
6. [Configuration Reference](#configuration-reference)
7. [Usage](#usage)
8. [Backwards Compatibility](#backwards-compatibility)

---

## Overview

### The Problem

The original distributed inversion transfers all shot data through the mosaic
warehouse (in-memory ZMQ). At scale this has three problems:

- All observed traces are held in the head pod's RAM
- Mosaic serialises and ships the full `Traces` array to every worker for every shot
- Gradient accumulation requires a synchronous ZMQ barrier (`__redux_adjoint__`)
  between all workers and the head after every iteration

### The Solution

Shot data and gradients are stored in MinIO (S3-compatible). Workers fetch observed
traces lazily — only their assigned shots, only when needed. After the adjoint run each
worker uploads its gradient directly to MinIO. An external accumulator service polls
for all worker files, sums them, and writes a final gradient. The head polls for that
file and feeds it to the optimiser, bypassing the ZMQ barrier entirely.

Key properties:

- **No kwargs plumbing**: the warehouse is registered globally via
  `mosaic.set_artifact_warehouse()`. All processes (head, workers, accumulator) find
  it via `mosaic.get_artifact_warehouse()` — no config object passed around.
- **Auto-configured**: `mrun` calls `ArtifactWarehouse.from_env()` automatically
  when `ARTIFACT_ENDPOINT` is set in the environment.
- **Run-scoped**: all objects are stored under a run-specific prefix
  (`ARTIFACT_RUN_ID`, set to the Argo workflow name) so multiple concurrent runs
  share the same bucket without collision.
- **Iteration-aware**: each worker receives the current iteration index through
  `func_kwargs` so gradient files land in the correct `iter_N/` prefix.
- **Stale-connection safe**: the MinIO client uses a 30 s read timeout so that
  idle connections between iterations are detected and retried quickly.
- **Clean exit**: the accumulator knows the total number of iterations upfront
  (`NUM_ITERS`) and exits with code 0 after processing all of them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Head Pod                                               │
│                                                         │
│  forward()                                              │
│    worker runs PDE → returns observed traces            │
│    upload {run}/shots/{id}/observed.npy  ────────────►  │
│    upload {run}/shots/{id}/wavelets.npy  ────────────►  │──► MinIO
│    shot.observed = ArtifactTraces(key)  (no RAM)        │
│                                                         │
│  load_artifacts()                                       │
│    download wavelets.npy (eager, small)                 │
│    shot.observed = ArtifactTraces(key)  (lazy)          │
│                                                         │
│  adjoint() — per iteration                              │
│    set_iteration(i) on head warehouse                   │
│    dispatch shots to workers (ArtifactTraces in         │
│    sub_problem; workers __reduce__ → download observed) │
│    poll pull_remote("{run}/gradients/iter_i/final.pkl") │◄── MinIO
│    optimiser.step(grad)                                 │
└─────────────────────────────────────────────────────────┘
           │                              ▲
           │ mosaic RPC (shot dispatch)   │ poll final.pkl
           ▼                              │
┌──────────────────────────┐    ┌─────────────────────────┐
│  Worker Pod K            │    │  Accumulator Pod         │
│                          │    │                          │
│  receives sub_problem    │    │  for i in range(NUM_ITERS│
│  ArtifactTraces.__reduce_│    │    polls iter_i/ prefix  │
│    downloads observed.npy│    │    folds worker_K.pkl    │
│  runs PDE forward+adjoint│    │    files in one by one   │
│  exec_remote() uploads   │    │    writes final.pkl ────►│──► MinIO
│    {run}/gradients/iter_i│    │  exits cleanly           │
│    /worker_K.pkl ────────┼──────────────────────────────────► MinIO
└──────────────────────────┘    └─────────────────────────┘
```

---

## End-to-End Flow

### Phase 1 — Forward pass

```python
problem.acquisitions.default()         # create shots (plain Traces objects)
shot.wavelets.data[:] = tone_burst()   # set source signals
await forward(problem, pde, vp_true)
```

Inside `forward()` (`stride/__init__.py`):

1. `art_warehouse.ensure_bucket()` — creates the MinIO bucket if absent
2. Worker runs the PDE, returns observed traces to the head
3. `shot.observed.data[:] = traces.data` — written into the plain `Traces` object
4. `push_remote('{run}/shots/{id}/observed.npy', data)` — uploaded to MinIO
5. `push_remote('{run}/shots/{id}/wavelets.npy', data)` — uploaded to MinIO
6. `shot.observed = ArtifactTraces(artifact_key=key)` — head drops the array,
   keeps only the S3 key

### Phase 2 — Inversion

```python
problem.acquisitions.default()         # create new shots (plain Traces)
problem.acquisitions.load_artifacts()  # wire artifact data
for block in optimisation_loop.blocks(NUM_ITERS):
    await adjoint(problem, pde, loss, optimisation_loop, optimiser, vp,
                  num_iters=1, select_shots=dict(num=N))
```

`load_artifacts()` (`stride/problem/acquisitions.py`):

- Downloads `wavelets.npy` eagerly per shot and writes into `shot.wavelets.data[:]`
- Replaces `shot.observed` with an `ArtifactTraces(artifact_key=...)` (lazy)

Inside `adjoint()` per iteration (`stride/__init__.py`):

1. `art_warehouse.set_iteration(iteration.abs_id)` — head tracks current iteration
2. `_kwargs['_abs_iteration'] = iteration.abs_id` — injected into per-shot kwargs
3. Workers receive the shot sub-problem; `ArtifactTraces._deserialisation_helper()`
   fires on the worker, downloads `observed.npy` from MinIO into a plain `Traces`
4. Worker runs PDE forward + adjoint; `Variable.adjoint()` calls
   `runtime.exec('redux-...', redux, grads, func_kwargs={'iteration': i})`
5. Worker's `ArtifactWarehouse.exec_remote()` pops `iteration` from `func_kwargs`,
   uploads gradient as `{run}/gradients/iter_{i}/worker_{K}.pkl`
6. Head calls `variable.pull(attr='grad')` → `tessera.pull()` →
   `pull_remote('{run}/gradients/iter_{i}/final.pkl', poll=True)` — blocks
7. Accumulator detects worker files as they arrive, folds each into a running sum,
   writes `final.pkl` once all `NUM_WORKERS` files are present
8. Head receives gradient, runs `optimiser.step()`

### Accumulator Service

Runs as a daemon pod via `python3 -m mosaic.runtime.gradient_accumulator`.
Reads `NUM_ITERS` and `NUM_WORKERS` from env, then processes exactly that many
iterations before exiting cleanly with code 0:

```
GradientAccumulator.from_env().run()
  → for i in range(num_iters):
      accumulate_iteration(i)
        poll until each worker_{K}.pkl arrives, fold into running sum immediately
        upload final.pkl
exit 0
```

Files are folded in as they arrive (pairwise sum) rather than downloading all at once,
so peak memory is two gradient arrays regardless of worker count.

---

## Code Changes

### New Files

#### `mosaic/runtime/artifact_warehouse.py`

`ArtifactWarehouse` — S3-backed warehouse that runs inline (no subprocess).

```python
class ArtifactWarehouse:
    @classmethod
    def from_env(cls, prefix='ARTIFACT') -> 'ArtifactWarehouse':
        # reads ARTIFACT_ENDPOINT, _ACCESS_KEY, _SECRET_KEY, _BUCKET, _SECURE,
        # _RUN_ID, _GRADIENT_PREFIX, _SHOT_PREFIX

    @property
    def gradient_prefix(self) -> str:
        # returns '{run_prefix}/gradients' when ARTIFACT_RUN_ID is set

    @property
    def shot_prefix(self) -> str:
        # returns '{run_prefix}/shots' when ARTIFACT_RUN_ID is set

    def ensure_bucket(self) -> None:
        # creates bucket if it does not exist

    def push_remote(self, key: str, data) -> ArtifactWarehouseObject:
        # numpy arrays → .npy format; everything else → pickle

    def pull_remote(self, key: str, poll: bool = False):
        # .npy keys → np.load; others → pickle.loads
        # poll=True: blocks with exponential backoff until key exists

    async def exec_remote(self, uid, func, func_args=None, func_kwargs=None):
        # pops 'iteration' from func_kwargs (default: self._iteration)
        # calls redux closure → extracts gradient → uploads worker_{K}.pkl

    @property
    def client(self):
        # lazily initialised Minio client with Timeout(connect=5, read=30)
        # and Retry(total=5, backoff_factor=0.2, status_forcelist=[500-504])
```

`ArtifactWarehouseObject` is a lightweight `(key, bucket)` handle mirroring
`WarehouseObject` for the SpillBuffer.

#### `mosaic/runtime/gradient_accumulator.py`

`GradientAccumulator` — daemon class that runs as its own pod. Reads `NUM_ITERS`
and `NUM_WORKERS` from env via `from_env()`. For each iteration, polls MinIO and
folds worker files into a running sum as they arrive (two arrays in memory at a
time), then writes `final.pkl`. Invokable directly as a module:

```python
class GradientAccumulator:
    @classmethod
    def from_env(cls) -> 'GradientAccumulator':
        # calls ArtifactWarehouse.from_env(), then reads NUM_WORKERS, NUM_ITERS

    def accumulate_iteration(self, iteration: int) -> None:
        # polls iter_{i}/ prefix; folds each worker_{K}.pkl into running sum;
        # uploads final.pkl once all NUM_WORKERS files have been folded in

    def run(self) -> None:
        # loops over range(num_iters), calling accumulate_iteration(i) each time

# Entry point: python3 -m mosaic.runtime.gradient_accumulator
if __name__ == '__main__':
    logging.basicConfig(...)
    GradientAccumulator.from_env().run()
```

### Modified Files

#### `mosaic/__init__.py`

Added global warehouse registry:

```python
_artifact_warehouse = None

def set_artifact_warehouse(warehouse): ...
def get_artifact_warehouse(): ...
```

#### `mosaic/runtime/__init__.py`

Exports the `artifact_warehouse` and `gradient_accumulator` modules.

#### `mosaic/cli/mrun.py`

Auto-detects `ARTIFACT_ENDPOINT` env var and configures the global warehouse:

```python
if os.environ.get('ARTIFACT_ENDPOINT'):
    from mosaic.runtime.artifact_warehouse import ArtifactWarehouse
    mosaic.set_artifact_warehouse(ArtifactWarehouse.from_env())
```

#### `mosaic/runtime/runtime.py`

`exec()` routes to artifact warehouse when configured:

```python
async def exec(self, uid, func, func_args=None, func_kwargs=None):
    art_warehouse = mosaic.get_artifact_warehouse()
    if art_warehouse is not None:
        return await art_warehouse.exec_remote(uid, func,
                                               func_args=func_args,
                                               func_kwargs=func_kwargs)
    # ... existing SpillBuffer path
```

#### `mosaic/core/tessera.py`

`pull(attr='grad')` polls `final.pkl` from S3 when warehouse is set:

```python
if attr == 'grad' and mosaic.get_artifact_warehouse() is not None:
    warehouse = mosaic.get_artifact_warehouse()
    key = '%s/iter_%d/final.pkl' % (warehouse.gradient_prefix, warehouse.iteration)
    grad = warehouse.pull_remote(key, poll=True)
    self.grad = grad
    return
```

#### `stride/problem/data.py`

`ArtifactTraces` — lazy `Traces` subclass backed by S3. Stores only `_artifact_key`.
`_data` always returns `None` and its setter is a no-op, so the head never holds the
array in RAM. On the worker, `_deserialisation_helper()` downloads the array from
MinIO and returns a plain `Traces` object.

#### `stride/problem/acquisitions.py`

`Shot._traces()` always returns plain `Traces` (or `DiskTraces`). `ArtifactTraces`
is only created explicitly in `load_artifacts()` and inside `forward()`.

`Acquisitions.load_artifacts()` — downloads wavelets eagerly, wires observed lazily:

```python
def load_artifacts(self, shot_ids=None) -> None:
    warehouse = mosaic.get_artifact_warehouse()
    for shot_id in ...:
        shot.wavelets.data[:] = warehouse.pull_remote(
            '%s/%d/wavelets.npy' % (warehouse.shot_prefix, shot_id))
        shot.observed = ArtifactTraces(
            artifact_key='%s/%d/observed.npy' % (warehouse.shot_prefix, shot_id), ...)
```

#### `stride/__init__.py`

`forward()` — uploads per-shot data to MinIO then replaces `shot.observed`:

```python
if art_warehouse is not None:
    art_warehouse.push_remote('%s/%d/observed.npy' % (art_warehouse.shot_prefix, id), ...)
    art_warehouse.push_remote('%s/%d/wavelets.npy' % (art_warehouse.shot_prefix, id), ...)
    shot.observed = ArtifactTraces(artifact_key=key_obs, ...)
```

`adjoint()` — sets iteration on head warehouse and injects `_abs_iteration` into
per-shot kwargs so workers know which `iter_N/` prefix to write to:

```python
if art_warehouse is not None:
    art_warehouse.set_iteration(iteration.abs_id)

async def loop(worker, shot_id):
    _kwargs = kwargs.copy()
    if art_warehouse is not None:
        _kwargs['_abs_iteration'] = iteration.abs_id
    ...
    fun_value = await fun.remote.adjoint(**_kwargs).result()
```

#### `stride/core.py`

`Variable.adjoint()` — pops `_abs_iteration` from kwargs (so operators don't see it),
passes it to `runtime.exec()` as `func_kwargs`:

```python
_abs_iter = kwargs_.pop('_abs_iteration', None)
_fkw = {'iteration': _abs_iter} if _abs_iter is not None else None
redux_grad = await runtime.exec('redux-%s' % node.op.uid, redux,
                                output_grads, func_kwargs=_fkw)
```

Also skips the `__redux_adjoint__` ZMQ barrier when the artifact warehouse is active:

```python
if redux and mosaic.get_artifact_warehouse() is not None:
    return   # accumulator service handles cross-worker summation
```

#### `k8s/scripts/simple_inverse_artifacts.py`

Reads `NUM_ITERS` from env so the iteration count stays in sync with the workflow
parameter. Uses `blocks(NUM_ITERS)` with `num_iters=1`:

```python
NUM_ITERS = int(os.environ.get('NUM_ITERS', '4'))
...
for block in optimisation_loop.blocks(NUM_ITERS):
    await adjoint(..., num_iters=1, select_shots=dict(num=N))
```

#### `k8s/workflows/stride-workflow.yaml`

- Added `gradient-accumulator` daemon task (depends only on `create-service`);
  command is now `python3 -m mosaic.runtime.gradient_accumulator`
- Added `num-iters` workflow parameter (default `"4"`)
- Added `ARTIFACT_*` and `NUM_ITERS` env vars to head, worker, and accumulator specs:

```yaml
- name: ARTIFACT_ENDPOINT
  value: "minio.argo.svc.cluster.local:9000"
- name: ARTIFACT_ACCESS_KEY
  value: "admin"
- name: ARTIFACT_SECRET_KEY
  value: "password"
- name: ARTIFACT_BUCKET
  value: "stride-data"
- name: ARTIFACT_RUN_ID
  value: "{{workflow.name}}"
- name: NUM_ITERS
  value: "{{workflow.parameters.num-iters}}"
```

---

## Object Key Layout

All objects are stored under a run-specific prefix derived from the Argo workflow
name, so concurrent runs share the same bucket without collision.

```
stride-data/
└── stride-x7k2p/              ← ARTIFACT_RUN_ID = workflow name, unique per run
    ├── shots/
    │   ├── 0/
    │   │   ├── observed.npy   ← uploaded during forward pass (lazy on worker)
    │   │   └── wavelets.npy   ← uploaded during forward pass (eager on head)
    │   └── 1/
    │       ├── observed.npy
    │       └── wavelets.npy
    └── gradients/
        ├── iter_0/
        │   ├── worker_0.pkl   ← uploaded by worker 0 via exec_remote()
        │   ├── worker_1.pkl   ← uploaded by worker 1 via exec_remote()
        │   └── final.pkl      ← written by accumulator; head polls for this
        ├── iter_1/
        │   └── ...
        └── iter_{N-1}/
            └── ...
```

Gradient files use `.pkl` (pickle) because the gradient is a raw numpy array
extracted from a `ScalarField`. Shot data uses `.npy` for efficiency.

---

## Configuration Reference

All configuration is via environment variables, read by `ArtifactWarehouse.from_env()`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_ENDPOINT` | *(required)* | MinIO / S3 host:port |
| `ARTIFACT_ACCESS_KEY` | `minioadmin` | Access key |
| `ARTIFACT_SECRET_KEY` | `minioadmin` | Secret key |
| `ARTIFACT_BUCKET` | `stride-data` | Bucket name |
| `ARTIFACT_SECURE` | `false` | Use HTTPS (`true`/`false`) |
| `ARTIFACT_RUN_ID` | `''` | Run-scoped key prefix (set to `{{workflow.name}}`) |
| `ARTIFACT_GRADIENT_PREFIX` | `gradients` | Key prefix for gradient objects |
| `ARTIFACT_SHOT_PREFIX` | `shots` | Key prefix for shot data |

The bucket is auto-created by `ensure_bucket()` if it does not exist. This is called
at the start of `forward()` on the head and inside `GradientAccumulator.from_env()`.

Workflow-level parameters (set in `stride-workflow.yaml` or overridden at submit time):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-workers` | `2` | Number of worker pods |
| `num-iters` | `4` | Total gradient accumulation iterations |
| `exp-name` | `simple` | Experiment name (used for output folder) |

---

## Usage

### Kubernetes (Minikube)

```bash
# Build and submit
eval $(minikube docker-env)
docker build -t stride-k8s:latest .
argo submit k8s/workflows/stride-workflow.yaml \
    -p exp-name=my-exp -p num-iters=8 -n argo --watch

# Stream logs
argo logs @latest -n argo -f

# Kill and delete
argo terminate @latest -n argo
argo delete @latest -n argo

# Delete all workflows
argo delete --all -n argo
```

Expected output (head pod):
```
Phase 1: Running forward pass...
Phase 1 complete. Observed traces uploaded to artifact store.
Phase 2: Starting inversion...
Loaded 2 shots via artifact store.
Beginning optimisation loop.
Done iteration 1 (out of 1), block 1 (out of 4) - Total loss ...
Done iteration 1 (out of 1), block 2 (out of 4) - Total loss ...
...
Inversion complete.
Final model range: [xxxx.x, xxxx.x] m/s
```

Expected output (accumulator pod):
```
[mosaic.runtime.gradient_accumulator 13:25:48] Started — 4 iteration(s), 2 worker(s) per iteration, prefix='stride-x7k2p/gradients'.
[mosaic.runtime.gradient_accumulator 13:25:49] Iter 0 — waiting for 2 file(s).
[mosaic.runtime.gradient_accumulator 13:25:50] Iter 0 — downloading worker_0.pkl (1/2).
[mosaic.runtime.gradient_accumulator 13:25:50] Iter 0 — folded worker_0 (1/2 accumulated).
[mosaic.runtime.gradient_accumulator 13:25:51] Iter 0 — downloading worker_1.pkl (2/2).
[mosaic.runtime.gradient_accumulator 13:25:51] Iter 0 — folded worker_1 (2/2 accumulated).
[mosaic.runtime.gradient_accumulator 13:25:51] Iter 0 done — final.pkl written (1.23s).
...
[mosaic.runtime.gradient_accumulator 13:26:30] All 4 iteration(s) complete. Exiting.
```

---

## Backwards Compatibility

All changes are additive. When `ARTIFACT_ENDPOINT` is not set, `mrun` does not
call `set_artifact_warehouse()`, `get_artifact_warehouse()` returns `None`, and
all new code paths are skipped. Existing scripts, tests, and the standard
`simple_inverse.py` workflow are completely unaffected.

| Existing usage | Impact |
|----------------|--------|
| `forward(problem, pde, vp)` | Zero — no warehouse → old path |
| `adjoint(problem, pde, ...)` | Zero — no warehouse → old path |
| `DiskTraces` / `lazy_loading=True` | Untouched |
| `problem.acquisitions.load(path=...)` | Unchanged |
| Existing tests | Unchanged |
