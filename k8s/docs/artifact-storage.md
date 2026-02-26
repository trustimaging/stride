# Artifact-Backed Lazy Data Loading

This document describes the cloud-native data persistence layer added to Stride on the
`dev-cloud` branch. It covers lazy loading of shot traces, per-iteration gradient
upload, MinIO deployment, and how to run the end-to-end artifact-backed inversion.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Code Changes](#code-changes)
4. [Object Key Layout](#object-key-layout)
5. [Configuration Reference](#configuration-reference)
6. [Usage](#usage)
7. [Backwards Compatibility](#backwards-compatibility)
8. [Out of Scope / Future Work](#out-of-scope--future-work)

---

## Overview

### The Problem

The original distributed inversion workflow transfers all shot data through the mosaic
warehouse (in-memory). This becomes a bottleneck at scale:

- All observed traces are held in the head pod's memory
- Mosaic serialises and ships the full `Traces` array to each worker for every shot
- Workers cannot independently fetch only the data they need

### The Solution: Artifact Store

Shot data and gradients are stored in a cloud-native object store (MinIO for
Minikube testing; any S3-compatible service long-term). Workers fetch observed
traces lazily — only the shots assigned to them, only when they need it. After
each iteration they also push their accumulated gradient directly to MinIO via
`GradientSink`, a mosaic tessera that runs on the worker and uploads
`gradients/iter_N/worker_K.npy` without any data passing through the head.

```
Phase 1 — Forward pass (head)
  worker computes observed traces
    → upload shots/N/observed.npy to MinIO
    → upload shots/N/wavelets.npy to MinIO
  shot.observed replaced with ArtifactTraces(key)   ← metadata only, no RAM

Phase 2 — Adjoint loop (per iteration)
  load_artifacts() downloads wavelets eagerly from MinIO (small; needed upfront)
  load_artifacts() wires shot.observed as lazy ArtifactTraces
  head publishes model (vp) to workers via mosaic warehouse (unchanged)
  worker receives ArtifactTraces → __reduce__ fires → downloads observed.npy from MinIO
  worker runs PDE forward + backward, accumulates gradient locally
  head dispatches GradientSink.upload() to each worker
  each worker uploads its gradient → gradients/iter_N/worker_K.npy  (independently)
  optimiser.step() pulls gradient internally (unchanged) and applies model update
```

Key properties:
- **Lazy**: workers only download data when their shot actually executes
- **Independent**: no worker coordination; each pod fetches its own shot and uploads its own gradient
- **Offloaded upload**: the head never reads or aggregates gradient data; each worker pushes its own file
- **Backend-agnostic**: named `artifact` not `s3`; MinIO client is S3-compatible

---

## Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  Head Pod                                                            │
│                                                                      │
│  forward()                                                           │
│    worker computes observed ──► upload observed.npy ──► MinIO        │
│                              ──► upload wavelets.npy ──► MinIO       │
│    shot.observed = ArtifactTraces(key)    (metadata only)            │
│                                                                      │
│  load_artifacts()                                                    │
│    download wavelets.npy from MinIO (eager, small)                   │
│    shot.observed = ArtifactTraces(key)  (lazy)                       │
│                                                                      │
│  adjoint()                                                           │
│    publish vp to workers (unchanged mosaic warehouse path)           │
│    per iteration: dispatch GradientSink.upload() to each worker ─────┼──┐
│    optimiser.step() pulls gradient internally (unchanged)            │  │
└──────────────────────────────────────────────────────────────────────┘  │
                │                                                          │ RPC call
  mosaic pickle │  ArtifactTraces.__reduce__                               │ (mosaic)
                ▼                                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Worker Pod K                                                                │
│                                                                              │
│  receives ArtifactTraces → _deserialisation_helper()                         │
│    → download_array() from MinIO → plain Traces with real data               │
│  runs PDE forward + backward, accumulates gradient in variable.grad          │
│  GradientSink.upload() ──► upload worker_K.npy ──────────────────────────►  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │  MinIO   │
                              │  (argo   │
                              │   ns)    │
                              └──────────┘
```

### ArtifactTraces: Lazy Loading Pattern

`ArtifactTraces` mirrors `DiskTraces` exactly. The only difference is the
storage backend:

| Aspect | `DiskTraces` | `ArtifactTraces` |
|--------|-------------|-----------------|
| Storage | Local filesystem | MinIO / S3-compatible |
| `_data` property | always returns `None` | always returns `None` |
| `load()` | `np.load(path)` | `download_array(client, bucket, key)` |
| Pickling | standard `__reduce__` | custom `__reduce__` → downloads on worker |
| Head memory | zero (path only) | zero (connection metadata only) |

The critical insight: by overriding `_data` to always return `None`, the array is
never held in the head's RAM. When mosaic pickles the object for a worker, the
custom `__reduce__` triggers the artifact store download on the worker side.

---

## Code Changes

### New Files

#### `stride/utils/artifacts.py`

Service-agnostic artifact store utilities.

```python
@dataclass
class ArtifactConfig:
    endpoint: str          # e.g. "minio.argo.svc.cluster.local:9000"
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    backend: str = 'minio'           # 'minio' | future: 's3', 'gcs'
    shot_prefix: str = 'shots'
    gradient_prefix: str = 'gradients'

    @classmethod
    def from_env(cls: type[ArtifactConfig], prefix: str = 'ARTIFACT') -> ArtifactConfig:
        # reads ARTIFACT_ENDPOINT, ARTIFACT_ACCESS_KEY, ARTIFACT_SECRET_KEY,
        # ARTIFACT_BUCKET, ARTIFACT_SECURE, ARTIFACT_BACKEND
        ...

def get_client(config: ArtifactConfig) -> Minio: ...
def ensure_bucket(client: Minio, bucket: str) -> None: ...
def upload_array(client: Minio, bucket: str, key: str, array: np.ndarray) -> None: ...
def download_array(client: Minio, bucket: str, key: str) -> np.ndarray: ...
```

All arrays are serialised as `.npy` via `BytesIO` — no HDF5/h5py dependency on
workers. `minio` is imported lazily inside `get_client()` to avoid import errors
on machines without the package.

#### `stride/utils/gradient_sink.py`

A `@mosaic.tessera`-decorated class. One instance is created per worker via
`GradientSink.remote(variable=optimiser.variable, len=runtime.num_workers)`.
When the head calls `gradient_sink.upload(...)`, mosaic dispatches the call to the
worker that owns that tessera instance.

```python
@mosaic.tessera
class GradientSink(Operator):
    def __init__(self, variable: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._variable = variable   # resolved to worker-local instance by mosaic

    def upload(self, iteration_id: int, worker_id: int,
               endpoint: str, access_key: str, secret_key: str,
               bucket: str, gradient_prefix: str,
               secure: bool = False, **kwargs: Any) -> None:
        # Runs ON THE WORKER
        config = ArtifactConfig(endpoint=endpoint, ...)
        key = '%s/iter_%d/worker_%d.npy' % (gradient_prefix, iteration_id, worker_id)
        upload_array(get_client(config), bucket, key,
                     np.asarray(self._variable.grad.data))
```

`numpy` and `stride.utils.artifacts` are imported lazily inside `upload()` so they
are only loaded on the worker at call time, not at module import time on the head.

#### `k8s/manifests/minio.yaml`

MinIO deployment in the `argo` namespace:
- **PVC**: 5 Gi `ReadWriteOnce` (persists across pod restarts)
- **StatefulSet**: `minio/minio:latest`, `server /data --console-address :9001`
- **Service** (ClusterIP): port 9000 (S3 API), port 9001 (web console)
- **Credentials**: set via `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` env vars (match these to `ARTIFACT_ACCESS_KEY` / `ARTIFACT_SECRET_KEY` in the workflow)
- **DNS**: `minio.argo.svc.cluster.local:9000`

#### `k8s/scripts/simple_inverse_artifacts.py`

End-to-end 2D acoustic FWI using the artifact store:

```
Phase 1: forward(problem_fwd, pde_fwd, vp_true, artifact_config=artifact_config)
         → uploads shots/{N}/observed.npy and shots/{N}/wavelets.npy for each shot

Phase 2: problem.acquisitions.default()          # create shots with geometry
         problem.acquisitions.load_artifacts(artifact_config)
           → downloads wavelets.npy per shot (eager)
           → wires shot.observed as ArtifactTraces (lazy)
         adjoint(problem, pde, loss, ..., artifact_config=artifact_config)
           → workers fetch observed lazily from MinIO
           → gradient uploaded per iteration
```

---

### Modified Files

#### `stride/problem/data.py` — `ArtifactTraces` class (line 1811)

New class added after `DiskTraces`. Key design:

```python
class ArtifactTraces(Traces):
    def __init__(self, **kwargs) -> None:
        # Store connection metadata; discard any 'data' kwarg
        artifact_config = kwargs.pop('artifact_config', None)
        self._artifact_endpoint = artifact_config.endpoint
        self._artifact_access_key = artifact_config.access_key
        # ... etc
        self._artifact_key = kwargs.pop('artifact_key', None)
        kwargs.pop('data', None)   # never hold data on head
        super().__init__(**kwargs)

    # ── Lazy data property ──────────────────────────────────────────
    @property
    def _data(self): return None    # head always data-free
    @_data.setter
    def _data(self, value): pass    # suppress parent assignments

    # ── Materialise from artifact store ─────────────────────────────
    def load(self, **kwargs) -> Traces:
        config = ArtifactConfig(endpoint=self._artifact_endpoint, ...)
        client = get_client(config)
        data = download_array(client, self._artifact_bucket, self._artifact_key)
        return Traces(data=data, transducer_ids=self.transducer_ids, grid=self.grid)

    # get() / get_extended() / plot() delegate to self.load()

    # ── Pickle path: worker downloads on deserialisation ────────────
    @classmethod
    def _deserialisation_helper(cls, state: dict) -> Traces:
        # Downloads .npy → returns plain Traces (not ArtifactTraces)
        config = ArtifactConfig(endpoint=state.pop('_artifact_endpoint'), ...)
        key = state.pop('_artifact_key')
        data = download_array(get_client(config), config.bucket, key)
        instance = Traces.__new__(Traces)
        instance._data = data
        for attr, value in state.items():
            setattr(instance, attr, value)
        return instance

    def __reduce__(self):
        return self._deserialisation_helper, (self._serialisation_helper(),)
```

`_serialisation_attrs` includes all connection metadata plus normal `Traces`
attributes (`name`, `uname`, shape fields, `_transducer_ids`, `_grid`, etc.) so
the deserialised `Traces` has full grid/transducer context.

#### `stride/problem/acquisitions.py`

**`Shot._traces()`** extended with artifact branch:

```python
def _traces(self, *args, **kwargs):
    artifact_config = kwargs.pop('artifact_config', None)
    if artifact_config is not None:
        key = '%s/%d/%s.npy' % (artifact_config.shot_prefix, self.id,
                                 kwargs.get('name', 'traces'))
        return ArtifactTraces(*args, **kwargs,
                              artifact_config=artifact_config,
                              artifact_key=key)
    elif kwargs.pop('lazy_loading', False):
        return DiskTraces(...)   # unchanged
    else:
        return Traces(...)       # unchanged
```

**`Acquisitions.load_artifacts()`** (new method):

```python
def load_artifacts(self, artifact_config: ArtifactConfig,
                   shot_ids: Optional[List[int]] = None) -> None:
    client = get_client(artifact_config)
    for shot_id in (shot_ids or list(self._shots.keys())):
        shot = self._shots[shot_id]

        # Wavelets — download eagerly (small; needed by the worker before the PDE runs)
        key_wav = '%s/%d/wavelets.npy' % (artifact_config.shot_prefix, shot_id)
        shot.wavelets.data[:] = download_array(client, artifact_config.bucket, key_wav)

        # Observed — lazy via ArtifactTraces; worker fetches on demand
        key_obs = '%s/%d/observed.npy' % (artifact_config.shot_prefix, shot_id)
        shot.observed = ArtifactTraces(
            name='observed',
            transducer_ids=shot.receiver_ids,
            grid=shot.grid,
            artifact_config=artifact_config,
            artifact_key=key_obs,
        )
```

`acquisitions.default()` must be called first (to create shots with allocated wavelet
arrays). `load_artifacts()` then fills in data from MinIO. This mirrors the interface
of `acquisitions.load()`, which reads both wavelets and observed from local HDF5 files.

#### `stride/__init__.py` — `forward()` changes

New keyword argument: `artifact_config=None`

```python
# After shot.observed is computed:
if artifact_config is not None:
    key_obs = '%s/%d/observed.npy' % (artifact_config.shot_prefix, shot_id)
    upload_array(artifact_client, artifact_config.bucket, key_obs, shot.observed.data)

    key_wav = '%s/%d/wavelets.npy' % (artifact_config.shot_prefix, shot_id)
    upload_array(artifact_client, artifact_config.bucket, key_wav, shot.wavelets.data)

    shot.observed = ArtifactTraces(
        name=shot.observed.name,
        transducer_ids=shot.observed.transducer_ids,
        grid=shot.observed.grid,
        artifact_config=artifact_config,
        artifact_key=key_obs,
    )
```

Both observed traces and wavelets are uploaded per shot. `artifact_client` is created
once before the shot loop via `get_client()` and `ensure_bucket()`. Wavelets are
uploaded here so that `load_artifacts()` in Phase 2 can be fully self-contained —
no local HDF5 files or disk access needed.

#### `stride/__init__.py` — `adjoint()` changes

New keyword argument: `artifact_config=None`

**`GradientSink` setup** (once, before the iteration loop):

```python
gradient_sink = None
if artifact_config is not None:
    gradient_sink = GradientSink.remote(
        variable=optimiser.variable,
        len=runtime.num_workers,
    )
```

Creating it outside the loop means mosaic instantiates the tessera on each worker
once at the start, rather than once per iteration.

**Per-worker upload dispatch** (after each iteration's shot loop, before `step()`):

```python
if gradient_sink is not None:
    @runtime.async_for(list(range(runtime.num_workers)), safe=safe)
    async def upload_loop(worker, _dummy):
        await gradient_sink.upload(
            iteration.abs_id,
            worker.indices[0],          # actual worker index → key worker_K.npy
            artifact_config.endpoint,
            artifact_config.access_key,
            artifact_config.secret_key,
            artifact_config.bucket,
            artifact_config.gradient_prefix,
            artifact_config.secure,
            runtime=worker,
        ).result()
    await upload_loop
```

`worker.indices[0]` is the worker's actual runtime index, so each worker's file is
named correctly regardless of mosaic's scheduling order.

`optimiser.step()` is called unchanged immediately after — it still does its own
internal `pull(attr='grad')` to aggregate the gradient for the model update. The
MinIO upload is independent of that path.

#### `stride/utils/__init__.py`

```python
from .artifacts import ArtifactConfig
from .gradient_sink import GradientSink
```

Both `ArtifactConfig` and `GradientSink` are importable directly from `stride.utils`.

#### `environment.yml`

Added `minio` to the `pip:` section so it is installed in the Docker image.

#### `k8s/workflows/stride-workflow.yaml`

Added to both head and worker container env sections:

```yaml
- name: ARTIFACT_ENDPOINT
  value: "minio.argo.svc.cluster.local:9000"
- name: ARTIFACT_ACCESS_KEY
  value: "minioadmin"
- name: ARTIFACT_SECRET_KEY
  value: "minioadmin"
- name: ARTIFACT_BUCKET
  value: "stride-data"
- name: ARTIFACT_BACKEND
  value: "minio"
```

#### `k8s/run.sh`

Added `deploy_minio()` function and `minio` subcommand. Included in `setup()` flow
after `setup_rbac` and before `build_image`:

```bash
deploy_minio() {
    kubectl apply -f "$SCRIPT_DIR/manifests/minio.yaml"
    kubectl wait -n "$ARGO_NAMESPACE" --for=condition=ready pod \
        -l app=minio --timeout=120s
}
```

#### `k8s/scripts/k8s_runner.py`

Updated `SCRIPTS` map: `'inverse_s3'` → `'inverse_artifacts'`

```python
SCRIPTS = {
    'forward':            'scripts.simple_forward',
    'inverse':            'scripts.simple_inverse',
    'inverse_artifacts':  'scripts.simple_inverse_artifacts',
}
```

---

## Object Key Layout

```
stride-data/
├── shots/
│   ├── 0/
│   │   ├── observed.npy      ← uploaded during forward pass (lazy on worker)
│   │   └── wavelets.npy      ← uploaded during forward pass (eager on head)
│   ├── 1/
│   │   ├── observed.npy
│   │   └── wavelets.npy
│   └── N/
│       ├── observed.npy
│       └── wavelets.npy
└── gradients/
    ├── iter_0/
    │   ├── worker_0.npy      ← uploaded directly by worker 0 (GradientSink)
    │   ├── worker_1.npy      ← uploaded directly by worker 1 (GradientSink)
    │   └── ...
    ├── iter_1/
    │   ├── worker_0.npy
    │   ├── worker_1.npy
    │   └── ...
    └── ...
```

Keys are controlled by `ArtifactConfig.shot_prefix` (default `'shots'`) and
`ArtifactConfig.gradient_prefix` (default `'gradients'`).

---

## Configuration Reference

### `ArtifactConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `endpoint` | *(required)* | Artifact store host:port |
| `access_key` | *(required)* | Access key |
| `secret_key` | *(required)* | Secret key |
| `bucket` | *(required)* | Bucket name |
| `secure` | `False` | Use HTTPS |
| `backend` | `'minio'` | Backend identifier (stubbed for future `'s3'`) |
| `shot_prefix` | `'shots'` | Object key prefix for shot data |
| `gradient_prefix` | `'gradients'` | Object key prefix for gradients |

### Environment Variables (read by `ArtifactConfig.from_env()`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_ENDPOINT` | *(required)* | MinIO / S3 host:port |
| `ARTIFACT_ACCESS_KEY` | `minioadmin` | Access key |
| `ARTIFACT_SECRET_KEY` | `minioadmin` | Secret key |
| `ARTIFACT_BUCKET` | `stride-data` | Bucket name |
| `ARTIFACT_SECURE` | `false` | Use HTTPS (`'true'`/`'false'`) |
| `ARTIFACT_BACKEND` | `minio` | Backend |

---

## Usage

### Python API

```python
from stride import forward, adjoint
from stride.utils import ArtifactConfig

artifact_config = ArtifactConfig.from_env()  # reads ARTIFACT_* env vars

# Phase 1: forward pass — uploads observed traces to artifact store
await forward(problem_fwd, pde_fwd, vp_true, artifact_config=artifact_config)

# Phase 2: create shots from geometry, load data from artifact store, then invert
problem.transducers.default()
problem.geometry.default('elliptical', N)
problem.acquisitions.default()          # allocates shots with correct shapes
problem.acquisitions.load_artifacts(artifact_config)  # wavelets eager, observed lazy
await adjoint(problem, pde, loss, optimisation_loop, optimiser, vp,
              num_iters=2, select_shots=dict(num=N),
              artifact_config=artifact_config)
```

### Kubernetes (Minikube)

```bash
# 1. Start Minikube
minikube start --cpus=4 --memory=8192

# 2. First-time setup (installs Argo, deploys MinIO, creates RBAC, builds image)
./k8s/run.sh setup

# 3. Submit the artifact-backed inversion
RUN_MODE=inverse_artifacts ./k8s/run.sh start

# 4. Watch logs
./k8s/run.sh logs

# 5. Inspect MinIO bucket via web console
kubectl port-forward -n argo svc/minio 9001:9001
# Open http://localhost:9001  (user: minioadmin / minioadmin)

# 6. Cleanup
./k8s/run.sh cleanup
```

Expected log output:
```
Phase 1: Running forward pass...
Phase 1 complete. Observed traces uploaded to artifact store.
Phase 2: Starting inversion...
Loaded 2 shots via artifact store.
Beginning optimisation loop.
Block complete.
Block complete.

Inversion complete.
Final model range: [1453.6, 1583.4] m/s
=== inverse_artifacts Complete ===
```

### Deploy MinIO only (without full setup)

```bash
./k8s/run.sh minio
```

---

## Backwards Compatibility

All changes are strictly additive. The `artifact_config=None` default on both
`forward()` and `adjoint()` means existing scripts and tests are completely
unaffected.

| Existing usage | Impact |
|----------------|--------|
| `forward(problem, pde, vp)` | Zero — no `artifact_config` means old path |
| `adjoint(problem, pde, ...)` | Zero — no `artifact_config` means old path |
| `DiskTraces` | Untouched — `ArtifactTraces` is a separate class added after it |
| `lazy_loading=True` in `adjoint()` | Unchanged — activates `DiskTraces` path |
| `problem.acquisitions.load(path=...)` | Unchanged — existing method |
| Existing tests | Unchanged — no `artifact_config` passed |

---

## Out of Scope / Future Work

- **Gradient reducer service**: workers now upload `worker_K.npy` independently, but
  there is no service yet that reduces them into a final `gradient.npy`. The next step
  is a standalone Argo daemon that polls MinIO for all `worker_K.npy` files per
  iteration, tree-reduces them, and uploads `gradients/iter_N/gradient.npy`. Once
  that exists, `optimiser.step()` can be decoupled from the internal `pull(attr='grad')`
  path and instead download the pre-reduced gradient from MinIO.
- **Model checkpoint upload**: save `vp` at each iteration for resume-from-checkpoint.
- **Resume-from-checkpoint**: restart inversion from a saved iteration after pod crash.
- **AWS S3 / GCS backends**: `backend` field is stubbed; `get_client()` returns
  a `minio.Minio` client (S3-compatible) regardless. A proper S3/GCS path would use
  `boto3` / `google.cloud.storage`.
