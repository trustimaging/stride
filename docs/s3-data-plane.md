# S3/MinIO Data Plane Integration

Optional S3/MinIO persistence layer for model parameters, shot data, and
gradients. Mosaic continues to handle all control flow and computation; S3 adds
a durable staging layer for cloud and Kubernetes deployments.

---

## Data Flow

### Original (mosaic-only)

```
                    mosaic RPC / warehouse
                    ~~~~~~~~~~~~~~~~~~~~~~
Head                                         Workers
 |                                              |
 |-- publish(vp) -----> warehouse -----------> worker0  (forward / adjoint)
 |                                             worker1
 |                                             ...
 |<--- grad accumulation via redux <-----------'
 |
 +-- optimiser.step()
 |       variable.pull(attr='grad')      <-- warehouse
 |       update variable in-place
 |
 +-- optimiser.dump()  -->  local HDF5
```

All data transfers (model broadcast, gradient reduction, shot data) happen
through mosaic's internal warehouse and RPC layer. Intermediate state lives
in-memory or in local HDF5 files written by `optimiser.dump()`.

### New (mosaic + S3 hybrid)

```
                    mosaic RPC / warehouse                 S3 / MinIO
                    ~~~~~~~~~~~~~~~~~~~~~~                 ~~~~~~~~~~
Head                                         Workers
 |                                              |
 |-- publish(vp) -----> warehouse -----------> worker0
 |                                             worker1
 |-- upload_model(vp) -----------------------> s3://bucket/models/iter_N/
 |                                              |
 |<--- grad accumulation via redux <-----------'
 |
 +-- optimiser.step()
 |       variable.pull(attr='grad')      <-- warehouse
 |       upload_array(raw_gradient)      --> s3://bucket/gradients/iter_N/
 |       update variable in-place
 |
 +-- optimiser.dump()  -->  local HDF5
 |
 +-- (optional) stage_shots_to_s3()      --> s3://bucket/shots/
```

The computation graph, shot dispatch, and gradient reduction are unchanged.
S3 writes happen **alongside** the existing flow at three points:

1. **Model snapshot** -- written to S3 at each iteration start, before workers
   receive the published args.
2. **Raw gradient** -- written to S3 inside `optimiser.pre_process()`, after
   the gradient is pulled from the warehouse but before processing.
3. **Shot staging** -- bulk upload of wavelet + observed data, called explicitly
   in user scripts when needed.

When `s3_config` is `None` (the default), no S3 code runs and behaviour is
identical to the original.

---

## S3 Bucket Layout

```
s3://bucket/
  models/
    iter_0/
      arg_0.npy            # vp array (np.save format)
    iter_1/
      arg_0.npy
    ...
    iter_forward/          # snapshot from forward() calls
      arg_0.npy
  gradients/
    iter_0/
      raw_gradient.npy     # raw accumulated gradient before processing
    iter_1/
      raw_gradient.npy
    ...
  shots/
    shot_00000.h5          # HDF5 with datasets: wavelets, observed
    shot_00001.h5
    ...
```

Prefixes (`models`, `gradients`, `shots`) are configurable via `S3Config`.

---

## Files Changed

### New Files

| File | Purpose |
|---|---|
| `stride/utils/s3.py` | S3 utility module: `S3Config` dataclass, upload/download helpers for arrays, models, shots, and gradients |
| `scripts/simple_forward.py` | Lightweight 2D forward script (100x100 grid, 8 transducers, layered model) |
| `scripts/simple_inverse.py` | Lightweight 2D inversion script (homogeneous initial model, gradient descent) |
| `scripts/simple_inverse_s3.py` | Same inversion with S3 persistence enabled via environment variables |

### Modified Files

| File | What changed |
|---|---|
| `stride/utils/__init__.py` | Added `from .s3 import *` |
| `stride/__init__.py` | Added `s3_config=None` kwarg to `forward()` and `adjoint()`; S3 model upload at publish time; S3 client init in adjoint; `s3_config` passed through to `optimiser.step()` |
| `stride/optimisation/optimisers/optimiser.py` | `LocalOptimiser.__init__` accepts `s3_config`; `pre_process()` uploads raw gradient to S3 after pulling from warehouse |

---

## Detailed Change Descriptions

### `stride/utils/s3.py` (new)

`S3Config` dataclass with fields:

```python
@dataclass
class S3Config:
    endpoint: str          # e.g. "localhost:9000"
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    model_prefix: str = "models"
    shot_prefix: str = "shots"
    gradient_prefix: str = "gradients"
```

`S3Config.from_env(prefix='MINIO')` reads `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY`, `MINIO_BUCKET`, and `MINIO_SECURE` from the environment.

Functions provided:

| Function | Description |
|---|---|
| `get_s3_client(config)` | Create a `minio.Minio` client (lazy import) |
| `ensure_bucket(client, bucket)` | Create bucket if it does not exist |
| `upload_array(client, bucket, key, array)` | Upload numpy array as `.npy` via `BytesIO` |
| `download_array(client, bucket, key)` | Download numpy array from `.npy` object |
| `upload_shot_data(client, config, shot_id, wavelets, observed)` | Upload shot as HDF5 |
| `download_shot_data(client, config, shot_id)` | Download shot HDF5, return `(wavelets, observed)` |
| `stage_shots_to_s3(client, config, problem)` | Bulk-upload all shots from a `Problem` |
| `upload_model(client, config, iteration_id, model_args)` | Upload model parameter arrays for an iteration |
| `download_model(client, config, iteration_id, num_args)` | Download model parameter arrays |
| `list_gradients(client, config, iteration_id)` | List gradient object keys for an iteration |
| `clear_iteration_gradients(client, config, iteration_id)` | Delete all gradient objects for an iteration |

The `minio` and `h5py` packages are imported lazily inside the functions that
need them, so the module loads without extra dependencies when S3 is not used.

### `stride/__init__.py`

**`forward()`** (line 89):
- Pops `s3_config` from kwargs (default `None`).
- After `published_args` are gathered, if `s3_config` is set: creates an S3
  client and calls `upload_model()` with the model args. Wrapped in
  try/except so S3 failures do not block the forward run.

**`adjoint()`** (line 236):
- Pops `s3_config` from kwargs (default `None`).
- Creates an S3 client once at function entry if config is provided.
- Inside the iteration loop, after publishing args: calls `upload_model()` to
  snapshot the current model to S3.
- Passes `s3_config=s3_config` to `optimiser.step()` so the optimiser can
  persist gradients.

### `stride/optimisation/optimisers/optimiser.py`

**`LocalOptimiser.__init__`** (line 49):
- Pops `s3_config` from kwargs and stores as `self.s3_config`.

**`pre_process()`** (line 86):
- Pops `s3_config` from kwargs (falls back to `self.s3_config`).
- After `self.variable.pull(attr='grad')`, if `s3_config` and `iteration` are
  both set: uploads the raw gradient array to
  `{gradient_prefix}/iter_{abs_id}/raw_gradient.npy`. Wrapped in try/except
  so S3 failures produce a warning but do not interrupt the optimisation.

---

## Usage

All stride scripts must be launched through mosaic's `mrun` command, which
spawns the monitor and worker processes. Running with bare `python` will hang
because no workers connect to the runtime.

```
mrun -nw <num_workers> python <script.py>
```

Run `mrun --help` to see all options.

### Forward + Inverse (no S3)

```bash
# Generate observed data (2 workers)
mrun -nw 2 python scripts/simple_forward.py

# Run inversion against the observed data
mrun -nw 2 python scripts/simple_inverse.py
```

### S3-Enabled Inverse

Start MinIO:

```bash
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

Run the S3 inversion (uses default MinIO credentials):

```bash
mrun -nw 2 python scripts/simple_inverse_s3.py
```

Override defaults with environment variables:

```bash
export MINIO_ENDPOINT=minio.cluster.local:9000
export MINIO_ACCESS_KEY=mykey
export MINIO_SECRET_KEY=mysecret
export MINIO_BUCKET=my-stride-bucket

mrun -nw 2 python scripts/simple_inverse_s3.py
```

### Using S3 in Your Own Scripts

```python
from stride import *
from stride.utils.s3 import S3Config

s3_config = S3Config(
    endpoint='localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    bucket='stride-data',
)

# Pass to forward
await forward(problem, pde, vp, s3_config=s3_config)

# Pass to adjoint + optimiser
optimiser = GradientDescent(vp, step_size=5, s3_config=s3_config)

await adjoint(problem, pde, loss,
              optimisation_loop, optimiser, vp,
              s3_config=s3_config)
```

---

## Dependencies

| Package | Required when | Install |
|---|---|---|
| `minio` | Using any S3/MinIO functionality | `pip install minio` |
| `h5py` | Using `upload_shot_data` / `download_shot_data` | `pip install h5py` |

Both are lazily imported. Existing code paths that do not pass `s3_config`
have zero new dependencies.

---

## Backward Compatibility

- `s3_config` defaults to `None` everywhere. When `None`, no S3 code executes.
- All S3 operations are wrapped in try/except with warning-level logging.
  A transient S3 failure will not crash the simulation.
- No existing function signatures changed in a breaking way. The new parameter
  is keyword-only and popped from `**kwargs`.
- Existing scripts (e.g. `breast2D/01_script_forward.py`) work without
  modification.
