# PR Plan: dev-cloud → master

Code changes to `mosaic/` and `stride/` only. Three PRs, merged in order.

---

## PR 1: Inverse Handshake (Dynamic Node Registration)

### Overview

In the existing mosaic architecture, the monitor spawns all nodes at startup via SSH or SLURM. The node count is fixed for the lifetime of the run. This PR inverts that relationship: the monitor starts in **dynamic mode** and waits for nodes to **phone home** — connecting to the monitor's address at any time during the run.

**How it works:**

1. The monitor starts with `mrun --dynamic` and writes its address/port to a key file. It enters `init_dynamic()` instead of `init_cluster()`, skipping node spawning entirely.

2. Each node starts independently with `mrun --node --phone-home`. It reads `MONITOR_HOST`, `MONITOR_PORT`, and `PUBSUB_PORT` from environment variables, then performs a ZMQ handshake with the monitor.

3. On connection, the node creates its warehouse subprocess and worker subprocesses. The workers handshake with the monitor individually. The monitor's `update_node()` detects the new node via `_monitored_nodes` and starts heartbeat monitoring.

4. Nodes can join at any time — the monitor's worker pool grows dynamically. This is the foundation for cloud deployments where pods are scheduled by Kubernetes independently of the mosaic runtime.

**UID scheme:**

Previously, UIDs were deterministic based on index (e.g., `node:0`, `worker:0:0`, `warehouse:0`). When a pod dies and a replacement starts with the same index, the UIDs collide — the monitor still holds stale ZMQ connections for the old UID.

This PR introduces a per-boot **instance ID** (8-char hex from `uuid4`), appended to every UID:

- Node: `node:{index}:{instance_id}` → e.g., `node:0:a3f7b2c1`
- Worker: `worker:{node_index}:{slot}:{instance_id}` → e.g., `worker:0:0:a3f7b2c1`
- Warehouse: `warehouse:{index}:{instance_id}` → e.g., `warehouse:0:a3f7b2c1`

A node and all its workers/warehouse share the same instance ID, so they form a traceable group. When a replacement pod starts, it gets a new instance ID — no collision with the dead pod's UIDs. The warehouse UID can be derived from the node UID by swapping the prefix: `node:0:abc` → `warehouse:0:abc`.

Workers are passed their UID explicitly via `runtime_uid` and connect to their node's warehouse via `local_warehouse_uid`, rather than deriving these from index alone.

**Comms layer changes:**

- **Address discovery**: `0.0.0.0` (used for binding in containers) is now treated as "auto-detect" rather than a routable address. A UDP probe to `8.8.8.8:53` finds the pod's actual IP, which works in K8s where hostnames may not resolve across pods.

- **Reconnection**: `connect_send()` now detects when a known UID reconnects from a different address (replacement pod) and tears down the old connection before establishing the new one.

- **Handshake buffering**: When a node is still completing its handshake with the monitor, RPCs from the head (e.g., `init_tessera`) may arrive before the ZMQ `shake` completes. These messages are now buffered during the handshake wait and replayed after completion, preventing lost RPCs.

### Changes

| File | Changes |
|------|---------|
| `mosaic/cli/mrun.py` | `--dynamic` flag (monitor waits for nodes), `--phone-home` flag (node reads monitor address from env vars) |
| `mosaic/__init__.py` | `local_warehouse_uid` parameter passthrough in `mosaic.init()` |
| `mosaic/runtime/monitor.py` | `init_dynamic()` mode, `register_node()` RPC, dynamic heartbeat setup in `update_node()`, `_get_total_workers()` helper, `num_workers`/`workers` properties for dynamic mode |
| `mosaic/runtime/node.py` | Phone-home mode (`MONITOR_HOST`/`MONITOR_PORT`/`PUBSUB_PORT` env vars). Per-boot `instance_id` (uuid) for unique UIDs. Workers passed `runtime_uid` and `local_warehouse_uid` |
| `mosaic/runtime/runtime.py` | `init_warehouse()` accepts `warehouse_uid` for explicit UID. `_uid_override` support. `local_warehouse_uid` kwarg for worker→warehouse routing |
| `mosaic/runtime/head.py` | Passes dynamic/phone-home config through to subprocess |
| `mosaic/comms/comms.py` | Address discovery: `0.0.0.0` treated as auto-detect, UDP probe for K8s. `connect_send()` handles reconnection on address change. Handshake buffering: RPCs arriving before shake completes are queued and replayed |
| `mosaic/runtime/warehouse.py` | `_warehouse_uid_for_node()` derives warehouse UID from node UID |
| `mosaic/types/warehouse_object.py` | `node_id`/`warehouse_id` fields for warehouse routing |

### Logging to review

| Log | Level | Keep? |
|-----|-------|-------|
| `NODE-INIT: handshake complete` | DEBUG | Yes — debug only |
| `NODE-INIT: init_warehouse() done` | DEBUG | Yes — debug only |
| `Successfully connected to monitor (phone-home mode)` | INFO | Yes — useful in production |
| `INIT-WORKERS: starting/subprocess started/wait_for done` | DEBUG | Yes — debug only |
| `INIT-WORKERS: N workers up on node:X` | INFO | Yes — confirms node ready |
| `COMMS-HS: sending hand/buffering/dispatching` | DEBUG/INFO | **Review** — several INFO-level handshake logs could be DEBUG |
| `CONNECT-SEND: address changed` | INFO | Yes — important for diagnosing reconnection |
| `COMMS-HAND: sending shake/shake sent` | DEBUG | Yes — debug only |

---

## PR 2: Artifact Warehouse (S3/MinIO Storage)

### Overview

In standard mosaic, gradients flow through the in-memory SpillBuffer warehouse: each worker computes a gradient, stores it in its local warehouse, the head pulls and accumulates them via `variable.pull(attr='grad')`. This requires all workers to be alive at pull time and ties gradient accumulation to the runtime's memory.

This PR introduces an **S3-backed artifact warehouse** that decouples data flow from the runtime:

**Shot data flow (forward pass):**

1. During `forward()`, each worker runs the PDE for its assigned shot, producing observed traces.
2. Instead of keeping traces in memory, the worker uploads them to S3 as `shots/{shot_id}/observed.npy` and `shots/{shot_id}/wavelets.npy`.
3. The in-memory traces are replaced with an `ArtifactTraces` object that stores only the S3 key. When data is needed later, `ArtifactTraces.load()` downloads it on demand.

**Gradient flow (adjoint pass):**

1. Each worker computes its gradient and uploads it to S3 as `gradients/iter_{N}/shot_{K}.pkl` (via `ArtifactWarehouse.exec_remote()`, which replaces the SpillBuffer `exec_remote()` path).
2. An external **gradient accumulator** daemon polls S3 for all expected shot files (listed in `shots.json`), sums them, and writes `gradients/iter_{N}/final.pkl`.
3. The head's `variable.pull(attr='grad')` polls S3 for `final.pkl` instead of pulling from the SpillBuffer.

**Key design decisions:**

- The artifact warehouse is **opt-in**: set globally via `mosaic.set_artifact_warehouse()` or auto-configured when the `ARTIFACT_ENDPOINT` env var is present. All existing code paths continue to work without it.
- `ArtifactTraces` is lazy — it holds a key, not data. This keeps memory usage constant regardless of how many shots are in the problem.
- The gradient accumulator is a standalone process (`python -m mosaic.runtime.gradient_accumulator`) that needs no connection to the mosaic runtime — it only talks to S3. It handles retries by watching the `attempt` field in `shots.json`: if the head rewrites `shots.json` with a new attempt number, the accumulator resets its accumulation.
- `stride/core.py` skips `__redux_adjoint__` (the SpillBuffer gradient path) when the artifact warehouse is configured, since gradients go directly to S3.

**Depends on:** PR 1 (needs phone-home and unique UIDs so that replacement workers upload gradients with non-colliding keys)

### Changes

| File | Changes |
|------|---------|
| `mosaic/runtime/artifact_warehouse.py` | **NEW.** `ArtifactWarehouse`: MinIO client, `push_remote()`/`pull_remote()`, `exec_remote()` for gradient upload, `set_iteration()`, `clear_iteration_gradients()`, `write_shot_list()`, `ensure_bucket()` |
| `mosaic/runtime/gradient_accumulator.py` | **NEW.** Daemon: polls S3 for `shot_N.pkl`, accumulates, writes `final.pkl`. Detects attempt changes via `shots.json` for retry support |
| `mosaic/runtime/__init__.py` | Exports `artifact_warehouse` module |
| `mosaic/__init__.py` | `set_artifact_warehouse()`/`get_artifact_warehouse()` globals, auto-init from `ARTIFACT_ENDPOINT` env var |
| `mosaic/cli/mrun.py` | `--artifact-storage` flag, `ARTIFACT_ENDPOINT` env var auto-detection |
| `mosaic/runtime/runtime.py` | `exec()` routes to `ArtifactWarehouse.exec_remote()` when configured |
| `mosaic/core/tessera.py` | `pull(attr='grad')` polls S3 for `final.pkl` when artifact warehouse set |
| `stride/__init__.py` | `forward()`: uploads observed/wavelets to S3 per shot, creates `ArtifactTraces`. Skip disk-load optimisation when warehouse configured |
| `stride/problem/data.py` | `ArtifactTraces` class: stores `_artifact_key`, `load()` downloads from S3 on demand |
| `stride/problem/acquisitions.py` | `load_artifacts()` loads from S3 (uses global warehouse). `_traces()` returns `ArtifactTraces` when set |
| `stride/core.py` | `__call_adjoint__` skips `__redux_adjoint__` when warehouse set |
| `stride/utils/artifacts.py` | **NEW.** `ArtifactConfig` dataclass |
| `stride/utils/__init__.py` | Exports `ArtifactConfig` |

### Logging to review

| Log | Level | Keep? |
|-----|-------|-------|
| `Uploaded observed traces and wavelets for shot N` | PERF | Yes — progress tracking |
| Gradient accumulator `Iter N — waiting/expecting/folded/done` | INFO | Yes — operator visibility into accumulation |
| `WARNING: ArtifactWarehouse.from_env() failed` | print/WARNING | **Review** — uses `print()` instead of logger |

---

## PR 3: Fault Tolerance for Inversions

### Overview

When a worker dies during a distributed inversion, the existing mosaic runtime has no recovery path — the `async_for` loop hangs waiting for a response that will never arrive, and the inversion stalls.

This PR adds a complete fault-tolerance layer to the adjoint inversion loop:

**Drop detection:**

- `_watch_workers()` is an event-driven coroutine that monitors the worker pool during each `async_for` iteration. It registers a callback on `_on_worker_count_changed` (a new callback list fired whenever `_workers` changes via `proxy_from_uid`/`remove_proxy_from_uid`). When the fraction of dropped workers (relative to the initial set) exceeds `drop_threshold`, it cancels the `async_for` loop task.

- The `drop_threshold` parameter controls tolerance: `0` means retry on any drop, `0.5` means accept up to 50% shot loss and continue with partial results.

**Retry loop:**

The `adjoint()` function wraps each iteration's shot processing in a `while True` retry loop:

1. `@runtime.async_for` dispatches shots to workers
2. `_watch_workers` monitors for drops in parallel
3. If the loop is cancelled (drop detected) or fails:
   - `clear_iteration_gradients()` deletes stale S3 data for this iteration
   - `write_shot_list()` writes a new `shots.json` with an incremented attempt number (so the accumulator resets)
   - If no workers remain, `_wait_for_workers(runtime, 1)` blocks until at least one replacement arrives
   - The loop retries from the top
4. If the loop completes and enough shots succeeded (based on `drop_threshold`), the iteration proceeds to the gradient step

**Worker readiness gating (`_wait_for_workers`):**

When all workers die and replacements join, there's a race: the worker appears in `_workers` (via ZMQ handshake) before its node has fully initialised (warehouse up, comms listening). Dispatching work immediately can hang.

`_wait_for_workers` has two phases:
- **Phase 1**: Event-driven wait for `runtime.num_workers >= target` using `_on_worker_count_changed` callbacks
- **Phase 2**: Polls `check_node_status()` on the monitor — an RPC that checks whether each worker's node has appeared in `_monitored_nodes` (which only happens after the node's full init: warehouse started, workers spawned, first `update_node` sent). Work is only dispatched after Phase 2 confirms readiness.

**RPC hang prevention:**

When a worker dies, there are two disconnect steps in the comms layer:
1. `runtime.disconnect()` — runs synchronously, cleans up bookkeeping
2. `comms.disconnect()` — scheduled as a background task, fails pending RPC Reply futures

The gap between (1) and (2) could leave in-flight RPCs hanging. This PR fixes it by having `runtime.disconnect()` directly call `OutboundConnection.disconnect()` to fail pending Reply futures **synchronously**, before any callbacks fire. This is idempotent — the later background `comms.disconnect()` finds the connection already closed.

Additionally, `send_recv_async()` now has a **20-second timeout** on RPC replies. If a worker dies between heartbeats and the synchronous disconnect hasn't caught it, the timeout acts as a safety net.

**Tessera slow-path for new workers:**

When a worker joins after tessera arrays were created, the arrays have no proxy for the new worker. `ArrayProxy._get_remote_method()` handles this via a "slow path": it creates a new `TesseraProxy` on the new worker, starts `__init_async__` in the background, and dispatches the method call. The init uses `asyncio.shield` to prevent iteration-retry cancellation from interrupting the worker's setup. `deregister_runtime()` cancels any pending init tasks when a worker disconnects, preventing dangling futures.

**Depends on:** PR 2 (retry loop uses `clear_iteration_gradients` and `write_shot_list` from the artifact warehouse)

### Changes

| File | Changes |
|------|---------|
| `stride/__init__.py` | `_wait_for_workers()`: event-driven wait with Phase 2 `check_node_status()` poll. `_watch_workers()`: cancels loop on drop threshold breach. `_start_worker_monitor()`: captures baseline UIDs. `adjoint()`: `while True` retry loop with `clear_iteration_gradients`, `write_shot_list`, attempt tracking. `drop_threshold` parameter |
| `mosaic/runtime/runtime.py` | `_on_worker_count_changed` callback list fired on worker join/leave. `_disconnected_runtimes` set. `disconnect()`: fails pending RPC Reply futures synchronously via `conn.disconnect()`. `_exclusive_proxy()`: skips dead workers. `async_for()`: waits for ≥1 worker when pool is empty |
| `mosaic/runtime/monitor.py` | `check_node_status()` RPC. `disconnect()`: cascades to workers, removes from `_monitored_nodes`, fires callbacks. Active node count by unique index (excludes stale entries) |
| `mosaic/comms/comms.py` | `_pending_reply_futures` list on `OutboundConnection`. `disconnect()` fails all pending futures with `RuntimeDisconnectedError`. Heartbeat: 3s interval, 2 attempts. `send_recv_async()`: 20s timeout on replies |
| `mosaic/core/base.py` | `__init_async__()`: `asyncio.shield` prevents cancellation from stopping worker setup. `RuntimeDisconnectedError` handler sets `_done_future` |
| `mosaic/core/tessera.py` | `ArrayProxy._get_remote_method()`: slow-path creates tessera on new workers on demand. `deregister_runtime()`: cancels pending init tasks |
| `mosaic/core/task.py` | `deregister_runtime()` and `deregister_proxy()` for dead worker cleanup |
| `mosaic/runtime/strategies.py` | `remove_worker()` removes dead workers from scheduler |
| `mosaic/utils/event_loop.py` | Minor event loop fix |

### Logging to review

| Log | Level | Keep? |
|-----|-------|-------|
| `WAIT-FOR-WORKERS: start/pool changed/done` | INFO | Yes — operator visibility |
| `WAIT-FOR-WORKERS: waiting for node(s)` | INFO | **Review** — could be DEBUG, fires every 1s poll |
| `WAIT-FOR-WORKERS: all N node(s) confirmed ready` | INFO | Yes |
| `_watch_workers started/check` | DEBUG | Yes — debug only |
| `_watch_workers: drop threshold exceeded — cancelling` | WARNING | Yes — important event |
| `FAULT-TOLERANCE: loop_task completed/cancelled/failed` | INFO/WARNING | Yes — key operational events |
| `ADJOINT-RETRY: iteration N attempt M` | INFO | Yes |
| `HEARTBEAT-EXPIRE: disconnecting node:X` | INFO | Yes — important event |
| `MONITOR-DISCONNECT: cascading/removed from strategy` | INFO | **Review** — verbose, could be DEBUG |
| `BARRIER: pending/still waiting/no previous` | INFO | **Review** — verbose, could be DEBUG |
| `TESSERA-ARRAY-DEREGISTER: removed N proxy/proxies` | INFO | **Review** — very verbose during drops, could be DEBUG |
| `TESSERA-SLOW-PATH: initialising X on new worker` | DEBUG | Yes — debug only |
| `ASYNC-FOR: 0 workers available, waiting` | INFO | Yes |
| `NODE: worker subprocess died unexpectedly` | INFO | Yes |
| `POOL-JOIN/POOL-REMOVE/POOL-SKIP` | DEBUG | Yes — debug only |
| `INIT-SHIELDED/INIT-DISCONNECT` | WARNING | Yes — important fault events |

### Changes to remove before merge

| File | What | Reason |
|------|------|--------|
| `stride/optimisation/optimisers/optimiser.py` | `DIAG iter=N raw grad.data sum=...` logging block (14 lines) | Debug diagnostics added during development, not needed in production. Computes `np.sum`/`np.min`/`np.max` on every gradient pull — unnecessary overhead |

---

## Shared files across PRs

| File | PR 1 | PR 2 | PR 3 |
|------|------|------|------|
| `mosaic/cli/mrun.py` | `--dynamic`, `--phone-home` | `--artifact-storage`, env var detect | — |
| `mosaic/__init__.py` | `local_warehouse_uid` | `set/get_artifact_warehouse()` | — |
| `mosaic/runtime/runtime.py` | `warehouse_uid`, `_uid_override` | `exec()` routing | `_on_worker_count_changed`, `disconnect()` RPC fix, `async_for` wait, `_exclusive_proxy` |
| `mosaic/comms/comms.py` | Address discovery, handshake buffering, reconnection | — | Pending reply futures, heartbeat tuning, RPC timeout |
| `mosaic/runtime/monitor.py` | `init_dynamic()`, `register_node()`, dynamic heartbeat | — | `check_node_status()`, `disconnect()` cascade, active node count |
| `stride/__init__.py` | — | Forward S3 upload, `ArtifactTraces` | `_wait_for_workers`, `_watch_workers`, retry loop |

### Merge order

1. **PR 1** (Inverse Handshake) — no dependencies
2. **PR 2** (Artifact Warehouse) — depends on PR 1
3. **PR 3** (Fault Tolerance) — depends on PR 2
