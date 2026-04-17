# Fault Tolerance in Stride / Mosaic (`dev-cloud` branch)

This document describes how the mosaic runtime handles nodes and workers joining, dropping out, and recovering. It covers the current state of the code as of the `dev-cloud` branch.

---

## Runtime Topology

```
Monitor (singleton, stable IP via K8s Service)
  |-- Head
  |-- Warehouse (global)
  |-- Node:0:<instance_id>
  |     |-- Warehouse:0:<instance_id>
  |     |-- Worker:0:0:<instance_id>
  |     +-- Worker:0:1:<instance_id>  (if num_workers > 1)
  +-- Node:1:<instance_id>
        |-- Warehouse:1:<instance_id>
        +-- Worker:1:0:<instance_id>
```

All runtimes communicate over ZMQ ROUTER/ROUTER sockets. The monitor is the hub: it heartbeats nodes, maintains the worker scheduling pool, and brokers tessera placement.

---

## Unique Instance IDs

Every pod boot generates a random 8-character hex string (`instance_id`). This ID is shared by the node, its warehouse, and all its workers, so a pod restart produces an entirely new set of UIDs that never collide with the dead ones.

### Where the instance ID is generated

```python
# mosaic/runtime/node.py — Node.init()
self._instance_id = uuid.uuid4().hex[:8]
```

### UID formats

| Runtime   | Format                                          | Example                    |
|-----------|-------------------------------------------------|----------------------------|
| Node      | `node:{node_index}:{instance_id}`               | `node:1:7f3a2c81`          |
| Warehouse | `warehouse:{node_index}:{instance_id}`          | `warehouse:1:7f3a2c81`     |
| Worker    | `worker:{node_index}:{slot}:{instance_id}`      | `worker:1:0:7f3a2c81`      |

### Where UIDs are set

**Node** (`mosaic/runtime/node.py`, `Node.init()`):

```python
self._uid_override = 'node:%d:%s' % (self.indices[0], self._instance_id)
# Set before await super().init() so ZMQ identity uses the unique UID
```

**Warehouse** (`mosaic/runtime/node.py`, `Node.init()`):

```python
self._warehouse_uid = 'warehouse:%d:%s' % (self.indices[0], self._instance_id)
await self.init_warehouse(indices=self.indices[0], warehouse_uid=self._warehouse_uid, **kwargs)
```

**Worker** (`mosaic/runtime/node.py`, `Node.init_workers()`):

```python
worker_uid = 'worker:%d:%d:%s' % (self.indices[0], worker_index, self._instance_id)
kwargs['runtime_uid'] = worker_uid
kwargs['local_warehouse_uid'] = self._warehouse_uid
mosaic.init('worker', *args, **kwargs, wait=True)
```

The `local_warehouse_uid` kwarg is propagated through `mosaic.init()` (which explicitly accepts it in its signature) into `runtime_config`, so each worker knows the unique UID of its local warehouse for routing warehouse fetches.

### Why unique UIDs matter

Without unique IDs, when a replacement pod connects with the same UID as the dead one, the monitor confuses them: cleanup of the dead runtime races with registration of the new one. Unique IDs eliminate this entirely -- dead and new runtimes are always different keys in every dict.

### UID parsing

`mosaic/runtime/runtime.py`, `BaseRPC.__init__()` parses `name:num:num:hex` UIDs by stopping at the first non-numeric part. The full UID is stored as `_uid_override` and returned by the `uid` property.

### WarehouseObject node_id derivation

`mosaic/types/warehouse_object.py` derives `node_id` from the worker UID so warehouse fetches route correctly:

```python
if runtime.uid.startswith('worker:'):
    parts = runtime.uid.split(':')
    if len(parts) >= 4:
        node_id = 'node:%s:%s' % (parts[1], parts[3])
    else:
        node_id = 'node:%s' % parts[1]
```

---

## Worker Join Sequence

```
Node pod starts
  |
  |-- Node.init() -> Runtime.init()
  |     +-- comms.handshake('monitor', addr, port)
  |           Sends: hand -> monitor
  |           Receives: shake <- monitor (with full network list)
  |           Node connects to all network members
  |
  |-- Node.init_warehouse()
  |     Starts warehouse subprocess with unique UID
  |     warehouse:{n}:{instance_id}
  |
  |-- Node.init_workers()
  |     For each worker slot:
  |       |-- Generate worker_uid = 'worker:{n}:{s}:{instance_id}'
  |       |-- Start subprocess: mosaic.init('worker', runtime_uid=worker_uid,
  |       |                                  local_warehouse_uid=warehouse_uid, ...)
  |       +-- comms.wait_for(worker_uid)  <- blocks until worker handshake completes
  |
  +-- Node.update_monitored_node()
        Sends resource stats -> monitor.update_node()
        Monitor starts heartbeat for this node (dynamic mode)
        Adds workers to RoundRobin pool
```

---

## Handshake Protocol

The handshake is bidirectional and mesh-forming. Every new runtime shakes hands with every existing runtime.

```
New runtime                         Monitor
     |                                 |
     |---- hand(my_addr, my_port) ---->|
     |                                 |  force-disconnect stale connection if exists
     |                                 |  connect_send(new_uid, addr, port)
     |                                 |  sleep(0.1)  <- ZMQ connection settle
     |                                 |  build network list (connected only)
     |<--- shake(full_network) --------|
     |                                 |
     |  For each member in network:    |
     |    comms.handshake(member, ...) |
     |    (bidirectional shake)        |
     |                                 |
     |---- listen() starts ----------->|
```

**Force-disconnect on re-hand** -- When a runtime receives a `hand` from a peer, it force-disconnects any existing live outbound connection before creating a new one. This ensures the shake reply goes on a fresh TCP socket.

**Persistent recv task** -- The handshake recv loop never cancels the pending `recv_async()` task. Cancelling a pyzmq asyncio recv consumes the socket's FD-readability event, causing subsequent receives to block indefinitely. Instead, the task is kept alive across timeouts (retrying `hand` every 60s).

---

## Heartbeat Mechanism

The monitor heartbeats every connected node. This is the primary mechanism for detecting node death.

```
Monitor                              Node
   |                                  |
   |---- heart() ------------------->|
   |                                  |  comms.heart() -> sends beat back
   |<--- beat() ---------------------|
   |                                  |
   |  On beat received:               |
   |    attempts reset to max+1       |
   |    reschedule heart in 3s        |
```

Default settings:
- `_heartbeat_max_attempts = 2`
- `_heartbeat_interval = 3` seconds
- Effective timeout before disconnect: ~6 seconds total (increasing intervals)

**Orphan guard** -- When `connect_send` replaces an `OutboundConnection`, the old connection's heartbeat timer may still be scheduled. The orphan guard checks `self._comms._send_conn.get(self.uid) is not self` and stops the stale timer.

---

## Disconnect / Cleanup

### Node Dropout Sequence

When a node pod dies (OOM, eviction, network partition):

```
Node goes silent
   |
   |  Monitor heartbeat fires, no beat received
   |  attempts_left: 2 -> 1 -> 0  (~6 seconds total)
   |
   v
HEARTBEAT-EXPIRE: monitor -> disconnecting node:X:<id>
   |
   |-- comms.disconnect(node:X:<id>, notify=True)
   |     |-- OutboundConnection.disconnect()
   |     |     |-- Cancel heartbeat timer
   |     |     +-- Cancel pending reply futures (-> RuntimeDisconnectedError)
   |     +-- Broadcast disconnect to all connected runtimes
   |
   +-- Monitor.disconnect(node:X:<id>)
         |-- _disconnected_runtimes.add(node:X:<id>)
         |-- Remove from strategy pool
         |-- del _monitored_nodes[node:X:<id>]
         +-- Cascade: disconnect each worker:X:*:<id>
               Each worker: _disconnected_runtimes.add(worker), remove from pool
```

### Runtime.disconnect() (on all runtimes)

`mosaic/runtime/runtime.py`:

```python
def disconnect(self, sender_id, uid):
    self._disconnected_runtimes.add(uid)

    # Clean up tessera/task proxies referencing the dead runtime
    for obj in self._tessera.values():
        obj.deregister_proxy(uid)
    for obj in self._tessera_proxy.values():
        obj.deregister_runtime(uid)
    for obj in self._tessera_proxy_array.values():
        obj.deregister_runtime(uid)
    for obj in self._task_proxy.values():
        obj.deregister_runtime(uid)

    self.remove_proxy_from_uid(uid)
    # Fires _on_worker_count_changed callbacks
```

`_disconnected_runtimes` is maintained on every runtime (not just the monitor). It is checked in two places:

1. **Monitor**: drops stale messages in `update_node`, `add_tessera_event`, `add_task_event`, etc.
2. **Task proxy init**: `TaskProxy.init()` and `TaskArrayProxy.init()` check it before sending tasks to a worker (see below).

### ArrayProxy.deregister_runtime()

`mosaic/core/tessera.py`:

```python
def deregister_runtime(self, uid):
    removed = [p for p in self._proxies if p.runtime_id == uid]
    for p in removed:
        task = getattr(p, '_pending_init_task', None)
        if task is not None and not task.done():
            task.cancel()
    self._proxies = [p for p in self._proxies if p.runtime_id != uid]
```

---

## Stale Worker Eviction in Strategy Pool

When a replacement node joins, its workers have new UIDs. The round-robin strategy evicts any stale workers from the same node index before adding the new ones:

```python
# mosaic/runtime/strategies.py — RoundRobin.update_node()
node_idx = int(updated.uid.split(':')[1])
prefix = 'worker:%d:' % node_idx
stale = {w for w in self._worker_list
         if w.startswith(prefix) and w not in incoming}
if stale:
    self._worker_list -= stale
```

This prevents the pool from accumulating dead worker entries when the old heartbeat hasn't expired yet at the time the replacement joins.

---

## Disconnect Race Condition Guard

There is a race between disconnect propagation and task proxy initialisation. If a worker disconnects *after* `async_for` dispatches a shot to it but *before* the task proxy registers itself with the runtime, `deregister_runtime` will never find the task proxy and its `_done_future` will never resolve -- causing the head to hang.

### Fix: early disconnect check in TaskProxy.init()

`mosaic/core/task.py`:

```python
# TaskProxy.init()
runtime_id = self.runtime_id
disconnected = getattr(self.runtime, '_disconnected_runtimes', set())
if runtime_id in disconnected:
    raise RuntimeDisconnectedError(
        'Remote runtime %s already disconnected before task init' % runtime_id)
```

The same check exists in `TaskArrayProxy.init()` before the dependency loop.

### Fix: _done_future resolution in __init_async__

`mosaic/core/base.py`:

```python
async def __init_async__(self, *args, **kwargs):
    try:
        await asyncio.shield(self.init(*args, **kwargs))
    except asyncio.CancelledError:
        # Shield allows init to continue despite iteration retry
        await self.init(*args, **kwargs)
    except RuntimeDisconnectedError as e:
        # Set _done_future so any code awaiting this proxy unblocks
        if (hasattr(self, '_done_future')
                and hasattr(self._done_future, 'done')
                and not self._done_future.done()):
            self._done_future.set_exception(e)
            self._done_future.exception()  # mark retrieved
        raise
```

The `hasattr(self._done_future, 'done')` guard is necessary because tessera proxies use `AwaitableOnly` for `_done_future` (which lacks a `.done()` method), while task proxies use `Future`.

### Fix: _exclusive_proxy uses try/finally

`mosaic/runtime/runtime.py`:

```python
@contextlib.asynccontextmanager
async def _exclusive_proxy(self, queue, safe=False):
    proxy = await queue.get()
    try:
        yield proxy
    finally:
        await queue.put(proxy)
```

Without `try/finally`, a `RuntimeDisconnectedError` from the yielded block would lose the proxy from the queue permanently, starving subsequent shots of that worker slot.

---

## Tessera Lazy Init on New Workers (Slow Path)

When a replacement worker joins with a new UID, existing `ArrayProxy` tessera objects (e.g. `pde`, `loss`) do not have a replica on it. The slow path in `ArrayProxy.__call__` creates one on demand:

```python
# mosaic/core/tessera.py — ArrayProxy.__call__
if task_proxies is None:
    proxy = TesseraProxy(cls, *init_args, runtime=new_worker, **init_kwargs)
    self._proxies.append(proxy)
    proxy._pending_init_task = asyncio.ensure_future(
        proxy.__init_async__(*init_args, **init_kwargs))
    proxy._pending_init_task.add_done_callback(_suppress_disconnected)
    task_proxies = proxy[item](*args, **kwargs)
```

Key details:

- A strong reference (`_pending_init_task`) prevents GC of the suspended init task.
- `_suppress_disconnected` retrieves the exception so asyncio doesn't log "Task exception was never retrieved" if the new worker dies before init completes.
- `__init_async__` wraps init in `asyncio.shield` so a monitor-triggered iteration cancellation does not abort the worker's setup mid-flight.
- If the new worker dies before init completes, `deregister_runtime()` cancels the pending init task.

---

## Iteration Retry on Worker Drop

Stride's `adjoint()` function supports a `drop_threshold` parameter that controls whether to restart an iteration when workers are lost.

### drop_threshold modes

| Value | Behaviour |
|-------|-----------|
| `0.0` | Restart on any single worker drop |
| `0.5` | Restart when >50% of original workers are lost |
| `None` | Never restart -- continue with surviving workers |

### Worker monitor

`_start_worker_monitor()` in `stride/__init__.py` captures the current set of worker UIDs as the baseline. It registers a callback on `_on_worker_count_changed` and starts `_watch_workers`, which wakes on every pool change:

```python
async def _watch_workers(runtime, initial_uids, threshold, task, event):
    n = len(initial_uids)
    while not task.done():
        await event.wait()
        event.clear()
        current_uids = set(w.uid for w in runtime.workers)
        lost = initial_uids - current_uids
        fraction = len(lost) / n if n > 0 else 0.0
        if n > 0 and fraction > threshold:
            task.cancel()
            return
```

Because it tracks the *original* UIDs, replacement workers joining with new UIDs do not mask drops.

The monitor is only started when `drop_threshold is not None` and an artifact warehouse is configured.

### Retry loop

When `loop_task` is cancelled, `adjoint()` catches `CancelledError` and retries:

```python
except asyncio.CancelledError:
    runtime._inside_async_for = False
    iteration.clear()
    optimiser.clear_grad()
    await _wait_for_workers(runtime, initial_worker_count)
    # Retry: loop back to while True
```

### Wait for replacement workers

`_wait_for_workers()` blocks until `runtime.num_workers` reaches the target count (or a 300s timeout expires). It wakes efficiently via `_on_worker_count_changed` events and logs a heartbeat every 30s so the operator can see the wait is active.

If the timeout expires without reaching the target, it proceeds with the surviving workers and logs a warning.

### drop_threshold=None (no restart)

When `drop_threshold=None`, no worker monitor is started. The `async_for` loop dispatches shots to whatever workers are in the pool. If a worker dies mid-iteration:

1. The shot on that worker raises `RuntimeDisconnectedError`
2. `async_for` catches it (with `safe=True`) and decrements `available_workers`
3. Remaining shots complete on surviving workers
4. The barrier waits for all in-flight tasks to finish
5. Gradient accumulation proceeds with fewer contributions

This mode is useful for disconnect testing and workloads that tolerate partial results.

---

## _on_worker_count_changed Callbacks

`Runtime._on_worker_count_changed` is a list of zero-argument callables that fire whenever a worker joins or leaves the local `_workers` pool:

- **Join**: fires in `proxy_from_uid()` when a new `worker:` UID is added
- **Leave**: fires in `remove_proxy_from_uid()` when a worker UID is removed

Used by `_wait_for_workers()` and `_watch_workers()` to wake efficiently rather than polling.

---

## async_for and Shot Dispatch

The head's `async_for` dispatches shots to workers via a queue:

```python
worker_queue = asyncio.Queue()
for worker in self._workers.values():
    await worker_queue.put(worker)

async def call(*iters):
    async with self._exclusive_proxy(worker_queue, safe=safe) as _worker:
        res = await func(_worker, *iters)
    return res

tasks = [asyncio.create_task(call(*each)) for each in zip(*iterables)]
```

When a shot fails with `RuntimeDisconnectedError` and `safe=True`, `async_for` retires that worker and continues. If all workers are gone, it raises `RuntimeError`.

After all shots complete (or fail), `async_for` calls `barrier()` to wait for all in-flight tasks across the mesh to drain.

---

## Log Reference

Key log prefixes at INFO level (debug-level handshake/warehouse/tessera logs omitted):

| Prefix | Runtime | Meaning |
|--------|---------|---------|
| `HEARTBEAT-EXPIRE` | monitor | Node heartbeat countdown reached 0 -- disconnecting |
| `HEARTBEAT-ORPHAN` | monitor | Stale heartbeat timer detected and stopped |
| `HEARTBEAT-SEND-FAIL` | monitor | Could not send `heart` -- disconnecting immediately |
| `COMMS-DISCONNECT` | any | Comms layer disconnecting a UID |
| `MONITOR-DISCONNECT` | monitor | Runtime-layer disconnect + cascade |
| `NODE-CONNECTED` | monitor | New node's first `update_node` received in dynamic mode |
| `STRATEGY-POOL` | monitor | Workers added to / evicted from the round-robin pool |
| `POOL-REMOVE` | any | Worker UID removed from local `_workers` dict |
| `ASYNC-FOR-BARRIER` | head | Waiting for / done with in-flight shot barrier |
| `WAIT-FOR-WORKERS` | head | Waiting for replacement workers to phone home |
| `ADJOINT-RETRY` | head | Iteration restarted after worker drop |
| `INIT-WORKERS` | node | Summary: all workers up and registered with monitor |
| `INIT-SHIELDED` | any | `asyncio.shield` caught a cancellation during init |
| `INIT-DISCONNECT` | any | `__init_async__` failed due to `RuntimeDisconnectedError` |

---

## Summary of Fault-Tolerance Coverage

| Scenario | Handled? | Mechanism |
|----------|----------|-----------|
| Worker process dies, node stays up | Yes | `resource_monitor` detects -> `comms.disconnect` |
| Node pod killed (heartbeat timeout) | Yes | Heartbeat expire (~6s) -> `Monitor.disconnect()` cascade |
| Worker replacement (unique UID) | Yes | New UID = no collision; clean registration |
| Node replacement (unique UID per boot) | Yes | New UID = clean registration; dead node expires naturally |
| Warehouse replacement (unique UID per boot) | Yes | `warehouse:{idx}:{instance_id}` -- no collision |
| Stale in-flight messages from dead runtime | Yes | `_disconnected_runtimes` guard on monitor |
| In-flight tessera ops when worker dies | Yes | `RuntimeDisconnectedError` + `safe=True` in `ArrayProxy` |
| New worker gets tessera replica | Yes | Lazy init (slow path) in `ArrayProxy.__call__` |
| Task proxy init races with disconnect | Yes | Early `_disconnected_runtimes` check + `_done_future` resolution |
| `_exclusive_proxy` leak on exception | Yes | `try/finally` returns proxy to queue |
| Pending reply futures on disconnect | Yes | `OutboundConnection.disconnect()` sets `RuntimeDisconnectedError` |
| Iteration restart on worker drop | Yes | `_watch_workers` + `CancelledError` retry in `adjoint()` |
| Continue without restart (drop_threshold=None) | Yes | `async_for` retires dead worker, proceeds with survivors |
| Wait for replacement workers | Yes | `_wait_for_workers` with `_on_worker_count_changed` events |
| Stale heartbeat timer after reconnect | Yes | Orphan guard in `heart()` + `stop_heartbeat()` on disconnect |
| Stale workers in strategy pool | Yes | `RoundRobin.update_node()` evicts by node index prefix |
