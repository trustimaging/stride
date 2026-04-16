# Fault Tolerance in Stride / Mosaic (`dev-cloud` branch)

This document describes how the mosaic runtime handles nodes and workers joining, dropping out, and recovering. It covers the current state of the code as of the `dev-cloud` branch, including known limitations.

---

## Runtime Topology

```
Monitor (singleton, stable IP via K8s Service)
  ├── Head
  ├── Warehouse (global)
  ├── Node:0:<instance_id>
  │     ├── Warehouse:0
  │     ├── Worker:0:0:<instance_id>
  │     └── Worker:0:1:<instance_id>  (if num_workers > 1)
  └── Node:1:<instance_id>
        ├── Warehouse:1
        └── Worker:1:0:<instance_id>
```

All runtimes communicate over ZMQ ROUTER/ROUTER sockets. The monitor is the hub: it heartbeats nodes, maintains the worker scheduling pool, and brokers tessera placement.

---

## Unique Instance IDs

### Workers

Every worker UID includes a per-boot UUID suffix so that a pod restart always produces a new, globally unique identity:

```
worker:{node_index}:{slot_index}:{instance_id}
```

Example: `worker:1:0:eca6e9d5`

**Where it is generated** — `mosaic/runtime/node.py`, `Node.init()`:

```python
# mosaic/runtime/node.py:61
self._instance_id = uuid.uuid4().hex[:8]
```

**Where it is applied** — `mosaic/runtime/node.py`, `Node.init_workers()`:

```python
# mosaic/runtime/node.py:186
worker_uid = 'worker:%d:%d:%s' % (self.indices[0], worker_index, self._instance_id)
kwargs['runtime_uid'] = worker_uid
mosaic.init('worker', *args, **kwargs, wait=True)
```

**Why this matters** — Without unique IDs, when a replacement worker pod connects with the same UID as the dead worker, the monitor can confuse them. Cleanup of the dead worker races with registration of the new one. Unique IDs eliminate this entire class of race conditions: dead and new workers are always different keys.

**UID parsing** — `mosaic/runtime/runtime.py`, `BaseRPC.__init__()`:
Parses `name:num:num:hex` UIDs by stopping at the first non-numeric part. The full UID is stored as `_uid_override` and returned by the `uid` property. `BaseRPC.__init__` and `Runtime.__init__` both accept a `runtime_uid` kwarg which is stored directly as `_uid_override`.

### Nodes

Nodes use the same per-boot instance ID scheme as workers:

```
node:{node_index}:{instance_id}
```

Example: `node:1:7f3a2c81`

**Where it is set** — `mosaic/runtime/node.py`, `Node.init()`:

```python
self._instance_id = uuid.uuid4().hex[:8]
self._uid_override = 'node:%d:%s' % (self.indices[0], self._instance_id)
# Must be before await super().init() so the unique UID is used for ZMQ identity
await super().init(**kwargs)
```

This means when a node pod is replaced, it connects as a completely new UID. The dead node's heartbeat expires naturally (it was keyed to the old unique UID), and the replacement joins cleanly as a fresh entry. The `startswith('node:')` checks and `indices[0]` integer lookups in the cascade disconnect still work correctly with this format.

---

## Worker Join Sequence

```
Node pod starts
  │
  ├─ Node.init() → Runtime.init()
  │     └─ comms.handshake('monitor', addr, port)
  │           Sends: hand → monitor
  │           Receives: shake ← monitor (with full network list)
  │           Node connects to all network members
  │           listen() starts
  │
  ├─ Node.init_workers()
  │     For each worker slot:
  │       ├─ Generate worker_uid = 'worker:{n}:{s}:{uuid8}'
  │       ├─ Start subprocess: mosaic.init('worker', runtime_uid=worker_uid, wait=True)
  │       └─ comms.wait_for(worker_uid)  ← blocks until worker handshake completes
  │
  └─ Node.update_monitored_node()
        Logs: "all N workers up on {uid} — sending update_monitored_node"
        Sends resource stats → monitor.update_node()
        Monitor starts heartbeat for this node (dynamic mode)
        Logs: "NODE-CONNECTED: {uid} joined — bringing N workers: [...]"
        Adds workers to RoundRobin pool
        Logs: "STRATEGY-POOL: added workers [...] (pool now N: [...])"
```

**Key file/line references:**

| Step | File | Location |
|------|------|----------|
| Node init | `mosaic/runtime/node.py` | `Node.init()` line 44 |
| Worker UID generation | `mosaic/runtime/node.py` | `init_workers()` line 186 |
| Worker subprocess start | `mosaic/runtime/node.py` | `init_workers()` line ~194 |
| Wait for worker shake | `mosaic/runtime/node.py` | `init_workers()` line ~208 |
| All-workers-ready log | `mosaic/runtime/node.py` | after `resource_monitor()` call |
| Monitor receives node stats | `mosaic/runtime/monitor.py` | `update_node()` |
| Monitor starts heartbeat | `mosaic/runtime/monitor.py` | `update_node()` — dynamic mode |
| Strategy pool update | `mosaic/runtime/strategies.py` | `RoundRobin.update_node()` |

When a worker subprocess starts, it also calls `Runtime.init()` → `comms.handshake('monitor', ...)`. It connects to all runtimes in the network and its `hand` message is processed by each runtime's `Runtime.hand()`:

```python
# mosaic/runtime/runtime.py
def hand(self, sender_id, address, port):
    # Logs: "HAND: {sender} connecting from {addr}:{port} (pool_before=[...])"
    self.proxy_from_uid(sender_id)   # registers worker in local _workers dict
    # Logs: "HAND-DONE: {sender} registered (pool_after=[...])"
    # Fires _on_worker_count_changed callbacks
```

The monitor's round-robin strategy adds the worker to its scheduling pool when it receives `update_node` from the node:

```python
# mosaic/runtime/strategies.py
def update_node(self, updated):
    before = set(self._worker_list)
    for worker_id in updated.sub_resources['workers'].keys():
        self._worker_list.add(worker_id)
    self._num_workers = len(self._worker_list)
    # Logs: "STRATEGY-POOL: added workers [...] from node {uid} (pool now N: [...])"
```

---

## Handshake Protocol

The handshake is bidirectional and mesh-forming. Every new runtime must shake hands with every existing runtime.

```
New runtime                         Monitor
     │                                 │
     │──── hand(my_addr, my_port) ────▶│
     │                                 │  force-disconnect stale connection if exists
     │                                 │  connect_send(new_uid, addr, port)
     │                                 │  sleep(0.1)  ← ZMQ connection settle
     │                                 │  build network list (connected only)
     │◀─── shake(full_network) ────────│
     │                                 │
     │  For each member in network:    │
     │    comms.handshake(member, ...) │
     │    (bidirectional shake)        │
     │                                 │
     │──── listen() starts ───────────▶│
```

**Persistent recv task** — `mosaic/comms/comms.py`, `handshake()`:

The handshake recv loop never cancels the pending `recv_async()` task. Cancelling a pyzmq asyncio recv consumes the socket's FD-readability event, causing subsequent receives to block indefinitely. Instead, the task is kept alive across timeouts:

```python
recv_task = asyncio.ensure_future(self.recv_async())
try:
    while True:
        done, _ = await asyncio.wait([recv_task], timeout=60)
        if not done:
            # Logs: "COMMS-HS: 60s timeout waiting for shake from {uid}, retrying hand"
            await self.send_async(uid, method='hand', ...)
            continue

        sender_id, response = recv_task.result()
        if uid == sender_id and response.method == 'shake':
            # Logs: "COMMS-HS: handshake complete with {uid}"
            break
        elif response.method not in ('shake', 'hand', 'reply'):
            # Buffer RPC calls that arrived during the handshake wait
            pending_msgs.append((sender_id, response))

        recv_task = asyncio.ensure_future(self.recv_async())
finally:
    if not recv_task.done():
        recv_task.cancel()
```

Buffered messages are dispatched after the handshake completes, once all outbound connections are established.

**Force-disconnect on re-hand** — `mosaic/comms/comms.py`, `hand()`:

When a runtime receives a `hand` from a peer, it always force-disconnects any existing live outbound connection for that peer before creating a new one. This ensures the shake reply goes out on a fresh TCP socket rather than a dead one:

```python
async def hand(self, sender_id, address, port):
    # Logs: "COMMS-HAND: recv hand from {sender} at {addr}:{port} (known=True/False)"
    existing = self._send_conn.get(sender_id)
    if existing is not None and existing.state == 'connected':
        existing.disconnect()
    self.connect_send(sender_id, address, port)
    await asyncio.sleep(0.1)
    # Network list includes only 'connected' state entries
    network = {uid: (conn.address, conn.port)
               for uid, conn in self._send_conn.items()
               if conn.state == 'connected'}
    # Logs: "COMMS-HAND: sending shake to {sender} (network=[...])"
    await self.send_async(sender_id, method='shake', network=network)
    # Logs: "COMMS-HAND: shake sent to {sender}"
```

**Address-change reconnect** — `mosaic/comms/comms.py`, `connect_send()`:

```python
existing = self._send_conn.get(uid)
address_changed = (existing is not None
                   and (existing.address != address or existing.port != port))
if uid != self._runtime.uid and (existing is None
                                 or existing.state == 'disconnected'
                                 or address_changed):
    if address_changed:
        # Logs: "CONNECT-SEND: address changed for {uid} ..."
    if existing is not None and existing.state == 'connected':
        existing.disconnect()
    self._send_conn[uid] = OutboundConnection(uid, address, port, ...)
    self._send_conn[uid].connect()
    self._send_conn[uid].shake()
```

With unique per-boot UIDs for nodes and workers, the `address_changed` path should not fire in normal operation. It is retained as a safety net, and remains relevant for warehouses (which still share a UID across restarts — see Known Issues).

---

## Heartbeat Mechanism

The monitor heartbeats every connected node. This is the primary mechanism for detecting node death.

**Flow:**

```
Monitor                              Node
   │                                  │
   │──── heart() ───────────────────▶ │
   │                                  │  comms.heart() → sends beat back
   │◀─── beat() ─────────────────────│
   │                                  │
   │  On beat received:               │
   │    attempts reset to max+1       │
   │    reschedule heart in 3s        │
```

**Countdown** — `mosaic/comms/comms.py`, `OutboundConnection.heart()`:

```python
async def heart(self):
    # Orphan guard: stop if this connection was replaced
    if self._state != 'connected' or self._comms._send_conn.get(self.uid) is not self:
        # Logs: "HEARTBEAT-ORPHAN: stopping stale timer for {uid}"
        self.stop_heartbeat()
        return

    self._heartbeat_attempts -= 1

    if self._heartbeat_attempts == 0:
        # Logs: "HEARTBEAT-EXPIRE: {monitor} → disconnecting {node}"
        await self._comms.disconnect(self.uid, self.uid, notify=True)
        await self._loop.run(self._runtime.disconnect, self.uid, self.uid)
        return

    interval = self._heartbeat_interval * self._heartbeat_max_attempts / self._heartbeat_attempts
    self._heartbeat_timeout = self._loop.timeout(self.heart, timeout=interval)
    # Logs: "HEARTBEAT: {monitor} → heart to {node} (attempts_left=N, next_check_in=Xs)"
    try:
        await self.send_async(method='heart')
    except Exception as exc:
        # Logs: "HEARTBEAT-SEND-FAIL: {monitor} → {node} failed: {exc} — disconnecting"
        self._heartbeat_timeout.cancel()
        await self._comms.disconnect(self.uid, self.uid, notify=True)
        await self._loop.run(self._runtime.disconnect, self.uid, self.uid)
```

Default settings (configured in `OutboundConnection.__init__`):
- `_heartbeat_max_attempts = 2`
- `_heartbeat_interval = 3` seconds
- Effective timeout before disconnect: ~6 seconds total (increasing intervals: 3s, then 6s)

**Heartbeat start** — `mosaic/runtime/monitor.py`, `update_node()`:

```python
# In dynamic mode, heartbeat starts when monitor first receives update_node from a node
if self.mode == 'dynamic':
    self._comms.start_heartbeat(sender_id)
    # Logs: "NODE-CONNECTED: {uid} joined — bringing N workers: [...] — total nodes now: N"
```

**Beat response** — `mosaic/comms/comms.py`, `OutboundConnection.beat()`:

```python
async def beat(self):
    self._heartbeat_attempts = self._heartbeat_max_attempts + 1
    self.stop_heartbeat()
    self.start_heartbeat()
```

**Node beat handler** — heartbeat responses from nodes are sent via `comms.heart()` (which is now unrestricted — the previous `'node' not in self.uid` guard has been removed):

```python
async def heart(self, sender_id):
    try:
        await self.send_async(sender_id, method='beat')
    except Exception as exc:
        # Logs: "HEART-BEAT-FAIL: {runtime} failed to send beat to {sender}: {exc}"
```

---

## Disconnect / Cleanup

**`OutboundConnection.disconnect()`** — `mosaic/comms/comms.py`:

```python
def disconnect(self):
    # Always cancel heartbeat, even if already disconnected
    # (handles stale timers that outlive the state transition).
    self.stop_heartbeat()

    if self._state != 'connected':
        return

    # Cancel pending reply futures so send_recv_async callers get
    # RuntimeDisconnectedError immediately rather than hanging.
    pending, self._pending_reply_futures = self._pending_reply_futures, []
    for future in pending:
        if not future.done():
            future.set_exception(RuntimeDisconnectedError(...))

    self._socket.disconnect(self.connect_address)
    super().disconnect()
```

---

## Node Dropout Sequence

When a node pod dies (OOM, eviction, network partition):

```
Node goes silent
   │
   │  Monitor heartbeat fires, no beat received
   │  attempts_left: 2 → 1 → 0  (≈6 seconds total)
   │
   ▼
HEARTBEAT-EXPIRE: monitor → disconnecting node:X:<id>
   │
   ├─ comms.disconnect(node:X:<id>, notify=True)
   │     ├─ Logs: "COMMS-DISCONNECT: node:X:<id>"
   │     ├─ OutboundConnection.disconnect()   ← heartbeat stopped, pending replies cancelled
   │     └─ send disconnect(node:X:<id>) to all connected runtimes
   │
   └─ runtime.disconnect(monitor, node:X:<id>)  ← Monitor.disconnect()
         ├─ Logs: "MONITOR-DISCONNECT: node:X:<id>"
         ├─ _disconnected_runtimes.add(node:X:<id>)
         ├─ _monitor_strategy.remove_worker(node:X:<id>)
         │     Logs: "MONITOR-DISCONNECT: removed {uid} from strategy pool (pool_size=N)"
         ├─ del _monitored_nodes[node:X:<id>]
         └─ Cascade: disconnect each worker:X:*:<id> of node:X:<id>
               Logs: "MONITOR-DISCONNECT: cascading to workers of node:X:<id>: [...]"
               Each worker: _disconnected_runtimes.add(worker), remove from pool
```

**`Monitor.disconnect()`** — `mosaic/runtime/monitor.py`:

```python
def disconnect(self, sender_id, uid):
    # Logs: "MONITOR-DISCONNECT: {uid} (sender={sender})"
    super().disconnect(sender_id, uid)            # tessera/task proxy cleanup
    self._disconnected_runtimes.add(uid)
    self._monitor_strategy.remove_worker(uid)
    # Logs: "MONITOR-DISCONNECT: removed {uid} from strategy pool (pool_size=N)"

    if uid in self._monitored_nodes:
        worker_ids = list(self._monitored_nodes[uid].sub_resources.get('workers', {}).keys())
        # Logs: "MONITOR-DISCONNECT: cascading to workers of {uid}: [...]"
        for worker_id in worker_ids:
            self.disconnect(sender_id, worker_id)

    del self._monitored_nodes[uid]
    del self._runtime_tessera[uid]
    del self._runtime_tasks[uid]
```

**`RoundRobin.remove_worker()`** — `mosaic/runtime/strategies.py`:

```python
def remove_worker(self, uid):
    if uid in self._worker_list:
        self._worker_list.discard(uid)
        self._num_workers = len(self._worker_list)
        if self._num_workers > 0:
            self._last_worker = self._last_worker % self._num_workers
        else:
            self._last_worker = -1
```

**`Runtime.disconnect()`** (on all runtimes) — `mosaic/runtime/runtime.py`:

```python
def disconnect(self, sender_id, uid):
    # Clean up all tessera proxies that reference the dead runtime
    for obj in self._tessera.values():
        obj.deregister_proxy(uid)
    for obj in self._tessera_proxy.values():
        obj.deregister_runtime(uid)
    for obj in self._tessera_proxy_array.values():
        obj.deregister_runtime(uid)
    for obj in self._task_proxy.values():
        obj.deregister_runtime(uid)
    self.remove_proxy_from_uid(uid)
    # Logs: "POOL-REMOVE: {uid} removed from {runtime} pool — remaining: [...], callbacks: N"
    # Fires _on_worker_count_changed callbacks
```

**`ArrayProxy.deregister_runtime()`** — `mosaic/core/tessera.py`:

```python
def deregister_runtime(self, uid):
    removed = [p for p in self._proxies if p.runtime_id == uid]
    for p in removed:
        task = getattr(p, '_pending_init_task', None)
        if task is not None and not task.done():
            # Logs: "TESSERA-DEREGISTER: cancelling pending init task for {uid} on {worker}"
            task.cancel()
    self._proxies = [p for p in self._proxies if p.runtime_id != uid]
    # Logs: "TESSERA-ARRAY-DEREGISTER: removed N proxy/proxies for {uid} from {tessera}"
```

---

## Worker Dropout Sequence

Workers are monitored indirectly via their parent node. If a worker process dies (but the node pod stays alive), the node detects it via `resource_monitor()`:

```python
# mosaic/runtime/node.py — resource_monitor(), runs every 1 second
for worker_id, worker in self._own_workers.items():
    if worker.subprocess.running() and not worker.subprocess._mp_process.is_alive():
        # Logs: "NODE: worker subprocess {uid} died unexpectedly — disconnecting"
        worker.subprocess._state = 'stopped'
        asyncio.ensure_future(self._comms.disconnect(worker_id, worker_id, notify=True))
```

The subprocess state is polled. If it enters a terminal state the node will stop sending its stats for that worker to the monitor in `update_monitored_node`. The monitor's heartbeat for the node remains active.

---

## Tessera on New Workers (Slow Path)

When a replacement worker joins with a new UID, existing `ArrayProxy` tessera objects (e.g. `pde`, `loss`) do not automatically have a replica on it. The slow path handles this transparently:

```python
# mosaic/core/tessera.py, ArrayProxy.__call__
if task_proxies is None:
    # Logs: "TESSERA-SLOW-PATH: initialising {cls} on new worker {uid}"
    proxy = TesseraProxy(cls, *init_args, runtime=new_worker, **init_kwargs)
    self._proxies.append(proxy)
    # Strong reference prevents GC of the suspended init task
    proxy._pending_init_task = asyncio.ensure_future(
        proxy.__init_async__(*init_args, **init_kwargs))
    proxy._pending_init_task.add_done_callback(_suppress_disconnected)
    task_proxies = proxy[item](*args, **kwargs)
```

`CMDBase.__init_async__` wraps the init in `asyncio.shield` so a monitor-triggered iteration cancellation does not abort a worker's setup mid-flight.

If the new worker dies before its init completes, `deregister_runtime()` cancels the pending init task rather than leaking it.

---

## `_on_worker_count_changed` Callbacks

`Runtime._on_worker_count_changed` is a list of zero-argument callables that fire whenever a worker joins or leaves the local `_workers` pool:

- **Join**: fires in `proxy_from_uid()` when a new `worker:` UID is added
- **Leave**: fires in `remove_proxy_from_uid()` when a worker UID is removed

This is used by `_wait_for_workers()` in `stride/__init__.py` to wake up efficiently when replacement workers arrive rather than polling.

---

## `_disconnected_runtimes` Guard

The monitor maintains a set of UIDs that have been disconnected. Any messages arriving from these UIDs are silently dropped:

```python
# mosaic/runtime/monitor.py
def update_node(self, sender_id, ...):
    if sender_id in self._disconnected_runtimes:
        return    # drop stale message
```

This guard is checked in: `update_node`, `add_tessera_event`, `add_task_event`, `add_tessera_profile`, `add_task_profile`, and all `_add_*` helpers.

Note: warehouse UIDs (`warehouse:1`) are NOT added to `_disconnected_runtimes` when their parent node drops, because the cascade in `Monitor.disconnect()` only covers the workers listed in `_monitored_nodes[uid].sub_resources['workers']`. Warehouses are not tracked there. This is related to the known issue below.

---

## Orphan Heartbeat Guard

When `connect_send` replaces an `OutboundConnection` (e.g. for a replacement node with a new pod IP), the old connection's heartbeat timer may still be scheduled. The orphan guard prevents it from firing on a stale connection:

```python
# mosaic/comms/comms.py, OutboundConnection.heart()
async def heart(self):
    if self._state != 'connected' or self._comms._send_conn.get(self.uid) is not self:
        # Logs: "HEARTBEAT-ORPHAN: stopping stale timer for {uid}"
        self.stop_heartbeat()
        return
```

Additionally, `OutboundConnection.disconnect()` always cancels the heartbeat timer first, even if the connection is already in a non-connected state.

---

## `RuntimeDisconnectedError` at the Tessera Layer

When a worker drops while tasks are in-flight, the tessera/task proxy layer raises `RuntimeDisconnectedError`. `ArrayProxy.__init_async__` handles this gracefully:

```python
# mosaic/core/tessera.py
for task in asyncio.as_completed(inits, timeout=timeout):
    try:
        await task
    except RuntimeDisconnectedError as exc:
        if safe:
            available_workers -= 1
            if available_workers <= 0:
                raise RuntimeError('No workers available to complete async workload')
        else:
            raise
```

The `safe=True` default means that as long as at least one worker remains alive, the array operation completes on remaining workers.

---

## Fault Tolerance in `stride` (Iteration Restart)

Stride's `adjoint()` function supports an optional `drop_threshold` parameter. When set, a worker-drop monitor runs concurrently with the shot dispatch loop and triggers an iteration restart if too many workers are lost.

### `drop_threshold`

`drop_threshold` is a float in `[0, 1)` representing the fraction of original workers that must be lost before restarting. `0` means restart on any single drop.

### Worker Monitor

`_start_worker_monitor()` — `stride/__init__.py`:

```python
def _start_worker_monitor(runtime, drop_threshold, loop_task):
    initial_uids = set(w.uid for w in runtime.workers)
    # Logs: "_start_worker_monitor: initial_uids=[...] threshold=N callbacks_before=N"
    event = asyncio.Event()
    runtime._on_worker_count_changed.append(event.set)
    monitor_task = asyncio.ensure_future(
        _watch_workers(runtime, initial_uids, drop_threshold, loop_task, event))
    ...
```

`_watch_workers()` wakes on every `_on_worker_count_changed` event, computes how many of the *original* UIDs are gone (replacement workers joining with new UIDs do not mask drops), and cancels `loop_task` if the threshold is exceeded:

```python
async def _watch_workers(runtime, initial_uids, threshold, task, event):
    n = len(initial_uids)
    while not task.done():
        await event.wait()
        event.clear()
        current_uids = set(w.uid for w in runtime.workers)
        lost = initial_uids - current_uids
        fraction = len(lost) / n if n > 0 else 0.0
        # Logs: "_watch_workers check — lost=[...] fraction=N threshold=N"
        if n > 0 and fraction > threshold:
            # Logs: "_watch_workers: drop threshold exceeded — cancelling"
            task.cancel()
            return
```

### Iteration Retry Loop

When `loop_task` is cancelled, `adjoint()` catches it and retries:

```python
except asyncio.CancelledError:
    runtime._inside_async_for = False
    iteration.clear()
    optimiser.clear_grad()
    # Logs: "ADJOINT-RETRY: iteration N — drop threshold exceeded, clearing and retrying.
    #        Workers now: N/N [...]"
    await _wait_for_workers(runtime, initial_worker_count)
    # Logs: "ADJOINT-RETRY: iteration N — proceeding with workers: [...]"
```

### Wait for Replacement Workers

`_wait_for_workers()` — `stride/__init__.py`:

Waits up to `timeout` seconds (default 300s) for `runtime.num_workers` to reach `target`. Wakes efficiently via `_on_worker_count_changed` events and logs a heartbeat every 30 seconds so the operator can see the wait is active:

```
WAIT-FOR-WORKERS: start — need N workers, have M (present: [...])
WAIT-FOR-WORKERS: still waiting at +30s — M/N workers present (present: [...])
WAIT-FOR-WORKERS: pool changed at +45s — N/N workers
                  (present: [...], joined: [...], left: [...])
WAIT-FOR-WORKERS: done after 45s — proceeding with N workers (target was N, present: [...])
```

If the timeout expires without reaching the target, it proceeds with the surviving workers and logs a warning.

---

## Log Reference

Key log prefixes and what they indicate:

| Prefix | Runtime | Meaning |
|--------|---------|---------|
| `HEARTBEAT-EXPIRE` | monitor | Node heartbeat countdown reached 0 — disconnecting |
| `HEARTBEAT-ORPHAN` | monitor | Stale heartbeat timer detected and stopped |
| `HEARTBEAT-SEND-FAIL` | monitor | Could not send `heart` — disconnecting immediately |
| `COMMS-HAND` | any | Received or sending a handshake `hand`/`shake` |
| `COMMS-HS` | any | Handshake progress (timeout retry, completion) |
| `COMMS-DISCONNECT` | any | Comms layer disconnecting a UID |
| `CONNECT-SEND` | any | Address changed for a UID — reconnecting |
| `MONITOR-DISCONNECT` | monitor | Runtime-layer disconnect + cascade |
| `NODE-CONNECTED` | monitor | New node's first `update_node` received in dynamic mode |
| `STRATEGY-POOL` | monitor | Workers added to / removed from the round-robin pool |
| `POOL-JOIN` | head/node | Worker UID added to local `_workers` dict |
| `POOL-REMOVE` | head/node | Worker UID removed from local `_workers` dict |
| `HAND` / `HAND-DONE` | any | `Runtime.hand()` before/after `proxy_from_uid` |
| `ASYNC-FOR-QUEUE` | head | Shots being dispatched with this worker set |
| `ASYNC-FOR-BARRIER` | head | Waiting for all in-flight shots to complete |
| `TESSERA-SLOW-PATH` | head | New worker getting a tessera replica on demand |
| `TESSERA-DEREGISTER` | head | Pending init task cancelled for dead worker |
| `WAIT-FOR-WORKERS` | head | Waiting for replacement workers to phone home |
| `ADJOINT-RETRY` | head | Iteration restarted after worker drop |
| `INIT-WORKERS` | node | Worker subprocess lifecycle on the node |

---

## Known Issues / Open Work

### 1. Warehouse UIDs are not unique per boot

**Symptom:** `warehouse:1` keeps the same UID across node pod restarts. When a replacement node pod starts, its warehouse subprocess sends a `hand` to the monitor but may get stuck in a 60s retry loop (`COMMS-HS: 60s timeout waiting for shake from monitor`).

**Root cause:** The monitor's outbound connection to the old `warehouse:1` is not closed during the node disconnect cascade (warehouses are not tracked in `_monitored_nodes[node].sub_resources['workers']`). The `comms.hand()` handler force-disconnects any live existing connection when it receives a `hand`, and `connect_send` handles address changes — but if the monitor's ZMQ socket does not flush the shake to the new pod IP within the 0.1s settle window, the message can be dropped.

**Planned fix:** Apply the same instance ID scheme to warehouses: `warehouse:{idx}:{instance_id}`, generated and overridden in `Warehouse.init()` the same way nodes now do it. This eliminates the UID collision entirely.

### 2. `address_changed` logic in `connect_send` is now vestigial for nodes/workers

**Context:** With unique per-boot UIDs for both nodes and workers, the `address_changed` path in `connect_send` should never fire in normal operation for those runtimes. It is retained as a safety net and remains actively used for warehouses (whose UIDs are not yet unique). Can be removed for nodes/workers once warehouse UIDs are also made unique.

---

## Summary of Fault-Tolerance Coverage

| Scenario | Handled? | Mechanism |
|----------|----------|-----------|
| Worker process dies, node stays up | Yes | `resource_monitor` detects → `comms.disconnect` |
| Node pod killed (heartbeat timeout) | Yes | Heartbeat expire (~6s) → `Monitor.disconnect()` cascade |
| Node pod killed (instant, before heartbeat) | Partial | Heartbeat countdown; up to ~6s detection lag |
| Worker replacement (unique UID) | Yes | New UID means no collision; clean registration |
| Node replacement (unique UID per boot) | Yes | New UID = clean registration; dead node heartbeat expires naturally |
| Stale in-flight messages from dead runtime | Yes | `_disconnected_runtimes` guard on monitor |
| In-flight tessera ops when worker dies | Yes | `RuntimeDisconnectedError` + `safe=True` in `ArrayProxy` |
| New worker gets tessera replica | Yes | Slow path in `ArrayProxy.__call__` with `asyncio.shield` init |
| Pending reply futures on disconnect | Yes | `OutboundConnection.disconnect()` sets `RuntimeDisconnectedError` |
| Iteration restart on worker drop | Yes | `_watch_workers` + `CancelledError` retry in `adjoint()` |
| Wait for replacement workers | Yes | `_wait_for_workers` with `_on_worker_count_changed` events |
| Stale heartbeat timer after reconnect | Yes | Orphan guard in `heart()` + `stop_heartbeat()` on disconnect |
| Shake lost if ZMQ not ready (0.1s settle) | Partial | Retry hand every 60s until shake arrives |
| Warehouse UID collision on pod restart | No | See Known Issues — warehouse UIDs not yet unique |
| Monitor reconnect from head/nodes | N/A | Monitor is singleton with stable K8s ClusterIP |
