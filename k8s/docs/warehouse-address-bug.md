# Warehouse Address Bug: Inverse Hangs on K8s

Detailed investigation of the bug that caused Stride inverse runs to hang
indefinitely on Kubernetes, while forward runs completed successfully.

---

## Symptom

When running `simple_inverse.py` via Argo Workflows on Minikube, the head
pod would print:

```
Beginning optimisation loop.
```

...and then hang forever. No error, no timeout, no crash. Forward runs
(`simple_forward.py`) completed without issues on the same cluster.

---

## Why Forward Works but Inverse Doesn't

The difference comes down to how model parameters are created and
distributed to workers.

### Forward path

```python
# simple_forward.py
vp = ScalarField(name='vp', grid=grid)
```

`ScalarField()` creates a local object. When the head publishes it to
workers, `runtime.put()` takes the **warehouse path**: it wraps the object
in a `WarehouseObject` and calls `warehouse.put_remote()`. This is a
simple store-and-retrieve — the monitor's warehouse stores the object, and
worker warehouses fetch it on demand via RPC.

### Inverse path

```python
# simple_inverse.py
vp = ScalarField.parameter(name='vp', grid=grid)
```

`.parameter()` creates a **remote tessera** — a proxy object backed by a
`TesseraProxy` that lives in the warehouse. When the head publishes it,
`runtime.put()` takes the **proxy path**: it calls `obj.push(publish=True)`,
which calls `warehouse.push_remote(__dict__=..., uid=..., publish=True)`.

With `publish=True`, the monitor's warehouse (`uid=warehouse`) iterates
over all nodes and pushes the object state to each node warehouse
(`warehouse:0`, `warehouse:1`, etc.). The node warehouses receive the
`push_remote` call, discover they don't have the object locally, and call
`get_remote` **back to the monitor's warehouse** to fetch it.

This is where the bug hits: the callback from node warehouse to monitor
warehouse uses the address the monitor's warehouse advertised — and that
address was wrong.

---

## Root Cause

### Address propagation chain

1. The head pod's `mrun` command starts the monitor with `--address 0.0.0.0`.
   This is correct for binding (accept connections on all interfaces).

2. The monitor spawns a **warehouse subprocess** via `mosaic/utils/subprocess.py`.
   The subprocess receives `parent_address='0.0.0.0'` and
   `monitor_address='0.0.0.0'` from the parent runtime.

3. The warehouse subprocess creates an `InboundConnection` (ZMQ ROUTER socket)
   to receive incoming messages. The `InboundConnection.address` property
   determines the advertised address — the one sent to other runtimes in the
   **network dict** during the handshake.

4. **The bug**: the original `InboundConnection.address` auto-detection logic:
   ```python
   @property
   def address(self):
       if self._address is None:
           # Try hostname, then UDP probe, then 127.0.0.1
           self._address = get_hostname()
           try:
               validate_address(self._address)
           except ValueError:
               # UDP probe fallback...
       return self._address
   ```

   The warehouse subprocess has `self._address = '0.0.0.0'` (inherited from
   monitor). Since `self._address is not None`, the auto-detection is
   **skipped entirely**, and the warehouse advertises `0.0.0.0` as its
   routable address.

### What happens at runtime

5. During the handshake, the monitor sends its network dict to connecting
   nodes. This dict includes entries like:
   ```
   warehouse → 0.0.0.0:3002
   ```

6. Node warehouses on other pods receive this dict and create outbound
   connections to `0.0.0.0:3002`. On the node's pod, `0.0.0.0` resolves
   to **the node's own localhost** — not the head pod. The connection
   silently goes nowhere.

7. When the monitor warehouse calls `push_remote(publish=True)` and the
   node warehouse tries to call `get_remote` back, it sends a ZMQ message
   to its own localhost:3002. Since nothing is listening there (the
   warehouse is on the head pod), the message is silently dropped.
   `asyncio.gather` in `push_remote` waits forever for the reply.

### Why there's no error

ZMQ DEALER sockets (used for outbound `send_conn`) silently queue
messages when the peer is unreachable. There's no TCP-level timeout
configured, and mosaic's `send_exception` handler also fails (it tries to
send the error back over the same broken connection). The result is a
completely silent hang.

---

## The Fix

### Change 1: Treat `0.0.0.0` as "not set" (`mosaic/comms/comms.py`)

Modified both `InboundConnection.address` and `Publication.address`
properties to treat `0.0.0.0` the same as `None` — triggering
auto-detection:

```python
@property
def address(self):
    if self._address is None or self._address == '0.0.0.0':
        # 0.0.0.0 is valid for binding but not routable — auto-detect
        self._address = None

        # Try to find a routable IP via UDP probe first (works in K8s
        # where hostname may not be resolvable from other pods)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 53))
                self._address = s.getsockname()[0]
            finally:
                s.close()
        except OSError:
            pass

        # Fall back to hostname
        if self._address is None:
            self._address = get_hostname()
            try:
                validate_address(self._address)
            except ValueError:
                self._address = '127.0.0.1'

    return self._address
```

### Why UDP probe first (not hostname)

The original code tried `get_hostname()` first, then fell back to the UDP
probe. In K8s, `get_hostname()` returns `socket.getfqdn(socket.gethostname())`
which gives the pod hostname (e.g., `stride-abc123-head-987654`). This
hostname passes `validate_address` because `socket.gethostbyname()` can
resolve it **locally** (on the same pod). But it's **not resolvable from
other pods** — Argo/K8s doesn't set up cross-pod DNS for individual pod
hostnames.

The UDP probe (`socket.connect(('8.8.8.8', 53)); s.getsockname()[0]`)
returns the pod's actual network IP (e.g., `10.244.0.5`) without sending
any packets — it just queries the routing table. This IP is routable
across all pods in the cluster.

### Files changed

| File | Change |
|------|--------|
| `mosaic/comms/comms.py` | `InboundConnection.address` and `Publication.address`: treat `0.0.0.0` as unset, prefer UDP probe over hostname for auto-detection |

### Other changes made during debugging (now reverted)

Temporary debug prints were added and subsequently removed from:
- `mosaic/runtime/warehouse.py` (init, get_remote, push_remote, publish)
- `mosaic/runtime/runtime.py` (put)
- `mosaic/core/tessera.py` (push)
- `stride/__init__.py` (adjoint)

### Dockerfile change

`k8s/Dockerfile.stride` was restructured to copy `environment.yml` first
and create the conda environment in a separate layer, so that code-only
changes don't trigger a full environment rebuild (~30s vs ~10min).

---

## How to Verify

```bash
eval $(minikube docker-env)
docker build -t stride-k8s:latest -f k8s/Dockerfile.stride .
argo delete --all -n argo
argo submit k8s/stride-workflow.yaml -n argo -p run-mode="inverse" --watch
```

Expected output (tail):
```
Updating variable vp,
  grad before processing in range [-4.2e-01, 3.6e-01]
  grad after processing in range [-3.3e-06, 2.9e-06]
  variable range before update [1.46e+03, 1.56e+03]
  taking final update step of 5.0e+00
  variable range after update [1.45e+03, 1.58e+03]
Done iteration 2 (out of 2), block 2 (out of 2) - Total loss 2.05e-02
====================================================================
Inversion complete.
Final model range: [1453.6, 1583.4] m/s
=== inverse Complete ===
Process ended with code: 0
```

---

## Debugging Timeline

1. **Observed**: Inverse hangs after "Beginning optimisation loop."
2. **Added debug prints** to `stride/__init__.py` adjoint: hang occurs
   during `published_args = await asyncio.gather(...)` — specifically
   when publishing `vp` (a `.parameter()` object).
3. **Traced code path**: `runtime.put()` → proxy path → `tessera.push(publish=True)`
   → `warehouse.push_remote(publish=True)`.
4. **Added debug prints** to `warehouse.push_remote`: hang occurs at
   `await asyncio.gather(*tasks)` — the monitor warehouse is waiting for
   replies from node warehouses.
5. **Checked node warehouse logs**: node warehouses receive `push_remote`
   but don't have the object locally. They call `get_remote` back to the
   monitor warehouse — and hang.
6. **Hypothesis**: connectivity issue between node warehouses and monitor
   warehouse.
7. **Added connection info** to warehouse init logs: node warehouses DO
   have a connection entry for `warehouse`, but the address is
   `0.0.0.0:3002`.
8. **Root cause confirmed**: `0.0.0.0` is the bind address, not a routable
   address. Node warehouses on other pods connect to their own localhost.
9. **First fix attempt**: treat `0.0.0.0` as unset, trigger auto-detection.
   Auto-detection returned pod hostname — not resolvable from other pods.
10. **Final fix**: restructured auto-detection to try UDP probe first
    (returns routable pod IP), hostname as fallback. Inverse completes
    successfully.
