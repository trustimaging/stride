# Cloud-Native Kubernetes Deployment: Mosaic Code Changes

A proposal for running Stride full-waveform inversions on Kubernetes, describing the
modifications made to the Mosaic distributed runtime to support dynamic worker registration
and cloud-native networking.

---

## Table of Contents

1. [Background](#background)
2. [The Core Problem](#the-core-problem)
3. [Overview of Changes](#overview-of-changes)
4. [Detailed Code Changes](#detailed-code-changes)
   - [1. `mosaic/runtime/node.py` — Phone-Home Initiation](#1-mosaicrutimenodepy--phone-home-initiation)
   - [2. `mosaic/runtime/monitor.py` — Dynamic Registration](#2-mosaicruntimemonitorpy--dynamic-registration)
   - [3. `mosaic/runtime/head.py` — Deferred Worker Wait](#3-mosaicruntimeheadpy--deferred-worker-wait)
   - [4. `mosaic/cli/mrun.py` — CLI Flags](#4-mosaiccliimrunpy--cli-flags)
   - [5. `mosaic/__init__.py` — Public API](#5-mosaicinit_py--public-api)
   - [6. `mosaic/comms/comms.py` — Address Resolution](#6-mosiaccommscommspy--address-resolution)
5. [How the Changes Work Together on Kubernetes](#how-the-changes-work-together-on-kubernetes)
6. [Deployment Architecture](#deployment-architecture)
7. [Validation](#validation)
8. [Remaining Work](#remaining-work)

---

## Background

Mosaic is the distributed computing layer underlying Stride. It manages a mesh of
runtimes — a **Monitor** that coordinates the cluster, **Nodes** that host compute
workers, and a **Head** that runs the user's script. In the original design, this
mesh was assembled by the Monitor: it SSHed or submitted SLURM jobs to spawn nodes,
passing them its own address so they could handshake back.

This architecture works well on traditional HPC clusters. It does not work on
Kubernetes for the following reasons:

- **No SSH between pods.** A pod cannot SSH into another pod. Kubernetes pod-to-pod
  communication is via TCP, not shell sessions.
- **No shared filesystem.** There is no network-mounted directory available to all
  pods for a key file.
- **Fixed-size node lists.** The original code required the full list of worker
  node addresses at startup, before any pods existed.
- **Non-routable bind addresses.** Mosaic binds sockets to `0.0.0.0` (all
  interfaces), which is correct for listening but is not a routable address that
  other pods can connect to. The original address auto-detection did not handle
  this case, causing the warehouse subprocesses to advertise `0.0.0.0` to the
  rest of the mesh.

---

## The Core Problem

The fundamental mismatch is one of **initiation direction**:

```
Original (Monitor-initiates):
  Monitor ──SSH/SLURM──► spawns Node
  Node ────handshake───► Monitor

Required for Kubernetes (Node-initiates):
  Monitor ── starts, writes address ──►
  Node ─── reads address, connects ──► Monitor
```

Kubernetes starts pods independently and in parallel. There is no mechanism for the
Monitor pod to spawn the worker pods; that is done by the orchestrator (Argo Workflows).
The only coordination channel available is the Kubernetes network and DNS. The solution
is to invert the handshake: nodes discover and connect to the Monitor rather than the
Monitor spawning the nodes.

---

## Overview of Changes

Six files in `mosaic/` were modified. The changes fall into two categories:

**Category A — Dynamic Mode (phone-home handshake inversion)**

These changes implement the inverted handshake: the monitor enters a waiting state,
and nodes self-register by reading the monitor's address from either a key file or
Kubernetes environment variables.

| File | What changed |
|------|-------------|
| `mosaic/runtime/node.py` | Phone-home logic in `Node.init()`: reads env vars directly |
| `mosaic/runtime/monitor.py` | `init_dynamic()`, `register_node()`, `_get_total_workers()`; key-file write on startup; heartbeat for dynamic nodes |
| `mosaic/runtime/head.py` | `wait_for_workers()` method; num_workers wait made optional |
| `mosaic/cli/mrun.py` | `--dynamic` and `--phone-home` flags; skips node-list discovery in dynamic mode |
| `mosaic/__init__.py` | `phone_home` and `timeout` parameters; independent `address`/`port` handling |

**Category B — Network Address Resolution**

This change fixes a bug that prevented inverse runs from working in any multi-host
environment where the monitor is bound to `0.0.0.0`.

| File | What changed |
|------|-------------|
| `mosaic/comms/comms.py` | `InboundConnection.address` and `Publication.address`: treat `0.0.0.0` as unset; try UDP probe before hostname |

---

## Detailed Code Changes

### 1. `mosaic/runtime/node.py` — Phone-Home Initiation

This is the entry point for the inverted handshake. `Node.init()` gains a
`phone_home` boolean parameter. When set, it reads the monitor's address directly
from environment variables before delegating to `super().init()`.

#### `Node.init()` — Phone-home branch

```diff
+        phone_home = kwargs.get('phone_home', False)
+        if phone_home:
+            self._phone_home = True
+            monitor_host = os.environ.get('MONITOR_HOST')
+            monitor_port = os.environ.get('MONITOR_PORT')
+            pubsub_port  = os.environ.get('PUBSUB_PORT')
+            if not (monitor_host and monitor_port and pubsub_port):
+                raise RuntimeError(
+                    'phone_home=True but MONITOR_HOST, MONITOR_PORT, '
+                    'and PUBSUB_PORT environment variables are not set'
+                )
+            kwargs['monitor_address'] = monitor_host
+            kwargs['monitor_port']    = int(monitor_port)
+            kwargs['pubsub_port']     = int(pubsub_port)
+        else:
+            self._phone_home = False
+
         await super().init(**kwargs)
```

In the Kubernetes deployment the Argo workflow injects the monitor's pod IP into
worker pods via `MONITOR_HOST`, `MONITOR_PORT`, and `PUBSUB_PORT` environment
variables (resolved from the K8s Service DNS name before `mrun` is invoked).
Reading them directly keeps the logic minimal and explicit — no helper function,
no file I/O, no ambiguous path argument.

---

### 2. `mosaic/runtime/monitor.py` — Dynamic Registration

The monitor required the most changes. Three methods were added and one was modified.

#### `Monitor.init()` — Key file and mode dispatch

```diff
-        if kwargs.get('dump_init', False):
+        if kwargs.get('dump_init', False) or self.mode == 'dynamic':
             self.init_file({})

         if self.mode in ['local', 'interactive']:
             await self.init_local(**kwargs)
+        elif self.mode == 'dynamic':
+            await self.init_dynamic(**kwargs)
         else:
             await self.init_cluster(**kwargs)
```

When the monitor starts in `dynamic` mode it unconditionally writes `monitor.key`
to disk. For Kubernetes this is used less often (env vars take precedence), but it
is useful for local testing where nodes connect via a shared file.

#### `init_dynamic()` — New method

```python
async def init_dynamic(self, **kwargs: Any) -> None:
    """
    Init in dynamic mode, waiting for nodes to phone home.

    In this mode, the Monitor does not spawn nodes. Instead, it waits
    for nodes to connect independently.
    """
    num_workers: int = kwargs.get('num_workers', 0)

    self.logger.info('Waiting for nodes to phone home (dynamic mode)')

    if num_workers > 0:
        timeout: float = kwargs.get('timeout', 300)
        tic = time.time()
        while self._get_total_workers() < num_workers:
            if (time.time() - tic) > timeout:
                raise RuntimeError(...)
            await asyncio.sleep(0.1)
```

`init_dynamic()` replaces both `init_local()` and `init_cluster()` for the
Kubernetes use case. Critically, it does **not** SSH into nodes or submit SLURM
jobs. It simply waits. If `num_workers=0` (the default when launched from the Argo
workflow head pod), it returns immediately and the runtime accepts connections as
they arrive — the `k8s_runner.py` script handles the wait explicitly using
`runtime.wait_for_workers()`.

#### `update_node()` — Heartbeat for dynamic nodes

```diff
-        if sender_id not in self._monitored_nodes:
+        is_new_node: bool = sender_id not in self._monitored_nodes
+        if is_new_node:
             self._monitored_nodes[sender_id] = MonitoredResource(sender_id)
+
+            if self.mode == 'dynamic':
+                self._comms.start_heartbeat(sender_id)
+                self.logger.info('Node %s connected (dynamic mode)' % sender_id)
```

In cluster mode, the monitor spawned nodes and started their heartbeats during
`init_cluster()`. In dynamic mode the nodes arrive asynchronously, so the heartbeat
is started on first contact when `update_node()` is called.

#### `register_node()` and `_get_total_workers()` — New helpers

`register_node()` provides an explicit registration pathway (callable via RPC from
a node). `_get_total_workers()` counts workers across all monitored nodes and is
used by both `init_dynamic()` and the head's `wait_for_workers()` indirectly.

---

### 3. `mosaic/runtime/head.py` — Deferred Worker Wait

Two changes were made to the head runtime.

#### `Head.init()` — Optional upfront wait

```diff
-        num_workers = kwargs.pop('num_workers')
-        timeout = 180
-        while len(self.workers) < num_workers:
-            if timeout is not None and (time.time() - tic) > timeout:
-                raise RuntimeError(...)
-            await asyncio.sleep(0.1)
+        num_workers = kwargs.pop('num_workers', 0)
+        if num_workers > 0:
+            tic = time.time()
+            timeout = kwargs.get('timeout', 180)
+            while len(self.workers) < num_workers:
+                ...
```

The original `Head.init()` always blocked until all workers were ready, with the
worker count fixed at startup. The change makes this wait conditional on
`num_workers > 0`. In the Kubernetes deployment the Argo workflow head pod is
launched with `-nw 0`, meaning the head starts immediately and `k8s_runner.py`
explicitly calls `wait_for_workers()` only once the script is ready to begin.
This separates infrastructure startup from the user's script lifecycle.

#### `wait_for_workers()` — New public method

```python
async def wait_for_workers(self, num_workers: int, timeout: Optional[float] = 180) -> int:
    """Wait for a specific number of workers to be available."""
    tic = time.time()
    while len(self.workers) < num_workers:
        if timeout is not None and (time.time() - tic) > timeout:
            raise RuntimeError(...)
        await asyncio.sleep(0.1)
    return len(self.workers)
```

This method is called from `k8s_runner.py` inside the head pod. Separating the
wait into an explicit API call means the user's script can perform any pre-flight
setup before blocking on worker availability, and can also re-wait if workers are
added dynamically mid-run.

---

### 4. `mosaic/cli/mrun.py` — CLI Flags

Two new flags expose the dynamic mode to the command line.

```diff
+@click.option('--dynamic', is_flag=True, default=False,
+              help='run monitor in dynamic mode, waiting for nodes to phone home')
+@click.option('--phone-home', is_flag=True, default=False,
+              help='connect to monitor address from MONITOR_HOST/MONITOR_PORT/PUBSUB_PORT env vars')
```

The `--dynamic` flag sets `mode='dynamic'` and bypasses node-list discovery:

```diff
-    if not local and runtime_type in [None, 'monitor']:
+    if not local and not dynamic and runtime_type in [None, 'monitor']:
         # sun grid engine / slurm / pbs node-list discovery
         ...
```

Without this guard, `mrun` would attempt to read `PE_HOSTFILE`, `SLURM_JOB_NODELIST`,
or `PBS_NODEFILE` — none of which exist in a Kubernetes environment — and fail. The
mode resolution is also made explicit:

```diff
-        'mode': 'local' if local is True else 'cluster',
+        # Determine mode
+        if dynamic:
+            mode = 'dynamic'
+        elif local:
+            mode = 'local'
+        else:
+            mode = 'cluster'
+        ...
+        'mode': mode,
```

In the Argo workflow, the **head pod** command is:
```
mrun --dynamic --address 0.0.0.0 --port 3000 -nw 0 python3 k8s_runner.py
```

The **worker pod** command is:
```
mrun --node --phone-home --address $POD_IP --port 3000 -nw 1 -i <index>
```

`--phone-home` is a boolean flag. When set, `Node.init()` reads `MONITOR_HOST`,
`MONITOR_PORT`, and `PUBSUB_PORT` directly from environment variables and uses
them as the monitor connection parameters.

---

### 5. `mosaic/__init__.py` — Public API

Two parameter additions and one bug fix.

#### New parameters

```diff
 def init(runtime_type='head', ...,
+         phone_home=False, timeout=None,
          ...):
```

`phone_home` is forwarded to `Node.init()`. `timeout` is forwarded to both
`Monitor.init_dynamic()` and `Head.init()` so all three runtimes use a consistent
timeout value.

#### Independent `address` and `port` handling

```diff
-    if address is not None and port is not None:
+    if address is not None:
         runtime_config['address'] = address
+    if port is not None:
         runtime_config['port'] = port
```

The original code required both `address` and `port` to be set, or neither. In
the Kubernetes head pod, the address is set (`0.0.0.0`) but the port may come from
a default. The fix allows them to be set independently.

---

### 6. `mosaic/comms/comms.py` — Address Resolution

This change fixes a silent bug that caused inverse runs to hang on any multi-host
deployment.

#### The bug

When the monitor starts with `--address 0.0.0.0`, its warehouse subprocess inherits
`_address = '0.0.0.0'`. The original `InboundConnection.address` property only
triggered auto-detection when `_address is None`. Since `'0.0.0.0' is not None`,
it was accepted as-is and broadcast to all other runtimes in the network dict.

Worker warehouses on remote pods received `0.0.0.0:3002` as the monitor warehouse's
address and connected to their own localhost, not the head pod. All subsequent
cross-warehouse calls (required by `ScalarField.parameter()` during inversion) were
silently dropped by ZMQ.

#### The fix

```diff
-        if self._address is None:
-            self._address = get_hostname()
-            try:
-                validate_address(self._address)
-            except ValueError:
-                # UDP probe...
+        if self._address is None or self._address == '0.0.0.0':
+            self._address = None
+
+            # UDP probe first (returns routable pod IP without sending packets)
+            try:
+                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
+                try:
+                    s.connect(('8.8.8.8', 53))
+                    self._address = s.getsockname()[0]
+                finally:
+                    s.close()
+            except OSError:
+                pass
+
+            # Hostname fallback
+            if self._address is None:
+                self._address = get_hostname()
+                try:
+                    validate_address(self._address)
+                except ValueError:
+                    self._address = '127.0.0.1'
```

Two changes:
1. `0.0.0.0` is now treated as "not set", resetting `_address` to `None` and
   triggering auto-detection.
2. The detection order is reversed: **UDP probe first, hostname second**. The
   original code tried the hostname first. In Kubernetes, `get_hostname()` returns
   the pod's hostname (e.g. `stride-abc123-head-987654`), which resolves locally
   via `/etc/hosts` but is not DNS-resolvable from other pods. The UDP routing
   table probe (`socket.connect(('8.8.8.8', 53)); getsockname()`) returns the
   pod's actual network IP (e.g. `10.244.0.5`) — routable across all pods — without
   sending any packets.

The fix is applied identically to both `InboundConnection` (ZMQ ROUTER, receives
RPC messages) and `Publication` (ZMQ PUB, sends pubsub messages).

---

## How the Changes Work Together on Kubernetes

The full startup sequence for a typical inversion run on Argo:

```
Argo DAG
│
├─ Step 1: create-service
│    kubectl apply Service monitor-svc-{workflow} → head pod
│    (stable DNS name available before head pod has an IP)
│
├─ Step 2a: head pod starts
│    mrun --dynamic --address 0.0.0.0 --port 3000 -nw 0 python3 k8s_runner.py
│    │
│    ├─ Monitor.init() → mode == 'dynamic'
│    │    writes monitor.key (not used on K8s but available for debug)
│    │    calls init_dynamic(num_workers=0) → returns immediately
│    │
│    └─ k8s_runner.py runs: await runtime.wait_for_workers(N, timeout=300)
│         blocks until N workers connect
│
└─ Step 2b: worker pods start (parallel daemons)
     mrun --node --phone-home --address $POD_IP --port 3000 -nw 1 -i <i>
     │
     ├─ DNS loop: resolve monitor-svc-{workflow} → head pod IP
     │    (waits until Service routes correctly)
     │
     ├─ MONITOR_HOST=<head-pod-ip> injected into env
     │
     ├─ Node.init()
     │    phone_home=True → reads MONITOR_HOST/PORT/PUBSUB_PORT from env
     │    kwargs[monitor_address] = head-pod-ip
     │    super().init() → handshake to monitor
     │
     └─ Monitor.update_node() receives registration
          mode == 'dynamic' → start_heartbeat(sender_id)
          wait_for_workers() counter increments

k8s_runner.py unblocks → runs simple_inverse.py / simple_forward.py
Workers receive shots, compute, return results
Head pod exits → Argo terminates daemon workers
```

The warehouse address fix ensures that at step "handshake to monitor", the network
dict the monitor broadcasts contains `10.244.x.x:3002` (routable pod IP), not
`0.0.0.0:3002`, so subsequent cross-warehouse calls during the inversion work
correctly.

---

## Deployment Architecture

The Kubernetes infrastructure is entirely defined in `k8s/`. A single Argo workflow
drives the entire run:

```
k8s/
├── docker/Dockerfile.stride          # stride-k8s:latest image
├── workflows/stride-workflow.yaml    # Argo DAG (service + head + workers)
├── scripts/
│   ├── k8s_runner.py                 # Head pod entry point
│   ├── simple_forward.py             # Forward-only test
│   ├── simple_inverse.py             # Full FWI (no external storage)
│   └── simple_inverse_s3.py          # FWI + MinIO gradient persistence
└── run.sh                            # Helper: setup / build / start / logs
```

**Image**: `stride-k8s:latest` built with `k8s/docker/Dockerfile.stride`. Multi-stage
build copies `environment.yml` first (cached layer) then installs Stride in editable
mode. The `minio` pip package is included for S3 support.

**Workflow parameters** (all overridable at submit time):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `run-mode` | `forward` | `forward`, `inverse`, or `inverse_s3` |
| `num-workers` | `2` | Worker pod count |
| `workers-per-node` | `1` | Worker processes per pod |
| `image` | `stride-k8s:latest` | Container image |
| `monitor-port` | `3000` | Monitor RPC port |
| `pubsub-port` | `3001` | PubSub port |

**RBAC**: A `stride-workflow` service account in the `argo` namespace with a Role
granting `pods`/`pods/log` and `services` permissions. Required for the
`create-service` workflow step (which creates the monitor K8s Service via `kubectl
apply` from inside the Argo executor).

**Quick start**:
```bash
minikube start --cpus=4 --memory=8192
./k8s/run.sh setup        # install Argo, build image, create RBAC
./k8s/run.sh start        # submit inverse workflow
./k8s/run.sh logs         # stream head pod logs
```

To run a specific mode:
```bash
RUN_MODE=forward  ./k8s/run.sh start
RUN_MODE=inverse  ./k8s/run.sh start
```

To scale up:
```bash
NUM_WORKERS=4 WORKERS_PER_NODE=2 ./k8s/run.sh start
```

---

## Validation

The implementation was tested end-to-end on Minikube. The `inverse` workflow was
submitted with `num-workers=2`, `workers-per-node=1`. The expected tail output from
the head pod logs:

```
Phase 1: Running forward pass to generate observed data...
Forward complete. Observed data saved to: /app/stride/k8s/exps/simple
Phase 2: Starting inversion...
Loaded 2 shots.
Beginning optimisation loop.
...
Updating variable vp,
  grad before processing in range [-4.2e-01, 3.6e-01]
  grad after processing in range [-3.3e-06, 2.9e-06]
  variable range before update [1.46e+03, 1.56e+03]
  taking final update step of 5.0e+00
  variable range after update [1.45e+03, 1.58e+03]
Done iteration 2 (out of 2), block 2 (out of 2) - Total loss 2.05e-02
Inversion complete.
Final model range: [1453.6, 1583.4] m/s
=== inverse Complete ===
```

Both Argo worker pods were marked Succeeded and the workflow completed with exit
code 0.

---

## Remaining Work

The changes in this branch are sufficient to run inversions on Kubernetes for
development and testing on Minikube. The following areas remain for a production
deployment:

**Persistent storage.** The current `simple_inverse.py` writes experiment output
inside the container at `/app/stride/k8s/exps/`. This is lost when the pod exits.
`simple_inverse_s3.py` addresses this by staging shot data and model checkpoints
to MinIO, but a MinIO deployment manifest has not yet been added to the `k8s/`
directory. Committing a `k8s/minio.yaml` and wiring its endpoint into the workflow
as a parameter would complete this.

**Resource limits.** The workflow currently specifies only resource `requests`
(1 CPU, 2Gi memory per pod) and no `limits`. For multi-tenant clusters, matching
requests and limits is recommended.

**Image registry.** The current build targets minikube's local Docker daemon
(`eval $(minikube docker-env)`) and uses `imagePullPolicy: Never`. For a real
cluster the image needs to be pushed to a registry (e.g. ECR, GCR, Docker Hub) and
the workflow's `image` parameter updated accordingly.

**Cloud provider.** The architecture is cloud-agnostic. The Argo workflow, RBAC,
and Mosaic changes are portable to any managed Kubernetes cluster (EKS, GKE, AKS).
The `hostPath` volume in the workflow would need to be replaced with a
`PersistentVolumeClaim` backed by the cloud provider's storage class.

**Autoscaling.** The phone-home architecture already supports adding workers
mid-run. A Kubernetes `HorizontalPodAutoscaler` or a custom controller could submit
additional worker pods to scale up during long inversions without restarting the
head.
