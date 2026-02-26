# Mosaic Dynamic Mode: Phone-Home Architecture

This document describes the changes made to Mosaic to support dynamic worker registration ("phone-home") and how to run distributed workloads on Kubernetes.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Changes](#architecture-changes)
3. [Code Changes](#code-changes)
4. [Usage](#usage)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Local Testing](#local-testing)

---

## Overview

### The Problem

In the original Mosaic architecture, the **Monitor must know all worker locations upfront**:

```
ORIGINAL FLOW:
  Monitor ──SSH/SLURM──► Node (Monitor spawns nodes, passes its address)
  Node ──handshake──► Monitor
```

This design has limitations:
- Worker count is fixed at startup
- Monitor must be able to SSH into worker nodes
- Cannot dynamically add workers during runtime
- Difficult to use with container orchestrators (Kubernetes, etc.)

### The Solution: Phone-Home

We reversed the handshake so **workers discover and connect to the Monitor**:

```
NEW FLOW (phone-home):
  Monitor writes address to monitor.key file (or K8s Service)
  Node reads monitor.key (started independently)
  Node ──handshake──► Monitor
  Node ──update_node──► Monitor (registration complete)
```

Benefits:
- Workers can be added at any time
- Works with Kubernetes, Argo, or any orchestrator
- No SSH required
- Dynamic scaling during runtime

---

## Architecture Changes

### Original Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User calls: mosaic.init('head', num_workers=4)              │
│                                                              │
│  Head ──► spawns Monitor subprocess                          │
│  Monitor ──► SSH/SLURM to spawn Nodes (needs node_list)      │
│  Nodes ──► spawn Workers                                     │
│  Workers ──► handshake back to Monitor                       │
│                                                              │
│  Problem: Monitor must know node addresses upfront           │
└──────────────────────────────────────────────────────────────┘
```

### New Architecture (Dynamic Mode)

```
┌──────────────────────────────────────────────────────────────┐
│  MONITOR (started first, waits for connections)              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  mosaic.init('monitor', mode='dynamic')                 │ │
│  │  - Writes monitor.key with address/port                 │ │
│  │  - Waits for nodes to phone home                        │ │
│  │  - Accepts registrations via update_node()              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ▲                                   │
│                          │ phone-home                        │
│                          │                                   │
│  NODES (started independently, discover monitor)             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  mosaic.init('node', phone_home=True)                  │ │
│  │  - Reads monitor address from MONITOR_HOST/PORT env vars│ │
│  │  - Initiates handshake to monitor                       │ │
│  │  - Spawns workers                                       │ │
│  │  - Calls update_node() to complete registration         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  HEAD (connects to existing monitor)                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  mosaic.run(main, mode='dynamic')                       │ │
│  │  - Connects to existing monitor                         │ │
│  │  - Runs user's computation                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Code Changes

### 1. Monitor Changes (`mosaic/runtime/monitor.py`)

#### Added `init_dynamic()` method
```python
async def init_dynamic(self, **kwargs):
    """
    Init in dynamic mode, waiting for nodes to phone home.

    In this mode, the Monitor does not spawn nodes. Instead, it waits
    for nodes to connect independently.
    """
    num_workers = kwargs.get('num_workers', 0)
    self.logger.info('Waiting for nodes to phone home (dynamic mode)')

    if num_workers > 0:
        # Optionally wait for minimum workers
        timeout = kwargs.get('timeout', 300)
        while self._get_total_workers() < num_workers:
            await asyncio.sleep(0.1)
```

#### Modified `update_node()` to start heartbeat for dynamic nodes
```python
def update_node(self, sender_id, update, sub_resources):
    is_new_node = sender_id not in self._monitored_nodes
    if is_new_node:
        self._monitored_nodes[sender_id] = MonitoredResource(sender_id)

        # In dynamic mode, start heartbeat for newly connected nodes
        if self.mode == 'dynamic':
            self._comms.start_heartbeat(sender_id)
            self.logger.info('Node %s connected (dynamic mode)' % sender_id)

    # ... rest of method
```

#### Modified `init()` to write key file in dynamic mode
```python
# Write key file if requested or in dynamic mode
if kwargs.get('dump_init', False) or self.mode == 'dynamic':
    self.init_file({})
```

### 2. Node Changes (`mosaic/runtime/node.py`)

#### Modified `init()` to support phone-home
```python
async def init(self, **kwargs):
    # Handle phone-home mode: read monitor address from environment variables
    phone_home = kwargs.get('phone_home', False)
    if phone_home:
        self._phone_home = True
        monitor_host = os.environ.get('MONITOR_HOST')
        monitor_port = os.environ.get('MONITOR_PORT')
        pubsub_port  = os.environ.get('PUBSUB_PORT')
        if not (monitor_host and monitor_port and pubsub_port):
            raise RuntimeError(
                'phone_home=True but MONITOR_HOST, MONITOR_PORT, '
                'and PUBSUB_PORT environment variables are not set'
            )
        kwargs['monitor_address'] = monitor_host
        kwargs['monitor_port']    = int(monitor_port)
        kwargs['pubsub_port']     = int(pubsub_port)

    await super().init(**kwargs)
    # ... rest of init
```

### 3. Head Changes (`mosaic/runtime/head.py`)

#### Added `wait_for_workers()` method
```python
async def wait_for_workers(self, num_workers, timeout=180):
    """
    Wait for a specific number of workers to be available.
    Useful in dynamic mode where workers connect at runtime.
    """
    tic = time.time()
    while len(self.workers) < num_workers:
        if timeout is not None and (time.time() - tic) > timeout:
            raise RuntimeError('Timed out waiting for workers')
        await asyncio.sleep(0.1)
    return len(self.workers)
```

### 4. CLI Changes (`mosaic/cli/mrun.py`)

#### Added new command-line options
```python
@click.option('--dynamic', is_flag=True, default=False,
              help='run monitor in dynamic mode, waiting for nodes to phone home')
@click.option('--phone-home', is_flag=True, default=False,
              help='connect to monitor via MONITOR_HOST/MONITOR_PORT/PUBSUB_PORT env vars')
```

### 5. API Changes (`mosaic/__init__.py`)

#### Updated `init()` signature
```python
def init(runtime_type='head', ..., phone_home=False, timeout=None, ...):
    """
    Parameters
    ----------
    mode : str
        Mode: 'local', 'cluster', or 'dynamic'
    phone_home : bool, optional
        If True, read monitor address from MONITOR_HOST/MONITOR_PORT/PUBSUB_PORT env vars
    timeout : float, optional
        Timeout for waiting for workers
    """
```

### 6. Warehouse Address Fix (`mosaic/comms/comms.py`)

#### Problem

Even after passing `--address $POD_IP` to the head and worker runtimes,
the warehouse subprocesses spawned internally by mosaic still advertised
`0.0.0.0` as their routable address. The monitor starts with
`--address 0.0.0.0` (correct for binding), and its warehouse subprocess
inherits this value. The original address auto-detection was guarded by
`if self._address is None`, so `0.0.0.0` was accepted as-is and
broadcast to other runtimes in the network dict during the handshake.

This broke inverse runs on K8s: `ScalarField.parameter()` creates proxy
objects that trigger cross-warehouse `push_remote(publish=True)` calls,
which require the monitor's warehouse to be reachable from node warehouses.
Node warehouses connected to `0.0.0.0:port` on their own pod (interpreting
`0.0.0.0` as localhost), causing the optimisation loop to hang silently.
Forward runs were unaffected because `ScalarField()` uses simple
store-and-retrieve via the monitor — no cross-warehouse callbacks needed.

#### Fix

Modified both `InboundConnection.address` and `Publication.address` in
`mosaic/comms/comms.py` to treat `0.0.0.0` the same as `None`, triggering
auto-detection. The probe order was also changed: UDP probe first (returns
the pod's actual routable IP without sending any packets), then hostname as
a fallback. The original code tried hostname first, which returned the pod
hostname — locally resolvable but not DNS-resolvable from other pods.

```python
@property
def address(self):
    if self._address is None or self._address == '0.0.0.0':
        # 0.0.0.0 is valid for binding but not routable — auto-detect
        self._address = None

        # UDP probe: query the routing table to get the pod's network IP.
        # No packets are sent; s.getsockname() returns the local address
        # the OS would use to reach 8.8.8.8.
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 53))
                self._address = s.getsockname()[0]
            finally:
                s.close()
        except OSError:
            pass

        # Hostname fallback (only if UDP probe failed)
        if self._address is None:
            self._address = get_hostname()
            try:
                validate_address(self._address)
            except ValueError:
                self._address = '127.0.0.1'

    return self._address
```

The same logic is applied to both `InboundConnection` (ZMQ ROUTER socket
for receiving RPC messages) and `Publication` (ZMQ PUB socket for pubsub).

See `docs/warehouse-address-bug.md` for the full debugging investigation.

---

## Usage

### Option 1: CLI (Multiple Terminals)

```bash
# Terminal 1: Start monitor in dynamic mode (writes monitor.key)
mrun --dynamic

# Terminal 2: Start a node that phones home
MONITOR_HOST=<address> MONITOR_PORT=3000 PUBSUB_PORT=3001 \
  mrun --node --phone-home -nw 4 -i 0

# Terminal 3: Add another node (dynamic scaling!)
MONITOR_HOST=<address> MONITOR_PORT=3000 PUBSUB_PORT=3001 \
  mrun --node --phone-home -nw 4 -i 1

# Terminal 4: Run your script
python my_inversion.py
```

### Option 2: Python API

```python
import mosaic

async def main(runtime):
    # Wait for workers if needed
    if len(runtime.workers) == 0:
        await runtime.wait_for_workers(4, timeout=300)

    # Run computation
    print(f"Running with {len(runtime.workers)} workers")
    # ...

# Start in dynamic mode
mosaic.run(main, mode='dynamic', num_workers=4)
```

### Option 3: Programmatic Process Spawning

See `k8s/scripts/demo_phone_home.py` for an example that spawns all processes from a single script.

---

## Kubernetes Deployment

### Architecture on Kubernetes

The entire distributed run is orchestrated by a single Argo DAG workflow.
The monitor is **embedded in the head pod** (not a separate deployment):
`mrun --dynamic` starts the monitor inline, then runs `k8s_runner.py` as
the user script. A K8s Service is created first so workers can discover
the head pod by a stable DNS name.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Kubernetes Cluster (argo namespace)            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  K8s Service: monitor-svc-{workflow-name}                 │ │
│  │  Selector: role=head  →  routes :3000/:3001 to Head pod   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ▲                                      │
│                          │ phone-home (TCP :3000)               │
│                          │                                      │
│  ┌─────────────┐         │    ┌─────────────┐                   │
│  │  Worker Pod │─────────┤    │  Worker Pod │  ...             │ │
│  │  Node:0     │         │    │  Node:1     │                   │
│  │  mrun --node│         └───►│  mrun --node│                   │
│  │  --phone-   │              │  --phone-   │                   │
│  │  home env   │              │  home env   │                   │
│  └─────────────┘              └─────────────┘                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Head Pod  (label: role=head)                             │ │
│  │  mrun --dynamic  ← embeds Monitor, writes no key file     │ │
│  │  k8s_runner.py   ← waits for workers, then runs script   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The Argo DAG runs steps in order:
1. `create-service` — creates the K8s Service
2. `head` + `workers` (parallel, both depend on service)
Workers resolve `monitor-svc-{workflow-name}` via K8s DNS to get the head
pod IP and connect. When the head pod exits the workers are terminated
automatically (Argo daemon pods).

### k8s Directory Overview

```
k8s/
├── docker/
│   └── Dockerfile.stride          # Multi-stage image: conda env + stride install
├── scripts/
│   ├── k8s_runner.py              # Head pod entry point: waits for workers, dispatches script
│   ├── simple_forward.py          # 2D acoustic forward simulation
│   ├── simple_inverse.py          # Full-waveform inversion (self-contained, no S3)
│   ├── simple_inverse_s3.py       # FWI with MinIO gradient/model persistence
│   └── demo_phone_home.py         # Local multi-process phone-home demo
├── workflows/
│   └── stride-workflow.yaml       # Argo DAG: service → head + workers
└── run.sh                         # Automation: setup / build / start / logs / status / stop
```

**`k8s/docker/Dockerfile.stride`** — Three-stage build (`anaconda → builder → user`). Copies
`environment.yml` first so the expensive conda-environment layer is cached unless dependencies
change. Installs `minio` pip package for S3 support. Copies `k8s/scripts/k8s_runner.py` to
the working directory root (`/app/stride/`) so the Argo workflow can launch it with
`python3 k8s_runner.py`.

**`k8s/scripts/k8s_runner.py`** — Entry point for the head pod. Reads `NUM_WORKERS`,
`WORKERS_PER_NODE`, `RUN_MODE`, and `EXP_NAME` from environment variables. Calls
`runtime.wait_for_workers()` before importing and running the selected script module.
Passes `POD_IP` to `mosaic.run()` so the head's runtime advertises the correct pod IP.

**`k8s/scripts/simple_forward.py`** — Standalone 2D acoustic forward run: 100×100 grid,
layered velocity model (1500/1600 m/s), 8 transducers, 500 kHz tone burst. Validates
worker connectivity and PDE compilation.

**`k8s/scripts/simple_inverse.py`** — Two-phase FWI: (1) forward pass to generate observed
data, (2) inversion from a homogeneous initial model using L2 loss and gradient descent.
2 blocks × 2 iterations. Self-contained — no S3 required.

**`k8s/scripts/simple_inverse_s3.py`** — Same as `simple_inverse.py` but stages shot data
and saves models/gradients to MinIO at each iteration. Reads S3 credentials from
`MINIO_*` environment variables.

**`k8s/scripts/demo_phone_home.py`** — Spawns a full monitor + nodes + head locally using
`subprocess`, demonstrating the phone-home flow without Kubernetes.

**`k8s/workflows/stride-workflow.yaml`** — Argo DAG with three templates: `monitor-service`
(creates a K8s Service that selects the head pod), `head` (runs `mrun --dynamic` with
`k8s_runner.py`), and `worker` (daemon pods that phone home via env vars). Parameterised
for `run-mode`, `num-workers`, `workers-per-node`, `image`.

**`k8s/run.sh`** — Helper script wrapping common operations. See Quick Start below.

### Quick Start

```bash
# Prerequisites: Minikube running
minikube start --cpus=4 --memory=8192

# First-time setup
./k8s/run.sh setup

# Run the inversion
./k8s/run.sh start

# Watch logs
./k8s/run.sh logs

# Check status
./k8s/run.sh status

# Scale up (add more workers)
NUM_NODES=4 WORKERS_PER_NODE=4 ./k8s/run.sh start

# Cleanup
./k8s/run.sh cleanup
```

### Workflow Details

1. **Setup Phase** (`./k8s/run.sh setup`)
   - Installs Argo Workflows v3.5.0 in the `argo` namespace
   - Builds `stride-k8s:latest` Docker image inside minikube's Docker daemon
   - Creates the `stride-workflow` service account with pod/service RBAC

2. **Execution Phase** (`./k8s/run.sh start`)
   - Submits `k8s/workflows/stride-workflow.yaml` to the `argo` namespace
   - Argo DAG runs:
     1. Creates `monitor-svc-{workflow-name}` K8s Service (routes → head pod)
     2. Launches head pod: `mrun --dynamic` embeds monitor, then `k8s_runner.py` waits for workers and runs the configured script
     3. Launches N worker daemon pods in parallel: each resolves the service DNS, connects via phone-home, and waits for shots
   - Worker daemons are automatically terminated when the head pod exits

3. **Dynamic Scaling**
   - Submit additional worker-only workflows targeting the same monitor service to add workers mid-run
   - New workers phone home and join the mesh; next iteration uses them

---

## Local Testing

### Without Kubernetes

Run the demo script that spawns everything locally:

```bash
python k8s/scripts/demo_phone_home.py
```

This script:
1. Starts Monitor in dynamic mode (subprocess)
2. Spawns Node processes that phone home
3. Runs a dummy computation to verify the mesh

### With Minikube

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Run the full setup and test
./k8s/run.sh setup
./k8s/run.sh start
./k8s/run.sh logs
```

Expected output (tail of head pod logs for `run-mode=inverse`):
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

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITOR_HOST` | `localhost` | Monitor hostname (for workers/head) |
| `MONITOR_PORT` | `3000` | Monitor comms port |
| `PUBSUB_PORT` | `3001` | Monitor pubsub port |
| `NODE_INDEX` | `0` | Node index (for workers) |
| `WORKERS_PER_NODE` | `2` | Workers per node |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `gradients` | Bucket for gradients |

### CLI Options

```bash
# Start monitor in dynamic mode (waits for nodes to phone home)
mrun --dynamic --address 0.0.0.0 --port 3000 -nw 0 python3 k8s_runner.py

# Start a node that phones home (reads MONITOR_HOST/MONITOR_PORT/PUBSUB_PORT from env)
MONITOR_HOST=<address> MONITOR_PORT=3000 PUBSUB_PORT=3001 \
  mrun --node --phone-home -nw 1 -i 0
```

---

## Troubleshooting

### Workers not connecting

1. Check the head pod logs (contains monitor output):
   ```bash
   kubectl logs -n argo -l role=head
   ```

2. Check worker pod logs:
   ```bash
   kubectl logs -n argo -l role=worker
   ```

3. Verify the monitor service is reachable from a worker pod:
   ```bash
   kubectl run test -n argo --rm -it --image=busybox -- nc -zv monitor-svc-<workflow-name> 3000
   ```

### Address resolution in Kubernetes

All mosaic runtimes must advertise **routable IP addresses**, not
hostnames. In K8s, pod hostnames (e.g., `stride-forward-xxx-head-123`)
are not DNS-resolvable from other pods. Use `status.podIP`:

- **Workers**: pass `--address $POD_IP` to `mrun`.
- **Head**: pass `address=os.environ['POD_IP']` to `mosaic.run()`.

Without this, the handshake fails with:
```
ValueError: Address and port combination <hostname>:<port> is not valid
```

The `--local` flag must **not** be used in K8s — it forces all
connections to `127.0.0.1` (see `comms.py:Connection._local`).

### Inverse hangs silently after "Beginning optimisation loop"

This was caused by the warehouse address bug — see [Code Change 6](#6-warehouse-address-fix-mosaiccommscommspy) above and `docs/warehouse-address-bug.md` for the full debugging investigation. The fix is already in `mosaic/comms/comms.py`.

### Head times out waiting for workers

Increase timeout or check worker status:
```bash
./k8s/run.sh status
argo list
```

### MinIO connection issues

Check MinIO is running:
```bash
kubectl logs -l app=minio
kubectl port-forward svc/minio 9001:9001
# Open http://localhost:9001 in browser
```
