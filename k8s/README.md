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
│  │  mosaic.init('node', phone_home='/path/to/monitor.key') │ │
│  │  - Reads monitor address from key file                  │ │
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

#### Added `read_monitor_key()` function
```python
def read_monitor_key(key_file):
    """
    Read monitor connection info from a key file.

    Returns dict with 'monitor_address', 'monitor_port', 'pubsub_port'.
    """
    with open(key_file, 'r') as file:
        lines = file.readlines()

    config = {}
    for line in lines:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            config[key] = value

    return {
        'monitor_address': config['ADD'],
        'monitor_port': int(config['PRT']),
        'pubsub_port': int(config['PUB']),
    }
```

#### Modified `init()` to support phone-home
```python
async def init(self, **kwargs):
    # Handle phone-home mode: read monitor address from key file
    phone_home = kwargs.get('phone_home', None)
    if phone_home is not None:
        self._phone_home = True
        monitor_config = read_monitor_key(phone_home)
        kwargs['monitor_address'] = monitor_config['monitor_address']
        kwargs['monitor_port'] = monitor_config['monitor_port']
        kwargs['pubsub_port'] = monitor_config['pubsub_port']

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
@click.option('--phone-home', type=str, required=False,
              help='path to monitor.key file for node to phone home to monitor')
```

### 5. API Changes (`mosaic/__init__.py`)

#### Updated `init()` signature
```python
def init(runtime_type='head', ..., phone_home=None, timeout=None, ...):
    """
    Parameters
    ----------
    mode : str
        Mode: 'local', 'cluster', or 'dynamic'
    phone_home : str, optional
        Path to monitor.key file for node to phone home
    timeout : float, optional
        Timeout for waiting for workers
    """
```

---

## Usage

### Option 1: CLI (Multiple Terminals)

```bash
# Terminal 1: Start monitor in dynamic mode
mrun --monitor --dynamic

# Terminal 2: Start a node that phones home
mrun --node --phone-home ./mosaic-workspace/monitor.key -nw 4 -i 0

# Terminal 3: Add another node (dynamic scaling!)
mrun --node --phone-home ./mosaic-workspace/monitor.key -nw 4 -i 1

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

See `demo_dynamic_inversion.py` for an example that spawns all processes from a single script.

---

## Kubernetes Deployment

### Architecture on Kubernetes

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  MinIO (S3-compatible storage)                            │ │
│  │  - Stores gradients from workers                          │ │
│  │  - Head aggregates gradients                              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Monitor Pod + Service                                    │ │
│  │  - ClusterIP service: mosaic-monitor:3000                 │ │
│  │  - Workers discover via K8s DNS                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ▲                                      │
│                          │ phone-home (TCP)                     │
│                          │                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Worker Pods (spawned by Argo Workflow)                   │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │ │
│  │  │ Node:0  │ │ Node:1  │ │ Node:2  │ │ Node:3  │  ...    │ │
│  │  │ 2 wkrs  │ │ 2 wkrs  │ │ 2 wkrs  │ │ 2 wkrs  │         │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │ │
│  │                                                           │ │
│  │  Each worker:                                             │ │
│  │  1. Phones home to mosaic-monitor:3000                    │ │
│  │  2. Receives work (shot IDs)                              │ │
│  │  3. Computes gradient                                     │ │
│  │  4. Stores to MinIO                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Head Job                                                 │ │
│  │  - Connects to Monitor                                    │ │
│  │  - Runs OptimisationLoop                                  │ │
│  │  - Distributes shots to workers                           │ │
│  │  - Aggregates gradients from MinIO                        │ │
│  │  - Updates model                                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `k8s/Dockerfile` | Container image with Mosaic + dependencies |
| `k8s/minio.yaml` | MinIO deployment for gradient storage |
| `k8s/monitor.yaml` | Monitor deployment + service |
| `k8s/worker-workflow.yaml` | Argo workflow to spawn worker pods |
| `k8s/head-job.yaml` | Head job that runs the optimisation |
| `k8s/run.sh` | Automation script |
| `k8s_inversion.py` | Main script (runs as monitor/worker/head) |

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
   - Installs Argo Workflows in the cluster
   - Builds `mosaic-worker:latest` Docker image
   - Deploys MinIO for gradient storage
   - Deploys Monitor as a Kubernetes Service

2. **Execution Phase** (`./k8s/run.sh start`)
   - Argo spawns N worker pods in parallel
   - Each worker phones home to `mosaic-monitor:3000`
   - Head job starts and runs OptimisationLoop
   - For each iteration:
     - Head distributes shot IDs to workers
     - Workers compute gradients → store to MinIO
     - Head aggregates gradients from MinIO
     - Head updates model

3. **Dynamic Scaling**
   - Add more workers anytime by submitting another Argo workflow
   - New workers phone home and join the mesh
   - Next iteration uses the new workers

---

## Local Testing

### Without Kubernetes

Run the demo script that spawns everything locally:

```bash
python demo_dynamic_inversion.py
```

This script:
1. Starts Monitor in dynamic mode (subprocess)
2. Spawns Node processes that phone home
3. Runs OptimisationLoop with dummy work

### With Minikube

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Run the full setup and test
./k8s/run.sh setup
./k8s/run.sh start
./k8s/run.sh logs
```

Expected output:
```
============================================================
HEAD: Starting Optimisation Loop
============================================================

Workers available: 4
  - worker:0:0
  - worker:0:1
  - worker:1:0
  - worker:1:1

Configuration:
  Blocks: 2
  Iterations/block: 3
  Shots: 16
  Workers: 4

============================================================
BLOCK 0
============================================================

--- Iteration 0 (abs: 0) ---
  Computing gradients on workers...
    Worker 0: 4 shots, norm=142.35
    Worker 1: 4 shots, norm=138.92
    Worker 2: 4 shots, norm=145.21
    Worker 3: 4 shots, norm=140.88
  Aggregating gradients from MinIO...
    Total gradient norm: 567.36
  Model range: [1443.2, 1556.8]

--- Iteration 1 (abs: 1) ---
...
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
# Monitor
mrun --monitor --dynamic

# Node (phone-home)
mrun --node --phone-home /path/to/monitor.key -nw 4 -i 0

# With explicit addresses (for K8s)
python k8s_inversion.py --worker  # Uses env vars
python k8s_inversion.py --head    # Uses env vars
python k8s_inversion.py --monitor # Uses env vars
```

---

## Troubleshooting

### Workers not connecting

1. Check Monitor is running and accessible:
   ```bash
   kubectl logs -l app=mosaic-monitor
   ```

2. Check worker logs:
   ```bash
   kubectl logs -l app=mosaic-worker
   ```

3. Verify service is reachable:
   ```bash
   kubectl run test --rm -it --image=busybox -- nc -zv mosaic-monitor 3000
   ```

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
