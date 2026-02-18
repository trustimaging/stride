# K8s Argo Workflow

Run Stride forward and inverse modeling on a local Minikube cluster using Argo Workflows.

---

## Architecture

```
Argo DAG Workflow
│
├─ 1. create-service     K8s Service (DNS: monitor-svc-<name>)
│         │                 ports 3000 (RPC) + 3001 (PubSub)
│         │
│    ┌────┴────┐
│    ▼         ▼
├─ 2. head    workers (x N, daemon)
│    │         │
│    │  mrun --node --phone-home env
│    │         │
│    │  ◄──────┘  workers phone home via Service DNS
│    │
│    mrun --dynamic (embedded monitor on :3000)
│    │
│    k8s_runner.py
│      ├─ wait_for_workers(N)
│      └─ run_mode_main(runtime)   # forward, inverse, or inverse_s3
│
└─ head exits → Argo kills daemon workers → Service auto-deleted
```

---

## Step-by-Step Flow

### 1. Create Service

Argo applies a K8s Service named `monitor-svc-<workflow-name>`. This gives
workers a stable DNS name to reach the head pod. The Service has an
`ownerReference` pointing to the Workflow, so it is automatically deleted
when the workflow finishes.

Selector: `role: head, workflow: <workflow-name>` — matches labels on the
head pod.

Ports:
- 3000 → RPC (mosaic control messages)
- 3001 → PubSub (mosaic pub/sub channel)

### 2. Head + Workers Start in Parallel

Once the Service exists, Argo launches the head and all worker pods
simultaneously.

**Head pod** runs:
```
python -m mosaic.cli.mrun --dynamic --address 0.0.0.0 --port 3000 \
    -nw 0 python k8s_runner.py
```

This does two things:
1. `mrun` starts an embedded mosaic **monitor** listening on `0.0.0.0:3000`
   (PubSub on `:3001`), then writes a `monitor.key` file.
2. `mrun` launches `k8s_runner.py` as a subprocess. The script calls
   `mosaic.run(main, address=POD_IP)` which reads `monitor.key` and
   connects as a **head** to the local monitor, using the pod's IP as its
   advertised address (so workers can reach it via the network dict
   returned during the handshake).

**Worker pods** (one per `withSequence` item) each run:
```
python -m mosaic.cli.mrun --node --phone-home env \
    --address $POD_IP --port 3000 \
    -nw <workers-per-node> -i <index>
```

Each worker reads `MONITOR_HOST`, `MONITOR_PORT`, and `PUBSUB_PORT` from
environment variables and phones home to the monitor running inside the
head pod (routed via the K8s Service). The `--address $POD_IP` flag
ensures the worker advertises its routable pod IP (not its hostname,
which isn't DNS-resolvable from other pods).

### 3. Wait for Workers

Inside `k8s_runner.py`, the head calls:
```python
await runtime.wait_for_workers(TOTAL_WORKERS, timeout=300)
```

This blocks until all worker pods have successfully phoned home. If workers
start before the monitor is ready, mosaic's phone-home mechanism retries
automatically.

### 4. Run Script

Once all workers are connected, the runner selects a script based on the
`RUN_MODE` environment variable (set via the `run-mode` workflow parameter):

| `run-mode` | Module | Description |
|------------|--------|-------------|
| `forward` | `scripts.simple_forward` | Forward modeling (default) |
| `inverse` | `scripts.simple_inverse` | Gradient-descent inversion |
| `inverse_s3` | `scripts.simple_inverse_s3` | Inversion with S3 persistence |

```python
script = import_module(SCRIPTS[RUN_MODE])
await script.main(runtime, exp_name=EXP_NAME)
```

The `exp-name` parameter (default `"simple"`) is passed via the `EXP_NAME`
environment variable and used to create the experiment directory
(`exps/<exp-name>/`). Each experiment gets its own isolated directory.

The inverse scripts are **self-contained**: they run a forward pass first
(Phase 1) to generate observed data with the true model, then run the
inversion (Phase 2) from a homogeneous initial guess. No separate forward
submission is needed.

### 5. Cleanup

When the head pod exits:
- Argo terminates all **daemon** worker pods automatically.
- The K8s Service is garbage-collected via its `ownerReference` to the
  Workflow.

---

## Files

| File | Purpose |
|------|---------|
| `k8s/Dockerfile.stride` | Docker image based on `devitocodes/bases:cpu-gcc` with Stride + mosaic installed |
| `k8s/k8s_runner.py` | Head entrypoint — waits for workers, runs forward/inverse script |
| `k8s/stride-workflow.yaml` | Argo DAG workflow definition |

---

## Workflow Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `run-mode` | `forward` | Script to run: `forward`, `inverse`, or `inverse_s3` |
| `num-workers` | `2` | Number of worker pods |
| `workers-per-node` | `1` | Mosaic workers per pod |
| `monitor-port` | `3000` | Monitor RPC port |
| `pubsub-port` | `3001` | Monitor PubSub port |
| `image` | `stride-k8s:latest` | Container image |

---

## Usage

### 1. Start Minikube

```bash
minikube start --cpus=4 --memory=12288 --disk-size=50g
```

The full Stride conda environment is large, so 12 GB memory and 50 GB disk
are recommended.

### 2. Install Argo Workflows

Create the `argo` namespace and install Argo:

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/quick-start-minimal.yaml
```

Wait for Argo to be ready:

```bash
kubectl wait -n argo --for=condition=available deployment/argo-server --timeout=120s
kubectl wait -n argo --for=condition=available deployment/workflow-controller --timeout=120s
```

Verify:

```bash
kubectl get pods -n argo
```

You should see `argo-server` and `workflow-controller` pods running.

### 3. Set Up RBAC

The workflow needs permission to create K8s Services (so workers can
discover the head pod via DNS) and to patch pods (Argo uses this to
manage pod status). Create a ServiceAccount and grant admin access in
the `argo` namespace:

```bash
kubectl create serviceaccount stride-workflow -n argo
kubectl create clusterrolebinding stride-workflow-pods --clusterrole=admin --serviceaccount=argo:stride-workflow
```

Verify:

```bash
kubectl get serviceaccount stride-workflow -n argo
kubectl get clusterrolebinding stride-workflow-pods
```

Note: these only need to be created once. They persist across workflow
submissions.

### Build the Docker Image

Minikube runs its own Docker daemon inside the VM. To make your image
available to the cluster without pushing to a registry, build it directly
inside Minikube's Docker:

```bash
# Point your shell at Minikube's Docker daemon
eval $(minikube docker-env)

# Build the image from the project root
docker build -t stride-k8s:latest -f k8s/Dockerfile.stride .
```

`eval $(minikube docker-env)` sets `DOCKER_HOST` and related env vars so
that `docker build` targets the Minikube VM, not your host machine. This
must be run in every new terminal session.

Verify the image works:

```bash
docker run --rm stride-k8s:latest python3 -c "import mosaic; print('ok')"
```

The workflow YAML references `imagePullPolicy: Never`, which tells
Kubernetes to use the local image and never try to pull from a remote
registry.

#### Docker Layer Caching

The Dockerfile is structured to maximise layer caching:

1. `environment.yml` is copied first (rarely changes).
2. `conda env create` + `pip install minio` run in a single layer — cached
   unless `environment.yml` changes.
3. The rest of the source code is then copied and `pip install -e .` runs
   on top of the cached environment.

This means code-only changes rebuild in ~30 seconds instead of ~10 minutes.

#### Rebuilding After Code Changes

```bash
eval $(minikube docker-env)
docker build -t stride-k8s:latest -f k8s/Dockerfile.stride .
```

#### Cleaning Up Old Images

If you run out of disk space:

```bash
eval $(minikube docker-env)

# Remove build cache (can grow to tens of GB)
docker builder prune --all -f

# Remove stopped containers and dangling images
docker container prune -f
docker image prune -f
```

To check current disk usage:

```bash
docker system df
```

#### Using a Remote Registry (Optional)

For real clusters (not Minikube), push the image to a registry:

```bash
docker build -t registry.example.com/stride-k8s:latest -f k8s/Dockerfile.stride .
docker push registry.example.com/stride-k8s:latest
```

Then override the image parameter when submitting:

```bash
argo submit k8s/stride-workflow.yaml -p image="registry.example.com/stride-k8s:latest"
```

Remove `imagePullPolicy: Never` from the workflow YAML when using a remote
registry.

### Submit Workflow

```bash
# Forward (default)
argo submit k8s/stride-workflow.yaml -n argo

# Inverse
argo submit k8s/stride-workflow.yaml -n argo -p run-mode="inverse"

# Inverse with S3 persistence
argo submit k8s/stride-workflow.yaml -n argo -p run-mode="inverse_s3"
```

Override parameters:

```bash
argo submit k8s/stride-workflow.yaml -n argo \
    -p run-mode="inverse" \
    -p num-workers="4" \
    -p workers-per-node="2"
```

### Watch Logs

```bash
argo logs -f @latest -n argo
```

### Check Status

```bash
argo list -n argo
kubectl get pods -n argo
kubectl get svc -n argo
```

### Cleanup

```bash
argo delete --all -n argo
```

---

## Troubleshooting

### Workers crash silently (exit code 1, no output)

Mosaic workers in `mode='cluster'` send log output to the remote monitor
(not local stdout). If the worker crashes before completing the
handshake, its logs are never delivered. To debug:

1. Add `-u` and `2>&1` to the worker command to capture Python
   tracebacks (which go to stderr).
2. Remove `exec` from the bash script so the exit code is captured.
3. Check the **head pod** logs — once connected, worker log messages
   appear there via the monitor's remote logger.

### `validate_address` error on pod hostname

```
ValueError: Address and port combination <pod-hostname>:3003 is not valid
```

Pod hostnames (e.g., `stride-forward-xxx-head-123456`) are not
DNS-resolvable from other pods. Both the head and worker runtimes must
advertise their **pod IP** (via `status.podIP`), not the hostname.

- **Head**: `k8s_runner.py` passes `address=os.environ['POD_IP']` to
  `mosaic.run()`.
- **Workers**: use `--address $POD_IP` in the mrun command.

### Warehouse subprocess address resolution

The monitor's warehouse subprocess inherits `0.0.0.0` as its bind address
from the monitor. Prior to the fix in `mosaic/comms/comms.py`, the
warehouse would advertise `0.0.0.0` as its routable address in the
network dict sent to other runtimes during the handshake. Worker
warehouses on other pods would then try to connect to `0.0.0.0:<port>`,
which resolves to their own localhost — not the head pod.

This manifests as a **silent hang** during inverse runs (forward works
because it doesn't use `publish=True` on warehouse objects). See
`docs/warehouse-address-bug.md` for the full investigation.

The fix: `InboundConnection.address` and `Publication.address` in
`mosaic/comms/comms.py` now treat `0.0.0.0` as "not yet set" and
auto-detect a routable IP via a UDP probe (`connect()` to `8.8.8.8:53`,
read `getsockname()`). This returns the pod's actual network IP.

### `--local` flag breaks cross-pod connections

Do **not** use `--local` on worker nodes. In mosaic, `mode='local'`
forces all outbound connections to `127.0.0.1` (see
`comms.py:Connection._local`), which prevents cross-pod communication.

### Docker build fails with "no route to host"

The Docker CLI is pointing at a stale minikube Docker daemon. Re-run:

```bash
eval $(minikube docker-env)
```

### Docker "no space left on device"

The build cache inside Minikube can grow to 30+ GB. Clean it:

```bash
eval $(minikube docker-env)
docker builder prune --all -f
```

### RBAC permission errors

If you see "forbidden: cannot patch resource pods", the service account
needs broader permissions:

```bash
kubectl create clusterrolebinding stride-workflow-pods --clusterrole=admin --serviceaccount=argo:stride-workflow
```
