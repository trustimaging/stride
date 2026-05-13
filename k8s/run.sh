#!/bin/bash
#
# Run Mosaic Inversion on Kubernetes with Argo
#
# Usage:
#   ./k8s/run.sh setup    # First time: install Argo, build image, create RBAC
#   ./k8s/run.sh build    # (Re)build the Docker image
#   ./k8s/run.sh start    # Submit the Argo inversion workflow
#   ./k8s/run.sh logs     # Stream logs from the active workflow
#   ./k8s/run.sh status   # Check status of all components
#   ./k8s/run.sh stop     # Delete active workflows
#   ./k8s/run.sh cleanup  # Remove all workflows and RBAC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
NUM_WORKERS="${NUM_WORKERS:-2}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-1}"
RUN_MODE="${RUN_MODE:-inverse}"
ARGO_NAMESPACE="${ARGO_NAMESPACE:-argo}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error(){ echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if minikube is running
check_minikube() {
    if ! minikube status &>/dev/null; then
        error "Minikube is not running. Start it with: minikube start --cpus=4 --memory=8192"
    fi
    log "Minikube is running"
}

# Install Argo Workflows
install_argo() {
    log "Installing Argo Workflows..."

    if kubectl get namespace argo &>/dev/null; then
        log "Argo namespace already exists"
    else
        kubectl create namespace argo
    fi

    kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/quick-start-minimal.yaml

    log "Waiting for Argo to be ready..."
    kubectl wait -n argo --for=condition=available deployment/argo-server --timeout=120s || true
    kubectl wait -n argo --for=condition=available deployment/workflow-controller --timeout=120s || true

    # The quick-start bundles its own MinIO Deployment for Argo artifact
    # storage. We use our own StatefulSet (k8s/manifests/minio.yaml), so
    # delete the bundled one to avoid two MinIO pods behind the same Service.
    kubectl delete deployment minio -n argo 2>/dev/null || true

    # Expose Argo UI via NodePort so it's accessible from the host
    kubectl patch svc argo-server -n argo -p '{"spec":{"type":"NodePort"}}'
    log "Argo installed (argo-server patched to NodePort)"
}

# Create RBAC for the stride-workflow service account
setup_rbac() {
    log "Setting up RBAC for stride-workflow service account..."

    kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: stride-workflow
  namespace: argo
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: stride-workflow
  namespace: argo
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch", "create", "delete", "patch"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["get", "list", "watch", "create", "delete", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: stride-workflow
  namespace: argo
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: stride-workflow
subjects:
  - kind: ServiceAccount
    name: stride-workflow
    namespace: argo
EOF

    log "RBAC configured"
}

# Deploy MinIO to the argo namespace
deploy_minio() {
    log "Deploying MinIO..."
    kubectl apply -f "$SCRIPT_DIR/manifests/minio.yaml"
    log "Waiting for MinIO to be ready..."
    kubectl wait -n "$ARGO_NAMESPACE" --for=condition=ready pod \
        minio-0 --timeout=120s
    # Expose MinIO via NodePort so it's accessible from the host
    kubectl patch svc minio -n argo -p '{"spec":{"type":"NodePort"}}'

    # Create buckets: stride-data for app artifacts, argo-artifacts for Argo log archiving
    kubectl exec -n "$ARGO_NAMESPACE" minio-0 -- mkdir -p /data/stride-data /data/argo-artifacts

    # Point Argo's artifact repository at the dedicated bucket
    kubectl patch configmap workflow-controller-configmap -n "$ARGO_NAMESPACE" --type merge \
        -p '{"data":{"artifactRepository":"s3:\n  bucket: argo-artifacts\n  endpoint: minio:9000\n  insecure: true\n  accessKeySecret:\n    name: my-minio-cred\n    key: accesskey\n  secretKeySecret:\n    name: my-minio-cred\n    key: secretkey\n"}}'

    log "MinIO is ready (API: minio.argo.svc.cluster.local:9000)"
    log "  Console: minikube service minio -n argo --url"
}

# Build Docker image inside minikube's Docker daemon
build_image() {
    log "Building Docker image stride-k8s:latest..."

    eval $(minikube docker-env)

    cd "$PROJECT_DIR"
    docker build -t stride-k8s:latest -f k8s/docker/Dockerfile.stride .

    log "Image built: stride-k8s:latest"
}

# Submit the Argo workflow
start() {
    check_minikube

    log "Submitting Argo workflow (mode=$RUN_MODE, workers=$NUM_WORKERS, workers-per-node=$WORKERS_PER_NODE)..."

    argo submit "$SCRIPT_DIR/workflows/stride-workflow.yaml" \
        -n "$ARGO_NAMESPACE" \
        -p run-mode="$RUN_MODE" \
        -p num-workers="$NUM_WORKERS" \
        -p workers-per-node="$WORKERS_PER_NODE"

    echo ""
    log "Workflow submitted!"
    echo ""
    echo "Watch logs with:   ./k8s/run.sh logs"
    echo "Check status with: ./k8s/run.sh status"
    echo ""
}

# Stream logs from the most recent workflow
show_logs() {
    local workflow
    workflow=$(argo list -n "$ARGO_NAMESPACE" --running --output name 2>/dev/null | head -1)
    if [ -z "$workflow" ]; then
        workflow=$(argo list -n "$ARGO_NAMESPACE" --output name 2>/dev/null | head -1)
    fi
    if [ -z "$workflow" ]; then
        error "No workflows found in namespace $ARGO_NAMESPACE"
    fi
    log "Streaming logs for $workflow (Ctrl+C to exit)..."
    argo logs -n "$ARGO_NAMESPACE" "$workflow" --follow
}

# Show status of all components
show_status() {
    echo ""
    log "=== Argo Workflows ==="
    argo list -n "$ARGO_NAMESPACE" 2>/dev/null || echo "(no workflows)"
    echo ""

    log "=== Pods ==="
    kubectl get pods -n "$ARGO_NAMESPACE" -o wide
    echo ""

    log "=== Services ==="
    kubectl get services -n "$ARGO_NAMESPACE"
    echo ""
}

# Stop all active workflows
stop() {
    log "Deleting active workflows..."
    argo delete --all -n "$ARGO_NAMESPACE" 2>/dev/null || true
    log "Stopped"
}

# Download plots from MinIO to local misc/plots/
export_plots() {
    local dest="$PROJECT_DIR/misc/plots"
    mkdir -p "$dest"

    # Get MinIO API NodePort
    local node_port
    node_port=$(kubectl get svc minio -n "$ARGO_NAMESPACE" -o jsonpath='{.spec.ports[?(@.name=="api")].nodePort}')
    local minio_ip
    minio_ip=$(minikube ip)
    local endpoint="${minio_ip}:${node_port}"

    log "Downloading plots from MinIO ($endpoint)..."

    python3 -c "
import os, sys
from minio import Minio

client = Minio('${endpoint}', access_key='admin', secret_key='password', secure=False)
dest = '${dest}'
found = False
for obj in client.list_objects('stride-data', prefix='plots/', recursive=True):
    if obj.object_name.endswith('.png'):
        fname = os.path.basename(obj.object_name)
        client.fget_object('stride-data', obj.object_name, os.path.join(dest, fname))
        print('  %s → %s/%s' % (obj.object_name, dest, fname))
        found = True
if not found:
    print('  No plots found in stride-data/plots/')
    sys.exit(1)
" || warn "No plots found in MinIO"
}

# Remove finished workflows, stale MinIO data, and Docker build cache
tidy() {
    log "Tidying up stale resources..."

    # Delete completed/failed/errored Argo workflows (keeps running ones)
    argo delete --completed -n "$ARGO_NAMESPACE" 2>/dev/null || true
    argo delete --status Error -n "$ARGO_NAMESPACE" 2>/dev/null || true
    argo delete --status Failed -n "$ARGO_NAMESPACE" 2>/dev/null || true

    # Clear old run data from MinIO
    kubectl exec -n "$ARGO_NAMESPACE" minio-0 -- \
        sh -c 'rm -rf /data/stride-data/stride-*' 2>/dev/null || true

    # Prune Docker build cache and dangling images
    minikube ssh "docker builder prune -a -f" 2>/dev/null || true
    minikube ssh "docker image prune -f" 2>/dev/null || true

    log "Tidy complete"
}

# Remove all workflows and RBAC
cleanup() {
    log "Cleaning up..."

    argo delete --all -n "$ARGO_NAMESPACE" 2>/dev/null || true
    kubectl delete serviceaccount stride-workflow -n "$ARGO_NAMESPACE" 2>/dev/null || true
    kubectl delete role stride-workflow -n "$ARGO_NAMESPACE" 2>/dev/null || true
    kubectl delete rolebinding stride-workflow -n "$ARGO_NAMESPACE" 2>/dev/null || true

    log "Cleanup complete"
}

# Full first-time setup
setup() {
    check_minikube
    install_argo
    setup_rbac
    deploy_minio
    build_image

    echo ""
    log "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Start an inversion: ./k8s/run.sh start"
    echo "  2. Watch the logs:     ./k8s/run.sh logs"
    echo "  3. Check status:       ./k8s/run.sh status"
    echo ""
}

# Main
case "${1:-}" in
    setup)
        setup
        ;;
    build)
        check_minikube
        build_image
        ;;
    minio)
        check_minikube
        deploy_minio
        ;;
    start)
        start
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    stop)
        stop
        ;;
    plots)
        check_minikube
        export_plots
        ;;
    tidy)
        check_minikube
        tidy
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {setup|build|start|logs|status|stop|plots|tidy|cleanup}"
        echo ""
        echo "Commands:"
        echo "  setup    - First-time setup (install Argo, build image, create RBAC)"
        echo "  build    - Rebuild the Docker image"
        echo "  start    - Submit the Argo inversion workflow"
        echo "  logs     - Stream logs from the active workflow"
        echo "  status   - Check status of all components"
        echo "  stop     - Delete active workflows"
        echo "  plots    - Export result plots from minikube to misc/plots/"
        echo "  tidy     - Remove finished workflows, stale MinIO data, Docker cache"
        echo "  cleanup  - Remove all workflows and RBAC"
        echo ""
        echo "Environment variables:"
        echo "  NUM_WORKERS=$NUM_WORKERS"
        echo "  WORKERS_PER_NODE=$WORKERS_PER_NODE"
        echo "  RUN_MODE=$RUN_MODE          (forward|inverse|inverse_artifacts)"
        echo "  ARGO_NAMESPACE=$ARGO_NAMESPACE"
        exit 1
        ;;
esac
