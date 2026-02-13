#!/bin/bash
#
# Run Mosaic Inversion on Kubernetes with Argo
#
# Usage:
#   ./k8s/run.sh setup    # First time: install Argo, build image, deploy
#   ./k8s/run.sh start    # Start the inversion (spawn workers + run head)
#   ./k8s/run.sh logs     # Watch the head job logs
#   ./k8s/run.sh status   # Check status of all components
#   ./k8s/run.sh cleanup  # Remove everything
#   ./k8s/run.sh stop     # Stop workers and head (keep monitor/minio)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
NUM_NODES="${NUM_NODES:-2}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-2}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

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

    log "Argo installed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."

    # Point to Minikube's Docker daemon
    eval $(minikube docker-env)

    cd "$PROJECT_DIR"
    docker build -t mosaic-worker:latest -f k8s/Dockerfile .

    log "Image built: mosaic-worker:latest"
}

# Deploy MinIO
deploy_minio() {
    log "Deploying MinIO..."
    kubectl apply -f "$SCRIPT_DIR/minio.yaml"

    log "Waiting for MinIO to be ready..."
    kubectl wait --for=condition=ready pod -l app=minio --timeout=120s

    log "MinIO deployed"
}

# Deploy Monitor
deploy_monitor() {
    log "Deploying Mosaic Monitor..."
    kubectl apply -f "$SCRIPT_DIR/monitor.yaml"

    log "Waiting for Monitor to be ready..."
    kubectl wait --for=condition=ready pod -l app=mosaic-monitor --timeout=120s

    log "Monitor deployed"
}

# Spawn workers via Argo
spawn_workers() {
    log "Spawning $NUM_NODES worker nodes ($WORKERS_PER_NODE workers each)..."

    # Delete any existing worker workflows
    argo delete --all 2>/dev/null || true

    # Submit new workflow
    argo submit "$SCRIPT_DIR/worker-workflow.yaml" \
        -p num-nodes="$NUM_NODES" \
        -p workers-per-node="$WORKERS_PER_NODE" \
        --wait=false

    log "Worker workflow submitted"

    # Wait a bit for workers to connect
    log "Waiting for workers to connect to monitor..."
    sleep 10
}

# Run head job
run_head() {
    log "Running head job (optimisation script)..."

    # Delete existing head job if any
    kubectl delete job mosaic-head 2>/dev/null || true

    # Wait for deletion
    sleep 2

    # Create new head job
    kubectl apply -f "$SCRIPT_DIR/head-job.yaml"

    log "Head job started"
}

# Show logs
show_logs() {
    log "Showing head job logs (Ctrl+C to exit)..."
    kubectl logs -f job/mosaic-head
}

# Show status
show_status() {
    echo ""
    log "=== Kubernetes Status ==="
    echo ""

    echo "--- Pods ---"
    kubectl get pods -o wide
    echo ""

    echo "--- Services ---"
    kubectl get services
    echo ""

    echo "--- Argo Workflows ---"
    argo list 2>/dev/null || echo "(no workflows)"
    echo ""

    echo "--- Jobs ---"
    kubectl get jobs
    echo ""
}

# Stop workers and head
stop() {
    log "Stopping workers and head..."

    # Delete Argo workflows (workers)
    argo delete --all 2>/dev/null || true

    # Delete head job
    kubectl delete job mosaic-head 2>/dev/null || true

    log "Stopped"
}

# Full cleanup
cleanup() {
    log "Cleaning up everything..."

    # Delete Argo workflows
    argo delete --all 2>/dev/null || true

    # Delete all resources
    kubectl delete -f "$SCRIPT_DIR/head-job.yaml" 2>/dev/null || true
    kubectl delete -f "$SCRIPT_DIR/monitor.yaml" 2>/dev/null || true
    kubectl delete -f "$SCRIPT_DIR/minio.yaml" 2>/dev/null || true

    log "Cleanup complete"
}

# Full setup
setup() {
    check_minikube
    install_argo
    build_image
    deploy_minio
    deploy_monitor

    echo ""
    log "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Start the inversion:  ./k8s/run.sh start"
    echo "  2. Watch the logs:       ./k8s/run.sh logs"
    echo "  3. Check status:         ./k8s/run.sh status"
    echo ""
}

# Start inversion
start() {
    check_minikube
    spawn_workers
    run_head

    echo ""
    log "Inversion started!"
    echo ""
    echo "Watch logs with: ./k8s/run.sh logs"
    echo ""
}

# Main
case "${1:-}" in
    setup)
        setup
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
    cleanup)
        cleanup
        ;;
    build)
        check_minikube
        build_image
        ;;
    *)
        echo "Usage: $0 {setup|start|logs|status|stop|cleanup|build}"
        echo ""
        echo "Commands:"
        echo "  setup    - First time setup (install Argo, build image, deploy)"
        echo "  start    - Start the inversion (spawn workers + run head)"
        echo "  logs     - Watch the head job logs"
        echo "  status   - Check status of all components"
        echo "  stop     - Stop workers and head (keep monitor/minio)"
        echo "  cleanup  - Remove everything"
        echo "  build    - Rebuild the Docker image only"
        echo ""
        echo "Environment variables:"
        echo "  NUM_NODES=$NUM_NODES"
        echo "  WORKERS_PER_NODE=$WORKERS_PER_NODE"
        exit 1
        ;;
esac
