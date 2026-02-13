#!/usr/bin/env python
"""
Kubernetes + Argo + MinIO Inversion Demo

This script demonstrates:
1. Monitor runs as a K8s service (workers phone home)
2. Workers compute gradients and store to MinIO
3. Head runs OptimisationLoop, aggregates gradients from MinIO

Can run in 3 modes:
  --monitor  : Run as the monitor service
  --worker   : Run as a worker node (phones home to monitor)
  --head     : Run the optimisation script
"""

import os
import sys
import json
import asyncio
import argparse
import numpy as np
from io import BytesIO

import mosaic


# =============================================================================
# Configuration (from environment or defaults)
# =============================================================================

MONITOR_HOST = os.environ.get('MONITOR_HOST', 'localhost')
MONITOR_PORT = int(os.environ.get('MONITOR_PORT', '3000'))
PUBSUB_PORT = int(os.environ.get('PUBSUB_PORT', '3001'))

MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.environ.get('MINIO_BUCKET', 'gradients')

NODE_INDEX = int(os.environ.get('NODE_INDEX', '0'))
WORKERS_PER_NODE = int(os.environ.get('WORKERS_PER_NODE', '2'))

NUM_BLOCKS = 2
NUM_ITERATIONS = 3
GRID_SHAPE = (100, 100)


# =============================================================================
# MinIO Helper
# =============================================================================

def get_minio_client():
    """Get MinIO client."""
    from minio import Minio
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def ensure_bucket(client, bucket_name):
    """Ensure bucket exists."""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def save_gradient_to_minio(client, iteration_id, worker_id, gradient):
    """Save gradient array to MinIO."""
    ensure_bucket(client, MINIO_BUCKET)

    # Serialize numpy array
    buffer = BytesIO()
    np.save(buffer, gradient)
    buffer.seek(0)

    object_name = f"iter_{iteration_id}/gradient_worker_{worker_id}.npy"
    client.put_object(
        MINIO_BUCKET,
        object_name,
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type='application/octet-stream'
    )
    return object_name


def load_gradients_from_minio(client, iteration_id):
    """Load all gradients for an iteration from MinIO."""
    prefix = f"iter_{iteration_id}/"
    gradients = []

    objects = client.list_objects(MINIO_BUCKET, prefix=prefix)
    for obj in objects:
        response = client.get_object(MINIO_BUCKET, obj.object_name)
        buffer = BytesIO(response.read())
        gradient = np.load(buffer)
        gradients.append(gradient)
        response.close()
        response.release_conn()

    return gradients


def clear_iteration_gradients(client, iteration_id):
    """Clear gradients for an iteration."""
    prefix = f"iter_{iteration_id}/"
    objects = client.list_objects(MINIO_BUCKET, prefix=prefix)
    for obj in objects:
        client.remove_object(MINIO_BUCKET, obj.object_name)


# =============================================================================
# Worker Tessera
# =============================================================================

@mosaic.tessera
class GradientWorker:
    """Worker that computes fake gradients and stores to MinIO."""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.minio_client = None

    def _get_minio(self):
        if self.minio_client is None:
            self.minio_client = get_minio_client()
        return self.minio_client

    async def compute_gradient(self, iteration_id, shot_ids):
        """
        Compute fake gradient for assigned shots and store to MinIO.

        Parameters
        ----------
        iteration_id : int
            Current iteration
        shot_ids : list
            Shot IDs assigned to this worker

        Returns
        -------
        dict
            Status and metadata
        """
        # Simulate gradient computation
        await asyncio.sleep(0.2)

        # Create fake gradient (in real code, this would be adjoint result)
        gradient = np.zeros(GRID_SHAPE, dtype=np.float32)

        for shot_id in shot_ids:
            # Add contribution from each shot
            np.random.seed(shot_id + iteration_id * 1000)
            shot_gradient = np.random.randn(*GRID_SHAPE).astype(np.float32)

            # Add some structure
            cx, cy = GRID_SHAPE[0] // 2, GRID_SHAPE[1] // 2
            for i in range(GRID_SHAPE[0]):
                for j in range(GRID_SHAPE[1]):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < 20:
                        shot_gradient[i, j] *= 2.0

            gradient += shot_gradient

        # Store to MinIO
        client = self._get_minio()
        object_name = save_gradient_to_minio(
            client, iteration_id, self.worker_id, gradient
        )

        return {
            'worker_id': self.worker_id,
            'iteration_id': iteration_id,
            'shots_processed': len(shot_ids),
            'gradient_norm': float(np.linalg.norm(gradient)),
            'minio_path': object_name,
        }


# =============================================================================
# Head Script (Optimisation Loop)
# =============================================================================

async def run_head(runtime):
    """Run the optimisation loop on the head."""
    from stride.optimisation import OptimisationLoop

    print("\n" + "=" * 60)
    print("HEAD: Starting Optimisation Loop")
    print("=" * 60)

    # Wait for workers
    num_workers = len(runtime.workers)
    if num_workers == 0:
        print("Waiting for workers to connect...")
        await runtime.wait_for_workers(1, timeout=300)
        num_workers = len(runtime.workers)

    print(f"\nWorkers available: {num_workers}")
    for w in runtime.workers:
        print(f"  - {w.uid}")

    # Create worker tesserae
    workers = [GradientWorker(i) for i in range(num_workers)]

    # Setup MinIO client for aggregation
    minio_client = get_minio_client()
    ensure_bucket(minio_client, MINIO_BUCKET)

    # Fake shot IDs
    num_shots = 16
    shot_ids = list(range(num_shots))

    # Create optimisation loop
    opt_loop = OptimisationLoop(name='k8s_inversion')

    # Model (simple array)
    model = np.ones(GRID_SHAPE, dtype=np.float32) * 1500.0
    step_size = 0.1

    print(f"\nConfiguration:")
    print(f"  Blocks: {NUM_BLOCKS}")
    print(f"  Iterations/block: {NUM_ITERATIONS}")
    print(f"  Shots: {num_shots}")
    print(f"  Workers: {num_workers}")

    # Run optimisation
    for block in opt_loop.blocks(NUM_BLOCKS):
        print(f"\n{'='*60}")
        print(f"BLOCK {block.id}")
        print(f"{'='*60}")

        for iteration in block.iterations(NUM_ITERATIONS):
            print(f"\n--- Iteration {iteration.id} (abs: {iteration.abs_id}) ---")

            # Clear previous gradients
            clear_iteration_gradients(minio_client, iteration.abs_id)

            # Distribute shots across workers
            shots_per_worker = len(shot_ids) // num_workers
            worker_tasks = []

            for i, worker in enumerate(workers):
                start_idx = i * shots_per_worker
                end_idx = start_idx + shots_per_worker
                if i == num_workers - 1:
                    end_idx = len(shot_ids)

                assigned_shots = shot_ids[start_idx:end_idx]
                task = worker.compute_gradient(iteration.abs_id, assigned_shots)
                worker_tasks.append(task)

            # Wait for all workers
            print("  Computing gradients on workers...")
            results = await asyncio.gather(*worker_tasks)

            for r in results:
                print(f"    Worker {r['worker_id']}: {r['shots_processed']} shots, "
                      f"norm={r['gradient_norm']:.2f}")

            # Aggregate gradients from MinIO
            print("  Aggregating gradients from MinIO...")
            gradients = load_gradients_from_minio(minio_client, iteration.abs_id)

            total_gradient = np.sum(gradients, axis=0)
            print(f"    Total gradient norm: {np.linalg.norm(total_gradient):.2f}")

            # Update model
            model = model - step_size * total_gradient
            model = np.clip(model, 1400, 1700)

            print(f"  Model range: [{model.min():.1f}, {model.max():.1f}]")

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMISATION COMPLETE")
    print("=" * 60)
    print(f"Final model range: [{model.min():.1f}, {model.max():.1f}]")


# =============================================================================
# Entry Points
# =============================================================================

def run_monitor_mode():
    """Run as monitor service."""
    print("Starting MONITOR in dynamic mode...")
    print(f"  Listening on port {MONITOR_PORT}")
    print(f"  PubSub on port {PUBSUB_PORT}")

    mosaic.init(
        'monitor',
        mode='dynamic',
        port=MONITOR_PORT,
        num_workers=0,
        log_level='info'
    )

    runtime = mosaic.runtime()
    loop = runtime.get_event_loop()

    print("Monitor ready. Waiting for workers to phone home...")
    loop.run_forever()


def run_worker_mode():
    """Run as worker node (phones home to monitor)."""
    print(f"Starting WORKER NODE {NODE_INDEX}...")
    print(f"  Workers per node: {WORKERS_PER_NODE}")
    print(f"  Phoning home to: {MONITOR_HOST}:{MONITOR_PORT}")

    mosaic.init(
        'node',
        runtime_indices=(NODE_INDEX,),
        monitor_address=MONITOR_HOST,
        monitor_port=MONITOR_PORT,
        pubsub_port=PUBSUB_PORT,
        num_workers=WORKERS_PER_NODE,
        log_level='info',
        wait=True
    )


def run_head_mode():
    """Run as head (optimisation script)."""
    print("Starting HEAD...")
    print(f"  Connecting to monitor: {MONITOR_HOST}:{MONITOR_PORT}")

    mosaic.run(
        run_head,
        monitor_address=MONITOR_HOST,
        monitor_port=MONITOR_PORT,
        pubsub_port=PUBSUB_PORT,
        num_workers=0,  # Don't wait, we'll wait inside
        mode='dynamic'
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='K8s Mosaic Inversion')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--monitor', action='store_true', help='Run as monitor')
    group.add_argument('--worker', action='store_true', help='Run as worker node')
    group.add_argument('--head', action='store_true', help='Run as head (script)')

    args = parser.parse_args()

    if args.monitor:
        run_monitor_mode()
    elif args.worker:
        run_worker_mode()
    elif args.head:
        run_head_mode()


if __name__ == '__main__':
    main()
