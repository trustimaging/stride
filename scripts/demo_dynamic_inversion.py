#!/usr/bin/env python
"""
Demo: Dynamic Mode with OptimisationLoop

Tests the phone-home mechanism by:
1. Starting monitor in dynamic mode
2. Spawning worker nodes that phone home
3. Running iterations using OptimisationLoop with dummy work on each worker
"""

import os
import sys
import time
import shutil
import asyncio
import subprocess
import multiprocessing
import numpy as np

import mosaic
from stride.optimisation import OptimisationLoop


# =============================================================================
# Configuration
# =============================================================================

NUM_NODES = 6
WORKERS_PER_NODE = 2
TOTAL_WORKERS = NUM_NODES * WORKERS_PER_NODE

NUM_BLOCKS = 2
NUM_ITERATIONS = 3


# =============================================================================
# Dummy Worker Tessera
# =============================================================================

@mosaic.tessera
class DummyWorker:
    """A simple worker that performs dummy calculations."""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.call_count = 0

    async def dummy(self, iteration_id, data):
        """
        Perform a dummy calculation.

        Parameters
        ----------
        iteration_id : int
            Current iteration number.
        data : np.ndarray
            Input data to process.

        Returns
        -------
        dict
            Result containing worker_id, processed data, and stats.
        """
        self.call_count += 1

        # Simulate some work
        await asyncio.sleep(0.1)

        # Do a simple calculation
        result = np.sum(data ** 2) + self.worker_id * 10

        return {
            'worker_id': self.worker_id,
            'iteration': iteration_id,
            'result': float(result),
            'call_count': self.call_count,
        }


# =============================================================================
# Main Inversion Logic
# =============================================================================

async def main(runtime):
    """Main routine using OptimisationLoop."""

    print("\n" + "=" * 60)
    print("Running Optimisation Loop")
    print("=" * 60)

    num_workers = len(runtime.workers)
    print(f"\nWorkers available: {num_workers}")
    for w in runtime.workers:
        print(f"  - {w.uid}")

    # Create one DummyWorker tessera per worker
    print(f"\nCreating {num_workers} DummyWorker tessera(s)...")
    workers = [DummyWorker(i) for i in range(num_workers)]

    # Create optimisation loop (without a problem, just for iteration tracking)
    opt_loop = OptimisationLoop(name='test_loop')

    # Run the optimisation loop
    print(f"\nRunning {NUM_BLOCKS} blocks x {NUM_ITERATIONS} iterations")
    print("-" * 60)

    all_results = []

    for block in opt_loop.blocks(NUM_BLOCKS):
        print(f"\n=== Block {block.id} ===")

        for iteration in block.iterations(NUM_ITERATIONS):
            print(f"\n  Iteration {iteration.id} (abs: {iteration.abs_id})")

            # Create dummy input data for each worker
            tasks = []
            for i, worker in enumerate(workers):
                data = np.random.randn(100) * (iteration.abs_id + 1)
                task = worker.dummy(iteration.abs_id, data)
                tasks.append(task)

            # Run all workers in parallel
            results = await asyncio.gather(*tasks)

            # Collect results
            total_result = sum(r['result'] for r in results)
            print(f"    Workers completed: {len(results)}")
            print(f"    Total result: {total_result:.2f}")

            for r in results:
                print(f"      Worker {r['worker_id']}: result={r['result']:.2f}, calls={r['call_count']}")

            all_results.append({
                'block': block.id,
                'iteration': iteration.id,
                'total': total_result,
            })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total iterations: {len(all_results)}")
    print(f"Total workers used: {num_workers}")

    for r in all_results:
        print(f"  Block {r['block']}, Iter {r['iteration']}: {r['total']:.2f}")

    print("\nTest complete!")


# =============================================================================
# Process Management
# =============================================================================

def run_monitor():
    """Run monitor in dynamic mode."""
    mosaic.init('monitor', mode='dynamic', num_workers=0, log_level='perf')
    runtime = mosaic.runtime()
    loop = runtime.get_event_loop()
    loop.run_forever()


def spawn_nodes(num_nodes, workers_per_node, key_file):
    """Spawn worker nodes that phone home."""
    processes = []

    for node_idx in range(num_nodes):
        cmd = [
            sys.executable, '-m', 'mosaic.cli.mrun',
            '--node',
            '--phone-home', key_file,
            '-nw', str(workers_per_node),
            '-i', str(node_idx),
            '--inproc'
        ]

        print(f"    Spawning node:{node_idx} with {workers_per_node} workers")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)

    return processes


def cleanup(monitor_proc, node_procs):
    """Clean up all processes."""
    for proc in node_procs:
        proc.terminate()
    for proc in node_procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    monitor_proc.terminate()
    try:
        monitor_proc.join(timeout=5)
    except:
        pass


# =============================================================================
# Entry Point
# =============================================================================

def run_test():
    """Run the complete test."""

    print("=" * 60)
    print("Dynamic Mode Test with OptimisationLoop")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Nodes: {NUM_NODES}")
    print(f"  Workers/node: {WORKERS_PER_NODE}")
    print(f"  Total workers: {TOTAL_WORKERS}")
    print(f"  Blocks: {NUM_BLOCKS}")
    print(f"  Iterations/block: {NUM_ITERATIONS}")

    # Clean workspace
    workspace = './mosaic-workspace'
    if os.path.exists(workspace):
        shutil.rmtree(workspace)

    # Start monitor
    print("\n[1] Starting monitor (dynamic mode)...")
    monitor_proc = multiprocessing.Process(target=run_monitor)
    monitor_proc.start()

    # Wait for key file
    key_file = os.path.join(workspace, 'monitor.key')
    start = time.time()
    while not os.path.exists(key_file):
        if time.time() - start > 30:
            print("ERROR: Monitor failed to start")
            monitor_proc.terminate()
            sys.exit(1)
        time.sleep(0.1)
    print(f"    Monitor ready: {key_file}")

    # Spawn nodes
    print(f"\n[2] Spawning {NUM_NODES} nodes...")
    node_procs = spawn_nodes(NUM_NODES, WORKERS_PER_NODE, key_file)

    # Wait for nodes to connect
    print("\n[3] Waiting for workers to connect...")
    time.sleep(3)

    # Run main
    print("\n[4] Running optimisation loop...")
    try:
        mosaic.run(main, mode='dynamic', num_workers=TOTAL_WORKERS, timeout=120)
    finally:
        print("\n[5] Cleaning up...")
        cleanup(monitor_proc, node_procs)

    print("\nDone!")


if __name__ == '__main__':
    run_test()
