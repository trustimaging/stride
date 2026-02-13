#!/usr/bin/env python
"""
Demo: Cluster Mode (local simulation)

Demonstrates how Mosaic's cluster mode works by distributing a Monte Carlo
estimation of pi across multiple workers. Instead of SSH-ing into remote
machines, we spawn local subprocesses to simulate what monitor.init_cluster
does under the hood.

Architecture (real cluster):
    Head ─→ Monitor ── init_cluster() ── SSH ──→ Node 0 (workers)
                                         SSH ──→ Node 1 (workers)

Architecture (this demo — local simulation):
    Head ─→ Monitor (dynamic mode, writes monitor.key)
              ↑
              ├── Node 0 subprocess (phones home via monitor.key)
              └── Node 1 subprocess (phones home via monitor.key)

The key insight is that init_cluster simply launches `mrun --node` processes
on remote machines via SSH. Here we launch the same `mrun --node` processes
locally, connecting them back to the monitor via the phone-home mechanism.

Usage:
    python demo_cluster_mode.py
    python demo_cluster_mode.py --num-nodes 4 --workers-per-node 2
"""

import argparse
import asyncio
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import random

import mosaic


# =============================================================================
# Configuration
# =============================================================================

NUM_NODES = 2
WORKERS_PER_NODE = 2


# =============================================================================
# Tessera: the distributed actor
# =============================================================================

@mosaic.tessera
class PiEstimator:
    """
    A tessera (remote actor) that estimates pi via Monte Carlo sampling.

    Each instance lives on a remote worker. Methods called on its proxy
    return task handles that resolve asynchronously.
    """

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.total_samples = 0

    async def sample(self, num_samples, seed):
        """
        Throw random darts at a unit square and count how many land
        inside the inscribed quarter-circle.  pi ~ 4 * (hits / total).

        Parameters
        ----------
        num_samples : int
            Number of random points to generate.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            hits (int), samples (int), worker_id (int).
        """
        rng = random.Random(seed)
        hits = 0
        for _ in range(num_samples):
            x = rng.random()
            y = rng.random()
            if x * x + y * y <= 1.0:
                hits += 1

        self.total_samples += num_samples

        return {
            'worker_id': self.worker_id,
            'hits': hits,
            'samples': num_samples,
        }


# =============================================================================
# Main entry point (runs on the Head runtime)
# =============================================================================

async def main(runtime):
    """
    Distribute Monte Carlo pi estimation across all available workers.
    """
    num_workers = len(runtime.workers)
    print(f"\nWorkers available: {num_workers}")
    for w in runtime.workers:
        print(f"  - {w.uid}")

    # Create one PiEstimator tessera per worker.
    # The monitor's round-robin strategy places each on a different worker.
    estimators = [PiEstimator(i) for i in range(num_workers)]

    samples_per_worker = 500_000
    num_rounds = 3

    print(f"\nEstimating pi with {num_workers} workers x "
          f"{samples_per_worker:,} samples x {num_rounds} rounds")
    print(f"Total samples: {num_workers * samples_per_worker * num_rounds:,}")
    print("-" * 50)

    grand_hits = 0
    grand_samples = 0

    for round_idx in range(num_rounds):
        # Submit work to all workers in parallel.
        # Each call returns immediately with a task handle (future).
        tasks = []
        for i, estimator in enumerate(estimators):
            seed = round_idx * num_workers + i
            task = estimator.sample(samples_per_worker, seed)
            tasks.append(task)

        # Await all results concurrently
        results = await asyncio.gather(*tasks)

        # Aggregate
        round_hits = sum(r['hits'] for r in results)
        round_samples = sum(r['samples'] for r in results)
        round_pi = 4.0 * round_hits / round_samples

        grand_hits += round_hits
        grand_samples += round_samples
        running_pi = 4.0 * grand_hits / grand_samples

        print(f"  Round {round_idx}: pi ~ {round_pi:.6f}  "
              f"(running estimate: {running_pi:.6f})")

    final_pi = 4.0 * grand_hits / grand_samples
    print("-" * 50)
    print(f"  Final estimate: pi ~ {final_pi:.8f}")
    print(f"  Actual value:   pi = 3.14159265...")
    print(f"  Error:          {abs(final_pi - 3.141592653589793):.8f}")


# =============================================================================
# Process management — simulates what init_cluster does via SSH
# =============================================================================

def run_monitor():
    """
    Start monitor in dynamic mode.

    In a real cluster, the monitor would call init_cluster() which SSHes
    into nodes.  Here we use dynamic mode so that locally-spawned node
    subprocesses can phone home instead.
    """
    mosaic.init('monitor', mode='dynamic', num_workers=0, log_level='perf')
    runtime = mosaic.runtime()
    loop = runtime.get_event_loop()
    loop.run_forever()


def spawn_nodes(num_nodes, workers_per_node, key_file):
    """
    Spawn local node subprocesses — equivalent to what init_cluster does
    via SSH on remote machines.

    Each subprocess runs:
        mrun --node --phone-home <key_file> -nw <workers> -i <index>

    This is the same command that init_cluster would run remotely:
        ssh <host> "mrun --node -i <index> --monitor-address ... -nw ..."
    """
    processes = []

    for node_idx in range(num_nodes):
        cmd = [
            sys.executable, '-m', 'mosaic.cli.mrun',
            '--node',
            '--phone-home', key_file,
            '-nw', str(workers_per_node),
            '-i', str(node_idx),
            '--inproc',
        ]

        print(f"    Spawning node:{node_idx} with {workers_per_node} workers "
              f"(pid will follow)")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(proc)

    return processes


def cleanup(monitor_proc, node_procs):
    """Terminate all spawned processes."""
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
    except Exception:
        pass


# =============================================================================
# Entry point
# =============================================================================

def run(num_nodes, workers_per_node):
    total_workers = num_nodes * workers_per_node
    workspace = './mosaic-workspace'
    key_file = os.path.join(workspace, 'monitor.key')

    print(f"\nConfiguration:")
    print(f"  Nodes:          {num_nodes} (local subprocesses)")
    print(f"  Workers/node:   {workers_per_node}")
    print(f"  Total workers:  {total_workers}")

    # Clean workspace from previous runs
    if os.path.exists(workspace):
        shutil.rmtree(workspace)

    # ── Step 1: Start monitor ────────────────────────────────────────────
    # In a real cluster, mosaic.run() spawns the monitor automatically and
    # init_cluster() is called inside it.  Here we start it manually so we
    # can spawn local nodes ourselves.
    print(f"\n[1/4] Starting monitor (dynamic mode)...")
    monitor_proc = multiprocessing.Process(target=run_monitor, daemon=True)
    monitor_proc.start()

    # Wait for the monitor to write its key file
    start = time.time()
    while not os.path.exists(key_file):
        if time.time() - start > 30:
            print("ERROR: Monitor failed to start (no key file after 30s)")
            monitor_proc.terminate()
            sys.exit(1)
        time.sleep(0.1)
    print(f"    Monitor ready ({key_file})")

    # ── Step 2: Spawn nodes ──────────────────────────────────────────────
    # This is the local equivalent of monitor.init_cluster(), which would
    # SSH into each node and run `mrun --node`.  We run the same command
    # locally, using --phone-home to connect back to the monitor.
    print(f"\n[2/4] Spawning {num_nodes} node(s) (simulating init_cluster)...")
    node_procs = spawn_nodes(num_nodes, workers_per_node, key_file)

    # ── Step 3: Wait for nodes to register ───────────────────────────────
    print(f"\n[3/4] Waiting for workers to register with monitor...")
    time.sleep(3)

    # ── Step 4: Run the head with our main function ──────────────────────
    print(f"\n[4/4] Running head (main function)...")
    try:
        mosaic.run(
            main,
            mode='dynamic',
            num_workers=total_workers,
            timeout=60,
        )
    finally:
        print("\nCleaning up...")
        cleanup(monitor_proc, node_procs)

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo: Mosaic cluster mode (local simulation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--num-nodes', '-n',
        type=int,
        default=NUM_NODES,
        help=f'Number of simulated nodes (default: {NUM_NODES})',
    )
    parser.add_argument(
        '--workers-per-node', '-nw',
        type=int,
        default=WORKERS_PER_NODE,
        help=f'Workers per node (default: {WORKERS_PER_NODE})',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("=" * 55)
    print("  Mosaic Cluster Mode Demo  —  Monte Carlo Pi")
    print("=" * 55)

    run(
        num_nodes=args.num_nodes,
        workers_per_node=args.workers_per_node,
    )
