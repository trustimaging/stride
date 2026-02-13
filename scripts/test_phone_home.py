#!/usr/bin/env python
"""
Test script for phone-home dynamic mode.

Run this script to test the reversed handshake mechanism.

Usage:
    # Option 1: Run the full test (starts monitor and nodes automatically)
    python test_phone_home.py

    # Option 2: Manual testing with multiple terminals
    # Terminal 1: python test_phone_home.py --monitor
    # Terminal 2: python test_phone_home.py --node 0
    # Terminal 3: python test_phone_home.py --node 1
"""

import os
import sys
import time
import argparse
import subprocess
import signal

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def start_monitor_dynamic():
    """Start monitor in dynamic mode."""
    import mosaic

    print("Starting monitor in dynamic mode...")
    print("Monitor key will be written to: ./mosaic-workspace/monitor.key")

    # Start monitor in dynamic mode, don't wait for workers
    mosaic.init('monitor', mode='dynamic', num_workers=0, log_level='info')

    print("Monitor is ready and waiting for nodes to phone home.")
    print("\nTo connect nodes, run in separate terminals:")
    print("  python test_phone_home.py --node 0")
    print("  python test_phone_home.py --node 1")

    # Keep running
    runtime = mosaic.runtime()
    loop = runtime.get_event_loop()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
        mosaic.stop()


def start_node_phone_home(node_index, num_workers=2):
    """Start a node that phones home to the monitor."""
    import mosaic

    key_file = './mosaic-workspace/monitor.key'

    print(f"Starting node:{node_index} with {num_workers} workers...")
    print(f"Reading monitor address from: {key_file}")

    # Wait for key file to exist
    timeout = 30
    start = time.time()
    while not os.path.exists(key_file):
        if time.time() - start > timeout:
            print(f"ERROR: Monitor key file not found after {timeout}s: {key_file}")
            print("Make sure the monitor is running first:")
            print("  python test_phone_home.py --monitor")
            sys.exit(1)
        time.sleep(0.5)

    # Start node with phone-home
    mosaic.init('node',
                runtime_indices=(node_index,),
                phone_home=key_file,
                num_workers=num_workers,
                log_level='info',
                wait=True)


def run_full_test():
    """Run full test with monitor and nodes."""
    import multiprocessing
    import mosaic

    print("=" * 60)
    print("Testing Phone-Home Dynamic Mode")
    print("=" * 60)

    # Clean up any existing key file
    key_file = './mosaic-workspace/monitor.key'
    if os.path.exists(key_file):
        os.remove(key_file)

    # Start monitor in a separate process
    def run_monitor():
        import mosaic
        mosaic.init('monitor', mode='dynamic', num_workers=0, log_level='info')
        runtime = mosaic.runtime()
        loop = runtime.get_event_loop()
        loop.run_forever()

    print("\n[1] Starting monitor in dynamic mode...")
    monitor_proc = multiprocessing.Process(target=run_monitor)
    monitor_proc.start()

    # Wait for monitor to write key file
    timeout = 10
    start = time.time()
    while not os.path.exists(key_file):
        if time.time() - start > timeout:
            print("ERROR: Monitor did not write key file")
            monitor_proc.terminate()
            sys.exit(1)
        time.sleep(0.2)

    print(f"   Monitor key file created: {key_file}")

    # Read and display the key file
    from mosaic.runtime import read_monitor_key
    config = read_monitor_key(key_file)
    print(f"   Monitor address: {config['monitor_address']}:{config['monitor_port']}")
    print(f"   PubSub port: {config['pubsub_port']}")

    # Start nodes in separate processes
    def run_node(index, num_workers):
        import mosaic
        mosaic.init('node',
                    runtime_indices=(index,),
                    phone_home='./mosaic-workspace/monitor.key',
                    num_workers=num_workers,
                    log_level='info',
                    wait=True)

    print("\n[2] Starting node:0 with 2 workers (phone-home)...")
    node0_proc = multiprocessing.Process(target=run_node, args=(0, 2))
    node0_proc.start()

    # Give node 0 time to register
    time.sleep(3)

    print("\n[3] Starting node:1 with 2 workers (phone-home)...")
    node1_proc = multiprocessing.Process(target=run_node, args=(1, 2))
    node1_proc.start()

    # Give node 1 time to register
    time.sleep(3)

    print("\n[4] Testing dynamic worker addition...")
    print("   Starting node:2 with 1 worker (late joiner)...")
    node2_proc = multiprocessing.Process(target=run_node, args=(2, 1))
    node2_proc.start()

    # Wait a bit then check status
    time.sleep(3)

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print("\nIf you saw 'Successfully registered with monitor' messages")
    print("for each node, the phone-home mechanism is working correctly!")
    print("\nCleaning up...")

    # Clean up
    node2_proc.terminate()
    node1_proc.terminate()
    node0_proc.terminate()
    monitor_proc.terminate()

    node2_proc.join(timeout=5)
    node1_proc.join(timeout=5)
    node0_proc.join(timeout=5)
    monitor_proc.join(timeout=5)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Test phone-home dynamic mode')
    parser.add_argument('--monitor', action='store_true',
                        help='Start monitor in dynamic mode')
    parser.add_argument('--node', type=int, metavar='INDEX',
                        help='Start node with given index that phones home')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of workers per node (default: 2)')

    args = parser.parse_args()

    if args.monitor:
        start_monitor_dynamic()
    elif args.node is not None:
        start_node_phone_home(args.node, args.workers)
    else:
        # Run full automated test
        run_full_test()


if __name__ == '__main__':
    main()
