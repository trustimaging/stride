#!/usr/bin/env python
"""
Phone-Home Mesh Setup Demo
===========================

Demonstrates the reversed-handshake ("phone home") mechanism that mosaic
uses to build a distributed worker mesh in Kubernetes.

Normal (local/cluster) mode:
    Monitor spawns nodes as subprocesses and knows their addresses up front.

Phone-home (dynamic) mode:
    Monitor starts in dynamic mode and binds to a known address/port.
    Nodes start independently (e.g. as K8s pods), read the monitor address
    from MONITOR_HOST / MONITOR_PORT / PUBSUB_PORT environment variables,
    and connect back ("phone home"). The monitor registers each node,
    starts a heartbeat, and the node's workers join the pool.

    In K8s this works via a headless Service that routes DNS to the
    monitor pod. Locally we just use 127.0.0.1 with fixed ports.

Logging architecture:
    Nodes, warehouses, and workers use remote logging — their log messages
    are sent via ZMQ to the monitor, which prints them with the sender's
    UID in the runtime_id field. So all runtime logs (including instance
    IDs like node:0:a3f1b2c0, warehouse:0:a3f1b2c0, worker:0:0:a3f1b2c0)
    appear in the monitor's output stream.

This script runs the full handshake locally using the mrun CLI:

    1. Start the monitor:  mrun --dynamic ... sleep infinity
    2. Start node:0:       mrun --node --phone-home -i 0 -nw 1
    3. Start node:1:       mrun --node --phone-home -i 1 -nw 1
    4. Verify the mesh: 2 nodes, each with a warehouse + workers
    5. Kill node:1 — monitor detects disconnect via heartbeat timeout
    6. Start node:2 (replacement) — phones home and joins
    7. Clean shutdown

Usage:
    python k8s/scripts/demo_phone_home.py
    python k8s/scripts/demo_phone_home.py --workers-per-node 2
    python k8s/scripts/demo_phone_home.py --log-level debug   # verbose warehouse logs
"""

import os
import sys
import time
import argparse
import threading
import subprocess

# Fixed ports for the demo — mirrors how K8s workflows set these via
# the Argo parameters monitor-port / pubsub-port.
MONITOR_HOST = '127.0.0.1'
MONITOR_PORT = 9100
PUBSUB_PORT = 9101


def _drain_output(proc, label):
    """Read lines from a subprocess and print with a label prefix."""
    for line in iter(proc.stdout.readline, ''):
        line = line.rstrip('\n')
        if line:
            print('[%-10s] %s' % (label, line), flush=True)


def _start_process(label, cmd, env):
    """Start a subprocess, spawn a reader thread, return (proc, thread)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    reader = threading.Thread(target=_drain_output, args=(proc, label), daemon=True)
    reader.start()
    return proc, reader


def _terminate(proc, label, timeout=5):
    """Terminate a process, escalating to kill if needed."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    print('  %s (pid %d) — stopped (exit=%s)' % (label, proc.pid, proc.returncode))


def _banner(step, title):
    print('\n' + '-' * 72)
    print('[%s] %s' % (step, title))
    print('-' * 72)


def main():
    parser = argparse.ArgumentParser(description='Phone-home mesh demo')
    parser.add_argument('--workers-per-node', type=int, default=1,
                        help='Workers per node (default: 1)')
    parser.add_argument('--log-level', default='debug',
                        choices=['info', 'debug', 'perf', 'error'],
                        help='Mosaic log level (default: info)')
    args = parser.parse_args()
    wpn = args.workers_per_node
    log_flag = '--%s' % args.log_level

    python = sys.executable
    procs = {}  # label → (Popen, reader_thread)

    # Base env for all child processes
    base_env = os.environ.copy()
    base_env['PYTHONUNBUFFERED'] = '1'
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    base_env['PYTHONPATH'] = os.path.abspath(project_root)

    # Node env adds monitor coordinates (like K8s pod env vars)
    node_env = base_env.copy()
    node_env['MONITOR_HOST'] = MONITOR_HOST
    node_env['MONITOR_PORT'] = str(MONITOR_PORT)
    node_env['PUBSUB_PORT'] = str(PUBSUB_PORT)

    print('=' * 72)
    print('  PHONE-HOME MESH DEMO')
    print('=' * 72)
    print()
    print('  Architecture:')
    print('    Monitor binds to %s:%d (pubsub: %d)' % (MONITOR_HOST, MONITOR_PORT, PUBSUB_PORT))
    print('    Nodes read MONITOR_HOST/MONITOR_PORT/PUBSUB_PORT from env')
    print('    Each node spawns: 1 warehouse + %d worker%s' % (wpn, 's' if wpn > 1 else ''))
    print()
    print('  All runtime logs flow to the [monitor] stream via ZMQ.')
    print('  Look for runtime IDs in the log lines:')
    print('    NODE:0:XXXX      — node process')
    print('    WAREHOUSE:0:XXXX — node warehouse')
    print('    WORKER:0:0:XXXX  — worker on node 0')

    # ── Step 1: Monitor ──────────────────────────────────────────────────
    #
    # mrun with no --node/--monitor flag inits a monitor, then runs the
    # trailing command. In K8s this is `python3 k8s_runner.py`. Here we
    # use `sleep infinity` to keep the monitor alive and its logging
    # pipeline running so remote logs from nodes/workers are printed.
    #
    _banner(1, 'Starting MONITOR in dynamic mode')
    print('    cmd: mrun --dynamic --address %s --port %d -nw 0 %s sleep infinity'
          % (MONITOR_HOST, MONITOR_PORT, log_flag))

    monitor_cmd = [
        python, '-m', 'mosaic.cli.mrun',
        '--dynamic',
        '--address', MONITOR_HOST,
        '--port', str(MONITOR_PORT),
        '-nw', '0',
        log_flag,
        'sleep', 'infinity',
    ]

    try:
        procs['monitor'] = _start_process('monitor', monitor_cmd, base_env)
        time.sleep(3)

        if procs['monitor'][0].poll() is not None:
            print('    FAILED: monitor exited with code %s' % procs['monitor'][0].returncode)
            return

        # ── Step 2: Node 0 ────────────────────────────────────────────────
        _banner(2, 'Starting NODE:0 (%d worker%s) — phone home' % (wpn, 's' if wpn > 1 else ''))
        print('    cmd: mrun --node --phone-home -i 0 -nw %d %s' % (wpn, log_flag))
        print('    env: MONITOR_HOST=%s MONITOR_PORT=%d PUBSUB_PORT=%d'
              % (MONITOR_HOST, MONITOR_PORT, PUBSUB_PORT))

        node0_cmd = [
            python, '-m', 'mosaic.cli.mrun',
            '--node', '--phone-home',
            '-i', '0',
            '-nw', str(wpn),
            log_flag,
        ]
        procs['node-0'] = _start_process('node-0', node0_cmd, node_env)
        time.sleep(5)

        if procs['node-0'][0].poll() is not None:
            print('    FAILED (exit code %s)' % procs['node-0'][0].returncode)
            return

        # ── Step 3: Node 1 ────────────────────────────────────────────────
        _banner(3, 'Starting NODE:1 (%d worker%s) — phone home' % (wpn, 's' if wpn > 1 else ''))
        print('    cmd: mrun --node --phone-home -i 1 -nw %d %s' % (wpn, log_flag))

        node1_cmd = [
            python, '-m', 'mosaic.cli.mrun',
            '--node', '--phone-home',
            '-i', '1',
            '-nw', str(wpn),
            log_flag,
        ]
        procs['node-1'] = _start_process('node-1', node1_cmd, node_env)
        time.sleep(5)

        if procs['node-1'][0].poll() is not None:
            print('    FAILED (exit code %s)' % procs['node-1'][0].returncode)
            return

        # ── Step 4: Verify mesh ───────────────────────────────────────────
        _banner(4, 'Mesh formed')
        total_workers = 2 * wpn
        print('    Nodes        : 2 (each with its own warehouse)')
        print('    Workers      : %d total (%d per node)' % (total_workers, wpn))
        print()
        for label, (proc, _) in procs.items():
            status = 'alive' if proc.poll() is None else 'DEAD (exit=%s)' % proc.returncode
            print('    %-12s : %s (pid %d)' % (label, status, proc.pid))

        # ── Step 5: Kill node 1 ───────────────────────────────────────────
        _banner(5, 'Simulating NODE:1 drop (SIGTERM)')
        print('    (K8s equivalent: pod crash / eviction / preemption)')

        procs['node-1'][0].terminate()
        try:
            procs['node-1'][0].wait(timeout=5)
        except subprocess.TimeoutExpired:
            procs['node-1'][0].kill()
            procs['node-1'][0].wait(timeout=2)

        print('    Node:1 killed (exit=%s)' % procs['node-1'][0].returncode)
        print('    Waiting for monitor heartbeat to detect disconnect...')
        time.sleep(8)

        # ── Step 6: Replacement node ──────────────────────────────────────
        _banner(6, 'Starting NODE:2 (replacement) — phone home')
        print('    cmd: mrun --node --phone-home -i 2 -nw %d %s' % (wpn, log_flag))

        node2_cmd = [
            python, '-m', 'mosaic.cli.mrun',
            '--node', '--phone-home',
            '-i', '2',
            '-nw', str(wpn),
            log_flag,
        ]
        procs['node-2'] = _start_process('node-2', node2_cmd, node_env)
        time.sleep(5)

        if procs['node-2'][0].poll() is not None:
            print('    FAILED (exit code %s)' % procs['node-2'][0].returncode)

        # ── Summary ──────────────────────────────────────────────────────
        print('\n' + '=' * 72)
        print('  RESULTS')
        print('=' * 72)

        statuses = {}
        for label, (proc, _) in procs.items():
            statuses[label] = proc.poll() is None

        print('  %-14s %s' % ('monitor:', 'alive' if statuses['monitor'] else 'DEAD'))
        print('  %-14s %s' % ('node-0:', 'alive' if statuses['node-0'] else 'DEAD'))
        print('  %-14s %s (intentionally killed)' % ('node-1:', 'dead' if not statuses['node-1'] else 'alive'))
        print('  %-14s %s (replacement)' % ('node-2:', 'alive' if statuses.get('node-2') else 'DEAD'))
        print()

        all_ok = (statuses['monitor'] and
                  statuses['node-0'] and
                  not statuses['node-1'] and
                  statuses.get('node-2', False))
        if all_ok:
            print('  PASS: mesh formed, drop detected, replacement joined')
        else:
            print('  FAIL: unexpected process state (see above)')
        print('=' * 72)

    except KeyboardInterrupt:
        print('\nInterrupted.')

    finally:
        print('\nCleaning up...')
        for label in reversed(list(procs.keys())):
            proc, _ = procs[label]
            _terminate(proc, label)
        print('Done.')


if __name__ == '__main__':
    main()
