#!/usr/bin/env python
"""
K8s head runner — waits for workers, then runs a stride script.

Launched by mrun inside the head pod:
    python -m mosaic.cli.mrun --dynamic --address 0.0.0.0 --port 3000 \
        -nw 0 python k8s_runner.py

Set RUN_MODE env var to choose the script:
    forward      — scripts.simple_forward  (default)
    inverse      — scripts.simple_inverse
    inverse_s3   — scripts.simple_inverse_s3

Set EXP_NAME env var to choose the experiment directory:
    simple       — exps/simple  (default)
"""

import os
import sys

# After restructuring, scripts live in k8s/scripts/ rather than a top-level
# scripts/ dir. When this file is copied to /app/stride/k8s_runner.py and run
# from WORKDIR /app/stride/, add /app/stride/k8s/ to sys.path so that
# `scripts.simple_forward` etc. still resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k8s'))

TOTAL_WORKERS = (int(os.environ.get('NUM_WORKERS', '2'))
                 * int(os.environ.get('WORKERS_PER_NODE', '1')))
TIMEOUT = int(os.environ.get('WORKER_TIMEOUT', '300'))
RUN_MODE = os.environ.get('RUN_MODE', 'forward')
EXP_NAME = os.environ.get('EXP_NAME', 'simple')

SCRIPTS = {
    'forward': 'scripts.simple_forward',
    'inverse': 'scripts.simple_inverse',
    'inverse_artifacts': 'scripts.simple_inverse_artifacts',
    'disconnect_test': 'scripts.disconnect_test',
}

import mosaic


async def main(runtime):
    print(f'Waiting for {TOTAL_WORKERS} workers (timeout={TIMEOUT}s)...')
    await runtime.wait_for_workers(TOTAL_WORKERS, timeout=TIMEOUT)
    print(f'Workers connected: {len(runtime.workers)}')

    if RUN_MODE not in SCRIPTS:
        print(f'Unknown RUN_MODE={RUN_MODE!r}, expected one of: {", ".join(SCRIPTS)}')
        sys.exit(1)

    module_path = SCRIPTS[RUN_MODE]
    print(f'=== Running {RUN_MODE} ({module_path}) exp={EXP_NAME} ===')

    from importlib import import_module
    script = import_module(module_path)
    await script.main(runtime, exp_name=EXP_NAME)

    print(f'=== {RUN_MODE} Complete ===')


if __name__ == '__main__':
    # Pass pod IP so the HEAD runtime advertises a routable address
    # (otherwise it uses the pod hostname, which workers can't resolve)
    pod_ip = os.environ.get('POD_IP')
    if pod_ip:
        mosaic.run(main, address=pod_ip)
    else:
        mosaic.run(main)