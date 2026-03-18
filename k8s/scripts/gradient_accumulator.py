#!/usr/bin/env python3
"""
Gradient accumulator service.

Watches the MinIO gradient bucket for per-worker gradient files.  As each
worker_{K}.pkl file arrives it is downloaded immediately and folded into a
running sum, so at most two full gradient arrays are in memory at once
(the running total plus the newly-arrived file).  Once all NUM_WORKERS files
have been folded in, the final sum is written as
``{gradient_prefix}/iter_{I}/final.pkl``.

Runs continuously — processing one iteration at a time — until the process is
killed (e.g. by Argo when the head exits).

Environment variables
---------------------
ARTIFACT_ENDPOINT         MinIO/S3 endpoint, e.g. "minio.argo.svc.cluster.local:9000"
ARTIFACT_ACCESS_KEY       Access key (default: "minioadmin")
ARTIFACT_SECRET_KEY       Secret key (default: "minioadmin")
ARTIFACT_BUCKET           Bucket name (default: "stride-data")
ARTIFACT_SECURE           "true"/"false" (default: "false")
ARTIFACT_GRADIENT_PREFIX  Key prefix for gradient objects (default: "gradients")
NUM_WORKERS               Total number of workers to wait for per iteration (required)
"""

import os
import time
import pickle
import datetime


def _ts() -> str:
    return datetime.datetime.now().strftime('%H:%M:%S')


def _log(msg: str) -> None:
    print('[accumulator %s] %s' % (_ts(), msg), flush=True)


def main() -> None:
    num_workers = int(os.environ['NUM_WORKERS'])
    num_iters = int(os.environ['NUM_ITERS'])

    from mosaic.runtime.artifact_warehouse import ArtifactWarehouse
    warehouse = ArtifactWarehouse.from_env()
    warehouse.ensure_bucket()
    client = warehouse.client
    bucket = warehouse.bucket
    gradient_prefix = warehouse.gradient_prefix

    _log('Started — %d iteration(s), %d worker(s) per iteration, prefix=%r.'
         % (num_iters, num_workers, gradient_prefix))

    import numpy as np

    for iteration in range(num_iters):
        prefix = '%s/iter_%d/' % (gradient_prefix, iteration)
        expected_keys = {
            '%s/iter_%d/worker_%d.pkl' % (gradient_prefix, iteration, k)
            for k in range(num_workers)
        }

        _log('Iter %d — waiting for %d file(s).' % (iteration, num_workers))
        iter_start = time.time()

        accumulated = None      # running sum; None until first file arrives
        folded = set()          # keys already folded into the sum
        wait = 1.0

        # Poll until every worker file has been folded into the running sum.
        while folded < expected_keys:
            existing = {
                obj.object_name
                for obj in client.list_objects(bucket, prefix=prefix)
            }
            newly_available = existing & expected_keys - folded

            for key in sorted(newly_available):
                worker_id = key.rsplit('_', 1)[-1].replace('.pkl', '')
                _log('Iter %d — downloading worker_%s.pkl (%d/%d).'
                     % (iteration, worker_id, len(folded) + 1, num_workers))

                raw = warehouse._download_bytes(key)
                arr = pickle.loads(raw)

                # Fold into running sum — only two arrays in memory at once.
                if accumulated is None:
                    accumulated = np.array(arr, dtype=float)
                else:
                    accumulated += arr

                folded.add(key)
                _log('Iter %d — folded worker_%s (%d/%d accumulated).'
                     % (iteration, worker_id, len(folded), num_workers))

            if folded < expected_keys:
                missing = len(expected_keys) - len(folded)
                _log('Iter %d — %d file(s) not yet available, retrying in %.1fs.'
                     % (iteration, missing, wait))
                time.sleep(wait)
                wait = min(wait * 1.5, 30.0)

        final_key = '%s/iter_%d/final.pkl' % (gradient_prefix, iteration)
        warehouse._upload_bytes(final_key, pickle.dumps(accumulated))

        elapsed = time.time() - iter_start
        _log('Iter %d done — final.pkl written (%.2fs).' % (iteration, elapsed))

    _log('All %d iteration(s) complete. Exiting.' % num_iters)


if __name__ == '__main__':
    main()
