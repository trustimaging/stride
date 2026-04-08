
from __future__ import annotations

import os
import json
import time
import pickle
import logging

import numpy as np

from .artifact_warehouse import ArtifactWarehouse


__all__ = ['GradientAccumulator']

logger = logging.getLogger(__name__)


class GradientAccumulator:
    """
    Polls MinIO for per-shot gradient files, sums them, and writes
    ``{gradient_prefix}/iter_{I}/final.pkl`` for each iteration.

    At the start of each iteration the head writes a ``shots.json`` file
    listing the expected shot IDs.  The accumulator polls that file so it
    knows exactly how many gradients to wait for.  If workers drop mid-
    iteration and the head continues below the fault-tolerance threshold,
    the head overwrites ``shots.json`` with only the completed shot IDs;
    the accumulator detects the update and finishes immediately without a
    blind timeout.

    As each ``shot_{id}.pkl`` file arrives it is downloaded and folded into
    a running sum, so at most two full gradient arrays are in memory at once.

    Parameters
    ----------
    warehouse : ArtifactWarehouse
        Configured warehouse instance used for all I/O.
    num_iters : int
        Total number of iterations to process before exiting.

    """

    def __init__(
        self,
        warehouse: ArtifactWarehouse,
        num_iters: int,
    ) -> None:
        self._warehouse = warehouse
        self._num_iters = num_iters

    @classmethod
    def from_env(cls) -> 'GradientAccumulator':
        """
        Create a ``GradientAccumulator`` from environment variables.

        Reads the same ``ARTIFACT_*`` variables as
        ``ArtifactWarehouse.from_env()``, plus:

        ``NUM_ITERS``
            Total number of iterations to process (required).

        Returns
        -------
        GradientAccumulator

        """
        warehouse = ArtifactWarehouse.from_env()
        warehouse.ensure_bucket()
        num_iters = int(os.environ['NUM_ITERS'])
        return cls(warehouse, num_iters)

    # ------------------------------------------------------------------ helpers

    def _poll_json(self, key: str):
        """Block until *key* exists in S3 and return its parsed JSON content."""
        wait = 1.0
        while True:
            try:
                return json.loads(self._warehouse._download_bytes(key))
            except Exception:
                time.sleep(wait)
                wait = min(wait * 1.5, 10.0)

    def _read_json(self, key: str):
        """Return parsed JSON for *key*, or ``None`` if not found."""
        try:
            return json.loads(self._warehouse._download_bytes(key))
        except Exception:
            return None

    # ------------------------------------------------------------------ main

    def accumulate_iteration(self, iteration: int) -> None:
        """
        Poll for shot gradient files for *iteration*, sum them, and write
        ``final.pkl``.

        The head writes ``shots.json`` before dispatching shots.  This method
        waits for that file, then polls for each listed ``shot_{id}.pkl``.
        If the head shrinks ``shots.json`` mid-iteration (worker drop, below
        threshold), the expected set is updated and accumulation finishes
        as soon as all remaining shots are present.

        Parameters
        ----------
        iteration : int
            Zero-based iteration index.

        """
        client  = self._warehouse.client
        bucket  = self._warehouse.bucket
        gprefix = self._warehouse.gradient_prefix

        shots_key = '%s/iter_%d/shots.json' % (gprefix, iteration)
        prefix    = '%s/iter_%d/' % (gprefix, iteration)

        logger.info('Iter %d — waiting for shots.json.', iteration)
        shot_ids = self._poll_json(shots_key)
        expected = {('%s/iter_%d/shot_%d.pkl' % (gprefix, iteration, s))
                    for s in shot_ids}
        logger.info('Iter %d — expecting %d shot(s): %s.',
                    iteration, len(expected), shot_ids)

        accumulated = None
        folded: set = set()
        iter_start  = time.time()
        wait        = 1.0

        while folded < expected:
            # Re-read shots.json — head may have shrunk it on a partial iteration
            updated = self._read_json(shots_key)
            if updated is not None:
                new_expected = {('%s/iter_%d/shot_%d.pkl' % (gprefix, iteration, s))
                                for s in updated}
                if new_expected != expected:
                    logger.info(
                        'Iter %d — shots.json updated (%d → %d shot(s)).',
                        iteration, len(expected), len(new_expected),
                    )
                    expected = new_expected

            existing = {
                obj.object_name
                for obj in client.list_objects(bucket, prefix=prefix)
            }
            newly_available = existing & expected - folded

            for key in sorted(newly_available):
                shot_id = key.rsplit('_', 1)[-1].replace('.pkl', '')
                logger.info(
                    'Iter %d — downloading shot_%s.pkl (%d/%d).',
                    iteration, shot_id, len(folded) + 1, len(expected),
                )
                arr = pickle.loads(self._warehouse._download_bytes(key))
                accumulated = (np.array(arr, dtype=float) if accumulated is None
                               else accumulated + arr)
                folded.add(key)
                logger.info(
                    'Iter %d — folded shot_%s (%d/%d accumulated).',
                    iteration, shot_id, len(folded), len(expected),
                )

            if folded < expected:
                time.sleep(wait)
                wait = min(wait * 1.5, 30.0)

        final_key = '%s/iter_%d/final.pkl' % (gprefix, iteration)
        self._warehouse._upload_bytes(final_key, pickle.dumps(accumulated))

        elapsed = time.time() - iter_start
        logger.info('Iter %d done — final.pkl written (%d/%d shots, %.2fs).',
                    iteration, len(folded), len(expected), elapsed)

    def run(self) -> None:
        """Loop over all iterations sequentially."""
        logger.info(
            'Started — %d iteration(s), prefix=%r.',
            self._num_iters, self._warehouse.gradient_prefix,
        )
        for i in range(self._num_iters):
            self.accumulate_iteration(i)
        logger.info('All %d iteration(s) complete. Exiting.', self._num_iters)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(name)s %(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    GradientAccumulator.from_env().run()
