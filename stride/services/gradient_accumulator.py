import os
import json
import time
import pickle
import logging

from mosaic.runtime.artifact_warehouse import ArtifactWarehouse


__all__ = ['GradientAccumulator']

logger = logging.getLogger(__name__)


class GradientAccumulator:
    """
    Polls the artifact store for per-shot gradient files, sums them, and
    writes ``{gradient_prefix}/iter_{N}/final.pkl`` for each iteration.

    At the start of each iteration the head writes a ``shots.json`` file
    listing the expected shot IDs. The accumulator polls that file, then
    polls for each listed ``shot_{S}.pkl`` and folds it into a running sum
    as it arrives.

    Parameters
    ----------
    warehouse : ArtifactWarehouse
        Configured warehouse instance used for all I/O.
    num_iters : int
        Total number of iterations to process before exiting.

    """

    def __init__(self, warehouse, num_iters):
        self._warehouse = warehouse
        self._num_iters = num_iters

    @classmethod
    def from_env(cls):
        """
        Build a ``GradientAccumulator`` from environment variables.

        Reads the same ``MOSAIC_ARTIFACT_*`` variables as
        :meth:`ArtifactWarehouse.from_env`, plus:

        - ``NUM_ITERS`` — total number of iterations to process (required).

        Returns
        -------
        GradientAccumulator

        """
        warehouse = ArtifactWarehouse.from_env()
        warehouse.ensure_bucket()
        num_iters = int(os.environ['NUM_ITERS'])
        return cls(warehouse, num_iters)

    def _poll_json(self, key):
        """Block until key exists, then return its parsed JSON content."""
        wait = 1.
        while True:
            try:
                return json.loads(self._warehouse._download_bytes(key))
            except Exception:
                time.sleep(wait)
                wait = min(wait * 1.5, 10.)

    def accumulate_iteration(self, iteration):
        """
        Wait for ``shots.json``, poll for per-shot gradient files,
        stream-sum them and write ``final.pkl``.

        Parameters
        ----------
        iteration : int
            Zero-based iteration index.

        """
        bucket = self._warehouse.bucket
        gradient_prefix = self._warehouse.gradient_prefix

        prefix = f'{gradient_prefix}/iter_{iteration}'
        shots_key = f'{prefix}/shots.json'

        logger.info(f'Iter {iteration} - waiting for shots.json')
        raw = self._poll_json(shots_key)
        shot_ids = raw['shot_ids']
        expected = {f'{prefix}/shot_{s}.pkl' for s in shot_ids}
        logger.info(f'Iter {iteration} - expecting {len(expected)} shot(s)')

        accumulated = None
        accumulated_prec = None
        folded = set()
        wait = 1.

        while folded < expected:
            existing = set(self._warehouse._backend.list_keys(bucket, prefix))
            newly_available = (existing & expected) - folded

            for key in sorted(newly_available):
                arr = pickle.loads(self._warehouse._download_bytes(key))
                accumulated = arr.copy() if accumulated is None else accumulated + arr

                prec_key = key.replace('.pkl', '_prec.pkl')
                try:
                    prec_arr = pickle.loads(self._warehouse._download_bytes(prec_key))
                    accumulated_prec = (prec_arr.copy() if accumulated_prec is None else accumulated_prec + prec_arr)
                except Exception:
                    pass

                folded.add(key)
                logger.info(f"Iter {iteration} - folded {key} ({len(folded)}/{len(expected)})")

            if folded < expected:
                time.sleep(wait)
                wait = min(wait * 1.5, 30.)

        final_key = f'{prefix}/final.pkl'
        self._warehouse._upload_bytes(final_key, pickle.dumps(accumulated))
        logger.info(f'Iter {iteration} - final.pkl written.')

        if accumulated_prec is not None:
            final_prec_key = f'{prefix}/final_prec.pkl'
            self._warehouse._upload_bytes(final_prec_key, pickle.dumps(accumulated_prec))
            logger.info(f'Iter {iteration} - final_prec.pkl written.')

    def run(self):
        """Loop over all iteration(s) sequentially."""
        logger.info(f'Started - {self._num_iters} iteration(s)')
        for i in range(self._num_iters):
            self.accumulate_iteration(i)
        logger.info(f'All {self._num_iters} iteration(s) complete. Exiting.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(name)s %(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    GradientAccumulator.from_env().run()
