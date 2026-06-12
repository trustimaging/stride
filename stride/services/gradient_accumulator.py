
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
    Polls the artifact store for per-task gradient files, sums them, and
    writes ``{result_prefix}/counter_{N}/final_grad.pkl`` for each counter.

    At the start of each counter the head writes a ``tasks.json`` file
    listing the expected task IDs. The accumulator polls that file, then
    polls for each listed ``task_{S}_grad.pkl`` and folds it into a running
    sum as it arrives.

    Parameters
    ----------
    artifact_warehouse : ArtifactWarehouse
        Configured artifact warehouse instance used for all I/O.
    num_iters : int
        Total number of counters to process before exiting.

    """

    def __init__(self, artifact_warehouse, num_iters):
        self._artifact_warehouse = artifact_warehouse
        self._num_iters = num_iters
        self._tasks_poll_interval = 1.0
        self._shots_poll_interval = 1.0

    @classmethod
    def from_env(cls):
        """
        Build a ``GradientAccumulator`` from environment variables.

        Reads the same ``MOSAIC_ARTIFACT_*`` variables as
        :meth:`ArtifactWarehouse.from_env`, plus:

        - ``STRIDE_NUM_ITERS`` — total number of counters to process (required).

        Returns
        -------
        GradientAccumulator

        """
        artifact_warehouse = ArtifactWarehouse.from_env()
        artifact_warehouse.ensure_bucket()
        num_iters = int(os.environ['STRIDE_NUM_ITERS'])
        return cls(artifact_warehouse, num_iters)

    def _poll_json(self, key):
        """Block until key exists, then return its parsed JSON content."""
        while True:
            try:
                return json.loads(self._artifact_warehouse._download_bytes(key))
            except Exception:
                time.sleep(self._tasks_poll_interval)

    def accumulate_counter(self, counter):
        """
        Wait for ``tasks.json``, poll for per-task gradient files,
        stream-sum them and write ``final_grad.pkl``.

        Per-task uploads are fully-pickled gradient objects (data + any
        prec sub-object). Summation uses the gradient's own ``__iadd__``
        so the prec is folded alongside the data automatically.

        Parameters
        ----------
        counter : int
            Zero-based counter index.

        """
        bucket = self._artifact_warehouse.bucket
        result_prefix = self._artifact_warehouse.result_prefix

        prefix = f'{result_prefix}/counter_{counter}'
        tasks_key = f'{prefix}/tasks.json'

        logger.info(f'Counter {counter} - waiting for tasks.json')
        raw = self._poll_json(tasks_key)
        task_ids = raw['task_ids']
        expected = {f'{prefix}/task_{s}_grad.pkl' for s in task_ids}
        logger.info(f'Counter {counter} - expecting {len(expected)} task(s)')

        accumulated = None
        folded = set()

        while folded < expected:
            existing = set(self._artifact_warehouse._backend.list_keys(bucket, prefix))
            newly_available = (existing & expected) - folded

            for key in sorted(newly_available):
                obj = pickle.loads(self._artifact_warehouse._download_bytes(key))
                if accumulated is None:
                    accumulated = obj
                else:
                    accumulated += obj
                folded.add(key)
                logger.info(f"Counter {counter} - folded {key} ({len(folded)}/{len(expected)})")

            if folded < expected:
                time.sleep(self._shots_poll_interval)

        final_key = f'{prefix}/final_grad.pkl'
        self._artifact_warehouse._upload_bytes(final_key, pickle.dumps(accumulated))
        logger.info(f'Counter {counter} - final_grad.pkl written.')

        # Per-task gradients are now folded into final_grad.pkl; delete them
        # so the bucket doesn't accumulate ~N MB of dead weight per counter.
        for key in folded:
            try:
                self._artifact_warehouse._backend.delete(self._artifact_warehouse._bucket, key)
            except Exception:
                pass
        logger.info(f'Counter {counter} - cleaned up {len(folded)} per-task gradient files.')

    def run(self):
        """Loop over all counters sequentially."""
        logger.info(f'Started - {self._num_iters} counter(s)')
        for i in range(self._num_iters):
            self.accumulate_counter(i)
        logger.info(f'All {self._num_iters} counter(s) complete. Exiting.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(name)s %(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    GradientAccumulator.from_env().run()
