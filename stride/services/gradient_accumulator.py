
import os
import json
import time
import pickle
import logging
from dataclasses import dataclass, field

from mosaic.runtime.artifact_warehouse import ArtifactWarehouse


__all__ = ['GradientAccumulator']

logger = logging.getLogger(__name__)


def _shot_id_from_key(key):
    """Extract numeric shot id from '..../task_<N>_grad.pkl' for sort ordering."""
    return int(key.rsplit('/', 1)[-1].split('_')[1])


@dataclass
class _CounterState:
    """Per-counter accumulation state mutated by the poll loop."""
    attempt: int
    expected: set
    accumulated: object = None
    folded: set = field(default_factory=set)


class GradientAccumulator:
    """
    Polls the artifact store for per-task gradient files, sums them, and
    writes ``{result_prefix}/counter_{N}/final_grad.pkl`` for each counter.

    At the start of each counter the head writes a ``tasks.json`` file
    listing the expected task IDs. The accumulator polls that file, then
    polls for each listed ``task_{S}_grad.pkl`` and folds it into a running
    sum as it arrives.

    If the head shrinks ``tasks.json`` mid-counter (partial-accept after a
    sub-threshold worker drop) or bumps its ``attempt`` field (retry
    rollback), the accumulator detects the change on its next poll and
    updates its expected set or resets accumulation accordingly.

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

    def _refresh_tasks(self, counter, tasks_key, prefix, state):
        """
        Re-read ``tasks.json`` and react to head-side updates.

        Mutates *state* in place:

        - If ``attempt`` bumped (head rolled back and re-dispatched), reset
          ``accumulated`` and ``folded`` so we start over for this counter.
        - If the task list shrunk (partial-accept) or otherwise changed,
          update ``expected``.

        Silently no-op if ``tasks.json`` is momentarily unreadable.

        Parameters
        ----------
        counter : int
            Zero-based counter index (used for log lines).
        tasks_key : str
            Artifact-store key for the counter's ``tasks.json``.
        prefix : str
            Counter prefix used to build per-task gradient keys.
        state : _CounterState
            Loop state to mutate in place.

        """
        try:
            updated = json.loads(
                self._artifact_warehouse._download_bytes(tasks_key)
            )
        except Exception:
            return

        new_attempt = updated.get('attempt', 0)
        if new_attempt != state.attempt:
            logger.info(
                f'Counter {counter} - attempt changed '
                f'({state.attempt} -> {new_attempt}), resetting accumulation.'
            )
            state.attempt = new_attempt
            state.accumulated = None
            state.folded = set()

        new_expected = {f'{prefix}/task_{s}_grad.pkl'
                        for s in updated['task_ids']}
        if new_expected != state.expected:
            logger.info(
                f'Counter {counter} - tasks.json changed '
                f'({len(state.expected)} -> {len(new_expected)} task(s)).'
            )
            state.expected = new_expected

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

        logger.debug(f'Counter {counter} - waiting for tasks.json')
        raw = self._poll_json(tasks_key)
        task_ids = raw['task_ids']

        state = _CounterState(
            attempt=raw.get('attempt', 0),
            expected={f'{prefix}/task_{s}_grad.pkl' for s in task_ids},
        )
        logger.debug(
            f'Counter {counter} - expecting {len(state.expected)} task(s) '
            f'(attempt {state.attempt})'
        )

        last_heartbeat = time.time()
        while state.folded < state.expected:
            self._refresh_tasks(counter, tasks_key, prefix, state)

            existing = set(
                self._artifact_warehouse._backend.list_keys(bucket, prefix)
            )
            newly_available = (existing & state.expected) - state.folded

            logger.debug(
                f'Counter {counter} - poll: '
                f'folded={len(state.folded)}/{len(state.expected)}, '
                f'attempt={state.attempt}, '
                f'existing_keys={len(existing)}, '
                f'newly_available={len(newly_available)}'
            )

            for key in sorted(newly_available, key=_shot_id_from_key):
                logger.debug(f'Counter {counter} - downloading {key}')
                try:
                    obj = pickle.loads(
                        self._artifact_warehouse._download_bytes(key)
                    )
                except Exception as exc:
                    # Key deleted by rollback between list_keys and
                    # download. Next _refresh_tasks call resets state
                    # to the new attempt.
                    logger.warning(
                        f'Counter {counter} - download failed for {key} '
                        f'({type(exc).__name__}: {exc}) — skipping'
                    )
                    continue
                if state.accumulated is None:
                    state.accumulated = obj
                else:
                    state.accumulated += obj
                state.folded.add(key)
                logger.debug(
                    f"Counter {counter} - folded {key} "
                    f"({len(state.folded)}/{len(state.expected)})"
                )

            if state.folded < state.expected:
                # Heartbeat every 10s so we can see the accumulator is alive
                # even when nothing new is arriving.
                now = time.time()
                if now - last_heartbeat > 10.0:
                    logger.info(
                        f'Counter {counter} - heartbeat: still waiting '
                        f'({len(state.folded)}/{len(state.expected)} folded, '
                        f'attempt={state.attempt})'
                    )
                    last_heartbeat = now
                time.sleep(self._shots_poll_interval)

        logger.info(
            f'Counter {counter} - fold complete '
            f'({len(state.folded)}/{len(state.expected)}), '
            f'writing final_grad.pkl'
        )

        final_key = f'{prefix}/final_grad.pkl'
        payload = pickle.dumps(state.accumulated)
        self._artifact_warehouse._upload_bytes(final_key, payload)
        logger.info(
            f'Counter {counter} - final_grad.pkl written '
            f'({len(payload)} bytes) to {final_key}'
        )

        # Per-task gradients are now folded into final_grad.pkl; delete them
        # so the bucket doesn't accumulate ~N MB of dead weight per counter.
        deleted = 0
        for key in state.folded:
            try:
                self._artifact_warehouse._backend.delete(
                    self._artifact_warehouse._bucket, key
                )
                deleted += 1
            except Exception as exc:
                logger.warning(
                    f'Counter {counter} - delete failed for {key} '
                    f'({type(exc).__name__}: {exc})'
                )
        logger.info(
            f'Counter {counter} - cleaned up {deleted}/{len(state.folded)} '
            f'per-task gradient files.'
        )

    def run(self):
        """Loop over all counters sequentially."""
        logger.info(
            f'Started - {self._num_iters} counter(s), '
            f'bucket={self._artifact_warehouse.bucket}, '
            f'result_prefix={self._artifact_warehouse.result_prefix}'
        )
        for i in range(self._num_iters):
            logger.info(f'==== Starting counter {i} ====')
            self.accumulate_counter(i)
            logger.info(f'==== Counter {i} done ====')
        logger.info(f'All {self._num_iters} counter(s) complete. Exiting.')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(name)s %(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    GradientAccumulator.from_env().run()
