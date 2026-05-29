import json
import os
import pickle
import time
from io import BytesIO

import numpy as np

__all__ = ['ArtifactWarehouse']


_instance = None


def _reset():
    """For testing: clear the cached artifact warehouse instance."""
    global _instance
    _instance = None


class ArtifactWarehouse:
    """
    Artifact-store-backed warehouse for shots and gradients.

    Parameters
    ----------
    backend : ArtifactBackend
        Pre-built backend instance (e.g. ``MinioBackend`` or ``S3Backend``).
    bucket : str
        Bucket name.
    run_prefix : str, optional
        Top-level key prefix scoping all objects to a single run (e.g.
        the Argo workflow name). Defaults to ``''`` (no prefix).
    gradient_prefix : str, optional
        Key prefix for gradient objects, defaults to ``'gradients'``.
    shot_prefix : str, optional
        Key prefix for shot data objects, defaults to ``'shots'``.

    """

    def __init__(
        self, backend, bucket, run_prefix='',
        gradient_prefix='gradients', shot_prefix='shots'
    ):
        self._backend = backend
        self._bucket = bucket
        self._run_prefix = run_prefix
        self._gradient_prefix = gradient_prefix
        self._shot_prefix = shot_prefix
        self._iteration = 0

    @classmethod
    def from_env(cls, prefix='MOSAIC_ARTIFACT'):
        """
        Builds ArtifactWarehouse from environment variables.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix, defaults to ``'MOSAIC_ARTIFACT'``.

        Returns
        -------
        ArtifactWarehouse

        """
        from mosaic.utils.artifacts import ArtifactConfig, ArtifactBackend

        config = ArtifactConfig.from_env(prefix)
        backend = ArtifactBackend.from_config(config)
        return cls(
            backend=backend,
            bucket=config.bucket,
            run_prefix=os.environ.get(f'{prefix}_RUN_ID', ''),
            gradient_prefix=os.environ.get(f'{prefix}_GRADIENT_PREFIX', 'gradients'),
            shot_prefix=os.environ.get(f'{prefix}_SHOT_PREFIX', 'shots'),
        )

    @property
    def bucket(self):
        return self._bucket

    @property
    def run_prefix(self):
        return self._run_prefix

    @property
    def gradient_prefix(self):
        if self._run_prefix:
            return f'{self._run_prefix}/{self._gradient_prefix}'
        return self._gradient_prefix

    @property
    def shot_prefix(self):
        if self._run_prefix:
            return f'{self._run_prefix}/{self._shot_prefix}'
        return self._shot_prefix

    @property
    def iteration(self):
        return self._iteration

    def set_iteration(self, iteration):
        self._iteration = iteration

    def ensure_bucket(self):
        self._backend.ensure_bucket(self._bucket)

    def _upload_bytes(self, key, data):
        self._backend.put(self._bucket, key, data)

    def _download_bytes(self, key):
        return self._backend.get(self._bucket, key)

    def _key_exists(self, key):
        return self._backend.exists(self._bucket, key)

    def push_remote(self, key, data):
        """
        Upload data to the artifact store.

        Numpy arrays are stored in ``.npy`` format; all other objects are
        pickled.

        Parameters
        ----------
        key : str
            Object key within the bucket.
        data : object
            Numpy array or any picklable object.

        Returns
        -------
        str
            The key the data was stored under.

        """
        if isinstance(data, np.ndarray):
            buf = BytesIO()
            np.save(buf, data)
            raw = buf.getvalue()
        else:
            raw = pickle.dumps(data)

        self._upload_bytes(key, raw)
        return key

    def pull_remote(self, key, poll=False, poll_interval=1.0, max_interval=30.0):
        """
        Download data from the artifact store.

        If ``poll`` is True, we wait until the key is available
        with exponential backoff (capped at ``max_interval`` seconds).

        Parameters
        ----------
        key : str
            Object key within the bucket.
        poll : bool, optional
            Block until the key is available, defaults to False.
        poll_interval : float, optional
            Initial polling interval in seconds, defaults to 1.0.
        max_interval : float, optional
            Maximum interval between polls in seconds, defaults to 30.0.

        Returns
        -------
        object
            Reconstructed numpy array or unpickled Python object.

        """
        wait = poll_interval
        while True:
            try:
                raw = self._download_bytes(key)
                break
            except Exception:
                if not poll:
                    raise
                time.sleep(wait)
                wait = min(wait * 1.5, max_interval)

        if key.endswith('.npy'):
            return np.load(BytesIO(raw))
        return pickle.loads(raw)

    def write_shot_list(self, iteration, shot_ids, attempt=0):
        """
        Write the list of expected shot IDs for the ``iteration`` to remote storage.

        The gradient accumulator polls this file to keep track of expected shots.
        During iteration retries, shots.json is rewritten - the accumulator detects
        and resets the running sum accordingly.

        Parameters
        ----------
        iteration : int
            Iteration index.
        shot_ids : list of int
            Shot IDs expected (or completed) for this iteration.
        attempt : int, optional
            Retry attempt counter, defaults to 0.

        """
        key = f'{self.gradient_prefix}/iter_{iteration}/shots.json'
        payload = {'shot_ids': shot_ids, 'attempt': attempt}
        self._upload_bytes(key, json.dumps(payload).encode())

    def clear_iteration_gradients(self, iteration):
        """
        Delete all per-shot gradients for the designated ``iteration``.

        Used prior retry attempts in order to remove stale gradients so the
        accumulator doesn't fold them with recalculated gradients.

        Parameters
        ----------
        iteration : int

        """
        prefix = f'{self.gradient_prefix}/iter_{iteration}/'
        for key in self._backend.list_keys(self._bucket, prefix):
            if key.endswith('.pkl'):
                self._backend.delete(self._bucket, key)

    async def exec_remote(self, uid, func, func_args=None, func_kwargs=None):
        """
        Run *func* to produce a per-shot gradient and upload it to the
        artifact store.

        Mirrors :meth:`Warehouse.exec_remote`, but uploads to
        ``{gradient_prefix}/iter_{N}/shot_{shot_id}.pkl`` (plus
        ``_prec.pkl`` for preconditioners).

        Parameters
        ----------
        uid
            Identifier of the variable being accumulated (e.g. ``"vp"``);
            accepted for interface compatibility but not used in the key.
        func : callable
            Async redux closure with signature
            ``func(rec_grads, *args, **kwargs)`` returning the per-shot
            gradient.
        func_args : tuple, optional
            Positional arguments to pass after ``rec_grads``.
        func_kwargs : dict, optional
            Keyword arguments to pass to ``func``. Two special keys are
            popped before forwarding:

            - ``iteration`` — overrides ``self.iteration`` for key construction.
            - ``shot_id`` — required; identifies the shot this gradient is for.

        Returns
        -------
        str
            S3 key the gradient was uploaded under.

        Raises
        ------
        ValueError
            If ``shot_id`` is not provided in ``func_kwargs``.

        """
        func_args = func_args or ()
        func_kwargs = dict(func_kwargs) if func_kwargs else {}
        iteration = func_kwargs.pop('iteration', self._iteration)
        shot_id = func_kwargs.pop('shot_id', None)

        if shot_id is None:
            raise ValueError('exec_remote requires shot_id in func_kwargs')

        # Run the redux closure
        result = await func(None, *func_args, **func_kwargs)

        # Unwrap result
        grad = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(grad, tuple):
            grad = grad[0]

        key = f'{self.gradient_prefix}/iter_{iteration}/shot_{shot_id}.pkl'

        # Upload
        if hasattr(grad, 'data'):
            self._upload_bytes(key, pickle.dumps(np.asarray(grad.data)))
            if hasattr(grad, 'prec') and grad.prec is not None:
                prec_key = key.replace('.pkl', '_prec.pkl')
                self._upload_bytes(prec_key, pickle.dumps(np.asarray(grad.prec.data)))
        elif grad is not None:
            self._upload_bytes(key, pickle.dumps(grad))

        return key
    

def artifact_warehouse():
    """
    Return the runtime/process-wide artifact warehouse if configured.

    On first call, constructs from env vars if ``MOSAIC_ARTIFACT_ENDPOINT``
    is set, otherwise returns ``None``. Cached for subsequent calls.

    Returns
    -------
    ArtifactWarehouse or None

    """
    global _instance
    if _instance is None and os.environ.get('MOSAIC_ARTIFACT_ENDPOINT'):
        try:
            _instance = ArtifactWarehouse.from_env()
        except Exception as e:
            import warnings
            warnings.warn(f'Failed to initialise artifact warehouse: {e}')
    return _instance