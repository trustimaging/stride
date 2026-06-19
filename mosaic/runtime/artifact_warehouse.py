
import json
import os
import pickle
import time
from io import BytesIO
import numpy as np

from mosaic.utils.artifacts import ArtifactConfig, ArtifactBackend


__all__ = ['ArtifactWarehouse']


class ArtifactWarehouse:
    """
    Artifact-store-backed warehouse for per-task data and per-task results.

    Parameters
    ----------
    backend : ArtifactBackend
        Pre-built backend instance (e.g. ``MinioBackend`` or ``S3Backend``).
    bucket : str
        Bucket name.
    run_prefix : str, optional
        Top-level key prefix scoping all objects to a single run (e.g.
        the Argo workflow name). Defaults to ``''`` (no prefix).
    result_prefix : str, optional
        Key prefix for accumulated result objects, defaults to ``'results'``.
    task_prefix : str, optional
        Key prefix for per-task data objects, defaults to ``'tasks'``.

    """

    def __init__(
        self, backend, bucket, run_prefix='',
        result_prefix='results', task_prefix='tasks',
    ):
        self._backend = backend
        self._bucket = bucket
        self._run_prefix = run_prefix
        self._result_prefix = result_prefix
        self._task_prefix = task_prefix
        self._counter = 0

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
        config = ArtifactConfig.from_env(prefix)
        backend = ArtifactBackend.from_config(config)
        return cls(
            backend=backend,
            bucket=config.bucket,
            run_prefix=os.environ.get(f'{prefix}_RUN_ID', ''),
            result_prefix=os.environ.get(f'{prefix}_RESULT_PREFIX', 'results'),
            task_prefix=os.environ.get(f'{prefix}_TASK_PREFIX', 'tasks'),
        )

    @property
    def bucket(self):
        return self._bucket

    @property
    def run_prefix(self):
        return self._run_prefix

    @property
    def result_prefix(self):
        if self._run_prefix:
            return f'{self._run_prefix}/{self._result_prefix}'
        return self._result_prefix

    @property
    def task_prefix(self):
        if self._run_prefix:
            return f'{self._run_prefix}/{self._task_prefix}'
        return self._task_prefix

    @property
    def counter(self):
        return self._counter

    def set_counter(self, counter):
        self._counter = counter

    def ensure_bucket(self):
        self._backend.ensure_bucket(self._bucket)

    def _upload_bytes(self, key, data):
        self._backend.put(self._bucket, key, data)

    def _download_bytes(self, key):
        return self._backend.get(self._bucket, key)

    def _key_exists(self, key):
        return self._backend.exists(self._bucket, key)

    @staticmethod
    def get_fs():
        """
        Return an ``s3fs`` filesystem for direct byte-range access to the
        bucket. Callers wrap this in their own file-opening logic

        Returns
        -------
        s3fs.S3FileSystem

        """
        import s3fs
        prefix = 'MOSAIC_ARTIFACT'
        endpoint = os.environ[f'{prefix}_ENDPOINT']
        secure = os.environ.get(f'{prefix}_SECURE', 'false').lower() == 'true'
        scheme = 'https' if secure else 'http'
        return s3fs.S3FileSystem(
            endpoint_url=f'{scheme}://{endpoint}',
            key=os.environ.get(f'{prefix}_ACCESS_KEY', 'minioadmin'),
            secret=os.environ.get(f'{prefix}_SECRET_KEY', 'minioadmin'),
            config_kwargs={'signature_version': 's3v4'},
            skip_instance_cache=True,
        )

    def upload_file(self, local_path, key):
        """
        Upload a local file to ``key`` in the bucket.

        Parameters
        ----------
        local_path : str
            Path to the local HDF5 file to upload.
        key : str
            Object key within the bucket.

        """
        with open(local_path, 'rb') as f:
            self._upload_bytes(key, f.read())

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

    def pull_remote(self, key=None, *, uid=None, attr=None, reply=True,
                    poll=False, poll_interval=1.0, max_interval=30.0):
        """
        Two modes:

        - ``pull_remote(key, poll=...)`` — download a single key directly,
          returning the deserialised payload.
        - ``pull_remote(uid=..., attr=...)`` — mirror of
          :meth:`mosaic.runtime.warehouse.Warehouse.pull_remote`. Returns
          ``{attr: value}`` where ``value`` is the unpickled object stored
          at ``{result_prefix}/counter_{counter}/final_{attr}.pkl``.

        Parameters
        ----------
        key : str, optional
            Object key within the bucket (single-key mode).
        uid : optional
            Variable UID (accepted for interface parity with the local
            warehouse; not used in key construction).
        attr : str, optional
            Attribute to pull (mirror mode).
        reply : bool, optional
            Accepted for interface parity with the local warehouse.
        poll : bool, optional
            Block until the key is available, defaults to False.
        poll_interval : float, optional
            Initial polling interval in seconds, defaults to 1.0.
        max_interval : float, optional
            Maximum interval between polls in seconds, defaults to 30.0.

        Returns
        -------
        object or dict
            Single-key mode: a numpy array or unpickled Python object.
            Mirror mode: ``{attr: value}`` dict.

        """
        if attr is not None:
            key = f'{self.result_prefix}/counter_{self.counter}/final_{attr}.pkl'
            # Mirror mode polls by default — the accumulator daemon may not
            # have folded the gradient yet when the head pulls.
            poll = True

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

        if attr is not None:
            return {attr: pickle.loads(raw)}

        if key.endswith('.npy'):
            return np.load(BytesIO(raw))
        return pickle.loads(raw)

    def write_task_list(self, counter, task_ids, attempt=0):
        """
        Write the list of expected task IDs for the given ``counter`` to
        remote storage.

        The accumulator polls this file to keep track of expected tasks.
        During retries, tasks.json is rewritten - the accumulator detects
        and resets the running sum accordingly.

        Parameters
        ----------
        counter : int
            Counter index.
        task_ids : list of int
            Task IDs expected (or completed) for this counter.
        attempt : int, optional
            Retry attempt counter, defaults to 0.

        """
        key = f'{self.result_prefix}/counter_{counter}/tasks.json'
        payload = {'task_ids': task_ids, 'attempt': attempt}
        self._upload_bytes(key, json.dumps(payload).encode())

    def clear_results(self, counter):
        """
        Delete all per-task result objects for the designated ``counter``.

        Used prior to retry attempts so the accumulator doesn't fold stale
        results with recalculated ones.

        Parameters
        ----------
        counter : int

        """
        prefix = f'{self.result_prefix}/counter_{counter}/'
        for key in self._backend.list_keys(self._bucket, prefix):
            if key.endswith('.pkl'):
                self._backend.delete(self._bucket, key)

    async def exec_remote(self, uid, func, func_args=None, func_kwargs=None,
                          serialise=None):
        """
        Run *func* and upload the result to the artifact store.

        The primary payload lands at
        ``{result_prefix}/counter_{N}/task_{task_id}.pkl``; additional
        suffixed payloads land at ``{primary}_{suffix}.pkl``.

        Parameters
        ----------
        uid
            Identifier of the variable being accumulated (e.g. ``"vp"``);
            accepted for interface compatibility but not used in the key.
        func : callable
            Async redux closure with signature
            ``func(rec_grads, *args, **kwargs)`` returning the per-task
            result.
        func_args : tuple, optional
            Positional arguments to pass after ``rec_grads``.
        func_kwargs : dict, optional
            Keyword arguments to pass to ``func``. Two special keys are
            popped before forwarding:

            - ``mosaic_counter`` — overrides ``self.counter`` for key construction.
            - ``mosaic_task_id`` — required; identifies the task this result is for.
        serialise : callable, optional
            Called as ``serialise(result)`` after *func* returns. Must
            return a ``dict[str, bytes]`` mapping suffix → payload. The
            empty-string suffix is the primary key; other suffixes become
            ``_{suffix}`` before ``.pkl``. Defaults to
            ``{'': pickle.dumps(result)}``.

        Returns
        -------
        str
            Primary S3 key the result was uploaded under.

        Raises
        ------
        ValueError
            If ``mosaic_task_id`` is not provided in ``func_kwargs``.

        """
        func_args = func_args or ()
        func_kwargs = dict(func_kwargs) if func_kwargs else {}
        counter = func_kwargs.pop('mosaic_counter', self._counter)
        task_id = func_kwargs.pop('mosaic_task_id', None)

        if task_id is None:
            raise ValueError('exec_remote requires mosaic_task_id in func_kwargs')

        # Run the redux closure
        result = await func(None, *func_args, **func_kwargs)

        # Serialise into one or more {suffix: bytes} blobs
        blobs = {'': pickle.dumps(result)} if serialise is None else serialise(result)

        primary_key = f'{self.result_prefix}/counter_{counter}/task_{task_id}.pkl'
        for suffix, blob in blobs.items():
            key = primary_key if not suffix else primary_key.replace('.pkl', f'_{suffix}.pkl')
            self._upload_bytes(key, blob)

        return primary_key
