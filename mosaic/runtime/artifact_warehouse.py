
from __future__ import annotations

import os
import pickle
import asyncio
from io import BytesIO


__all__ = ['ArtifactWarehouse', 'ArtifactWarehouseObject']


class ArtifactWarehouseObject:
    """
    Lightweight handle pointing to data stored in the artifact warehouse.

    Mirrors the role of ``WarehouseObject`` for the SpillBuffer: a thin
    reference that records where data lives without holding the data itself.

    Parameters
    ----------
    key : str
        Object key within the bucket.
    bucket : str
        Bucket name.

    """

    def __init__(self, key: str, bucket: str) -> None:
        self.key = key
        self.bucket = bucket

    def __repr__(self) -> str:
        return '<ArtifactWarehouseObject key=%s bucket=%s>' % (self.key, self.bucket)


class ArtifactWarehouse:
    """
    S3/MinIO-backed warehouse that runs inline (no subprocess).

    Each process holds its own client and communicates directly with the
    object store.  The warehouse is registered globally via
    ``mosaic.set_artifact_warehouse()`` so all runtimes share the same
    configuration without passing it through kwargs.

    Parameters
    ----------
    endpoint : str
        MinIO/S3 endpoint, e.g. ``"minio.argo.svc.cluster.local:9000"``.
    access_key : str
        Access key for authentication.
    secret_key : str
        Secret key for authentication.
    bucket : str
        Bucket name.
    secure : bool, optional
        Whether to use HTTPS, defaults to False.
    run_prefix : str, optional
        Top-level key prefix that scopes all objects to a single run,
        e.g. the Argo workflow name.  All shot and gradient keys are stored
        under ``{run_prefix}/shots/...`` and ``{run_prefix}/gradients/...``.
        Defaults to ``''`` (no prefix — legacy behaviour).
    gradient_prefix : str, optional
        Key prefix for gradient objects, defaults to ``'gradients'``.
    shot_prefix : str, optional
        Key prefix for shot data objects, defaults to ``'shots'``.

    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        run_prefix: str = '',
        gradient_prefix: str = 'gradients',
        shot_prefix: str = 'shots',
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._secure = secure
        self._run_prefix = run_prefix
        self._gradient_prefix = gradient_prefix
        self._shot_prefix = shot_prefix
        self._iteration = 0
        self._client = None

    @classmethod
    def from_env(cls, prefix: str = 'ARTIFACT') -> 'ArtifactWarehouse':
        """
        Create an ``ArtifactWarehouse`` from environment variables.

        Reads ``{prefix}_ENDPOINT``, ``{prefix}_ACCESS_KEY``,
        ``{prefix}_SECRET_KEY``, ``{prefix}_BUCKET``, ``{prefix}_SECURE``,
        ``{prefix}_GRADIENT_PREFIX``, and ``{prefix}_SHOT_PREFIX``.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix, defaults to ``'ARTIFACT'``.

        Returns
        -------
        ArtifactWarehouse

        """
        return cls(
            endpoint=os.environ[f'{prefix}_ENDPOINT'],
            access_key=os.environ.get(f'{prefix}_ACCESS_KEY', 'minioadmin'),
            secret_key=os.environ.get(f'{prefix}_SECRET_KEY', 'minioadmin'),
            bucket=os.environ.get(f'{prefix}_BUCKET', 'stride-data'),
            secure=os.environ.get(f'{prefix}_SECURE', 'false').lower() == 'true',
            run_prefix=os.environ.get(f'{prefix}_RUN_ID', ''),
            gradient_prefix=os.environ.get(f'{prefix}_GRADIENT_PREFIX', 'gradients'),
            shot_prefix=os.environ.get(f'{prefix}_SHOT_PREFIX', 'shots'),
        )

    # ------------------------------------------------------------------ properties

    @property
    def client(self):
        """Lazily-initialised MinIO client."""
        if self._client is None:
            from minio import Minio
            import urllib3
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=5, read=30),
                retries=urllib3.Retry(
                    total=5,
                    backoff_factor=0.2,
                    status_forcelist=[500, 502, 503, 504],
                ),
            )
            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                http_client=http_client,
            )
        return self._client

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def run_prefix(self) -> str:
        return self._run_prefix

    @property
    def gradient_prefix(self) -> str:
        if self._run_prefix:
            return '%s/%s' % (self._run_prefix, self._gradient_prefix)
        return self._gradient_prefix

    @property
    def shot_prefix(self) -> str:
        if self._run_prefix:
            return '%s/%s' % (self._run_prefix, self._shot_prefix)
        return self._shot_prefix

    @property
    def iteration(self) -> int:
        return self._iteration

    # ------------------------------------------------------------------ setup

    def set_iteration(self, iteration: int) -> None:
        """Set the current inversion iteration (used to construct gradient keys)."""
        self._iteration = iteration

    def write_shot_list(self, iteration: int, shot_ids: list) -> None:
        """
        Write the list of expected shot IDs for *iteration* to S3.

        Called by the head before dispatching shots each iteration, and again
        after the loop if workers dropped below the threshold (to tell the
        accumulator which shots actually completed).

        Parameters
        ----------
        iteration : int
            Zero-based absolute iteration index.
        shot_ids : list of int
            Shot IDs expected (or completed) for this iteration.

        """
        import json
        key = '%s/iter_%d/shots.json' % (self.gradient_prefix, iteration)
        self._upload_bytes(key, json.dumps(shot_ids).encode())

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist."""
        if not self.client.bucket_exists(self._bucket):
            self.client.make_bucket(self._bucket)

    # ------------------------------------------------------------------ low-level I/O

    def _upload_bytes(self, key: str, data: bytes,
                      content_type: str = 'application/octet-stream') -> None:
        buf = BytesIO(data)
        self.client.put_object(
            self._bucket, key, buf,
            length=len(data),
            content_type=content_type,
        )

    def _download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _key_exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self._bucket, key)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ public API

    def push_remote(self, key: str, data) -> ArtifactWarehouseObject:
        """
        Upload data to the artifact store.

        Numpy arrays are stored in ``.npy`` format; all other objects
        are pickled.

        Parameters
        ----------
        key : str
            Object key within the bucket.
        data
            Data to upload (numpy array or picklable object).

        Returns
        -------
        ArtifactWarehouseObject

        """
        import numpy as np

        if isinstance(data, np.ndarray):
            buf = BytesIO()
            np.save(buf, data)
            raw = buf.getvalue()
        else:
            raw = pickle.dumps(data)

        self._upload_bytes(key, raw)
        return ArtifactWarehouseObject(key, self._bucket)

    def pull_remote(self, key: str, poll: bool = False,
                    poll_interval: float = 1.0):
        """
        Download data from the artifact store.

        Keys ending in ``.npy`` are loaded with ``numpy.load``; all
        others are unpickled.  If *poll* is ``True`` the call blocks
        until the key is available.

        Parameters
        ----------
        key : str
            Object key within the bucket.
        poll : bool, optional
            Block until the key is available, defaults to False.
        poll_interval : float, optional
            Initial polling interval in seconds, defaults to 1.0.

        Returns
        -------
        object

        """
        import time
        import numpy as np

        wait = poll_interval
        while True:
            try:
                raw = self._download_bytes(key)
                break
            except Exception:
                if not poll:
                    raise
                time.sleep(wait)
                wait = min(wait * 1.5, 30.0)

        if key.endswith('.npy'):
            return np.load(BytesIO(raw))
        else:
            return pickle.loads(raw)

    async def exec_remote(self, uid, func,
                          func_args=None,
                          func_kwargs=None) -> ArtifactWarehouseObject:
        """
        Run *func* to produce a gradient, then upload it to S3.

        Mirrors the ``Warehouse.exec_remote`` interface but instead of
        storing the result in the SpillBuffer it serialises the gradient
        as ``{gradient_prefix}/iter_{I}/worker_{K}.pkl`` in the artifact
        store.

        Parameters
        ----------
        uid
            Identifier for the accumulated object.
        func
            Async redux function with signature ``func(rec_grads, *args)``.
        func_args
            Positional arguments to pass after *rec_grads*.
        func_kwargs
            Keyword arguments to pass to *func*.

        Returns
        -------
        ArtifactWarehouseObject

        """
        import numpy as np
        import mosaic

        func_args = func_args or ()
        func_kwargs = dict(func_kwargs) if func_kwargs else {}
        iteration = func_kwargs.pop('iteration', self._iteration)
        shot_id   = func_kwargs.pop('shot_id', None)

        # Call the redux closure (rec_grads=None — no prior accumulated value)
        result = await func(None, *func_args, **func_kwargs)

        # result is a list/tuple of summed gradients; take the first
        grad = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(grad, tuple):
            grad = grad[0]

        # Key by shot ID when available, fall back to worker index
        runtime = mosaic.runtime()
        worker_id = (runtime.indices[0]
                     if (runtime is not None and len(runtime.indices))
                     else 0)
        id_part = ('shot_%d' % shot_id) if shot_id is not None else ('worker_%d' % worker_id)
        key = '%s/iter_%d/%s.pkl' % (self.gradient_prefix, iteration, id_part)

        if hasattr(grad, 'data'):
            self._upload_bytes(key, pickle.dumps(np.asarray(grad.data)))
        elif grad is not None:
            self._upload_bytes(key, pickle.dumps(grad))

        return ArtifactWarehouseObject(key, self._bucket)
