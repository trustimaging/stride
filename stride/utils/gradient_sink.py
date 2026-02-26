
from __future__ import annotations

from typing import Any

import mosaic
from stride.core import Operator


__all__ = ['GradientSink']


@mosaic.tessera
class GradientSink(Operator):
    """
    One instance per worker. Uploads that worker's accumulated gradient to MinIO.

    Parameters
    ----------
    variable : Any
        Worker-local variable tessera instance (resolved from proxy by mosaic).

    """

    def __init__(self, variable: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._variable = variable

    def upload(self, iteration_id: int, worker_id: int,
               endpoint: str, access_key: str, secret_key: str,
               bucket: str, gradient_prefix: str,
               secure: bool = False, **kwargs: Any) -> None:
        """
        Upload this worker's accumulated gradient to the artifact store.

        Runs on the worker. Reads ``variable.grad.data`` and uploads it as
        ``{gradient_prefix}/iter_{iteration_id}/worker_{worker_id}.npy``.

        Parameters
        ----------
        iteration_id : int
            Absolute iteration index.
        worker_id : int
            Index of this worker (used as part of the object key).
        endpoint : str
            Artifact store host:port.
        access_key : str
            Access key for authentication.
        secret_key : str
            Secret key for authentication.
        bucket : str
            Destination bucket name.
        gradient_prefix : str
            Object key prefix, e.g. ``'gradients'``.
        secure : bool, optional
            Whether to use HTTPS, defaults to False.

        """
        import numpy as np
        from stride.utils.artifacts import ArtifactConfig, get_client, upload_array

        config = ArtifactConfig(endpoint=endpoint, access_key=access_key,
                                secret_key=secret_key, bucket=bucket, secure=secure)
        client = get_client(config)
        key = '%s/iter_%d/worker_%d.npy' % (gradient_prefix, iteration_id, worker_id)
        upload_array(client, bucket, key, np.asarray(self._variable.grad.data))
        mosaic.logger().perf('Uploaded gradient iter_%d/worker_%d.npy'
                             % (iteration_id, worker_id))
