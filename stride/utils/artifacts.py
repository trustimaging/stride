
from __future__ import annotations

import os
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minio import Minio


__all__ = ['ArtifactConfig', 'get_client', 'ensure_bucket',
           'upload_array', 'download_array']


@dataclass
class ArtifactConfig:
    """
    Configuration for artifact store (MinIO / S3-compatible) data persistence.

    Parameters
    ----------
    endpoint : str
        Artifact store endpoint, e.g. ``"minio.argo.svc.cluster.local:9000"``.
    access_key : str
        Access key for authentication.
    secret_key : str
        Secret key for authentication.
    bucket : str
        Bucket name for storing artifacts.
    secure : bool, optional
        Whether to use HTTPS, defaults to False.
    backend : str, optional
        Storage backend. ``'minio'`` (default) uses the MinIO Python client,
        which is compatible with any S3-compatible store. Future values: ``'s3'``.
    shot_prefix : str, optional
        Object key prefix for shot data, defaults to ``'shots'``.
    gradient_prefix : str, optional
        Object key prefix for gradient data, defaults to ``'gradients'``.

    """
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    backend: str = 'minio'
    shot_prefix: str = 'shots'
    gradient_prefix: str = 'gradients'

    @classmethod
    def from_env(cls: type[ArtifactConfig], prefix: str = 'ARTIFACT') -> ArtifactConfig:
        """
        Create an ArtifactConfig from environment variables.

        Reads ``{prefix}_ENDPOINT``, ``{prefix}_ACCESS_KEY``,
        ``{prefix}_SECRET_KEY``, ``{prefix}_BUCKET``, ``{prefix}_SECURE``,
        and ``{prefix}_BACKEND`` from the environment.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix, defaults to ``'ARTIFACT'``.

        Returns
        -------
        ArtifactConfig

        """
        return cls(
            endpoint=os.environ[f'{prefix}_ENDPOINT'],
            access_key=os.environ.get(f'{prefix}_ACCESS_KEY', 'minioadmin'),
            secret_key=os.environ.get(f'{prefix}_SECRET_KEY', 'minioadmin'),
            bucket=os.environ.get(f'{prefix}_BUCKET', 'stride-data'),
            secure=os.environ.get(f'{prefix}_SECURE', 'false').lower() == 'true',
            backend=os.environ.get(f'{prefix}_BACKEND', 'minio'),
        )


def get_client(config: ArtifactConfig) -> Minio:
    """
    Create a MinIO client from an ArtifactConfig.

    Parameters
    ----------
    config : ArtifactConfig
        Artifact store configuration.

    Returns
    -------
    Minio
        MinIO client instance.

    """
    from minio import Minio
    return Minio(
        config.endpoint,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    """
    Create a bucket if it does not already exist.

    Parameters
    ----------
    client : Minio
        MinIO client.
    bucket : str
        Bucket name.

    """
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_array(client: Minio, bucket: str, key: str, array: np.ndarray) -> None:
    """
    Upload a numpy array to the artifact store as a ``.npy`` object.

    Parameters
    ----------
    client : Minio
        MinIO client.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).
    array : np.ndarray
        Numpy array to upload.

    """
    buf = BytesIO()
    np.save(buf, array)
    buf.seek(0)
    client.put_object(
        bucket, key, buf,
        length=buf.getbuffer().nbytes,
        content_type='application/octet-stream',
    )


def download_array(client: Minio, bucket: str, key: str) -> np.ndarray:
    """
    Download a numpy array from the artifact store.

    Parameters
    ----------
    client : Minio
        MinIO client.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).

    Returns
    -------
    np.ndarray
        Downloaded numpy array.

    """
    response = client.get_object(bucket, key)
    try:
        buf = BytesIO(response.read())
        return np.load(buf)
    finally:
        response.close()
        response.release_conn()
