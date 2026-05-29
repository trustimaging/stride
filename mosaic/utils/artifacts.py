import os
import numpy as np
import pickle

from io import BytesIO
from dataclasses import dataclass
from abc import ABC, abstractmethod

__all__ = ['ArtifactConfig', 'ArtifactBackend']

@dataclass
class ArtifactConfig:
    """
    Configuration for artifact store (MinIO / S3-compatible) data persistence.

    Parameters
    ----------
    endpoint : str
        Artifact store endpoint, e.g. ``"minio.argo.svc.cluster.local:9000"`` or ``"s3.eu-west-2.amazonaws.com"``.
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
        ``'s3'`` uses the boto3 Python client.
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
    def from_env(cls, prefix = 'MOSAIC_ARTIFACT'):
        """
        Create an ArtifactConfig from environment variables.

        Reads ``{prefix}_ENDPOINT``, ``{prefix}_ACCESS_KEY``,
        ``{prefix}_SECRET_KEY``, ``{prefix}_BUCKET``, ``{prefix}_SECURE``,
        and ``{prefix}_BACKEND`` from the environment.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix, defaults to ``'MOSAIC_ARTIFACT'``.

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


class ArtifactBackend(ABC):
    """
    Interface for object-store backends, e.g. MinIO or S3. 
    """
    backend_name: str
    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'backend_name'):
            ArtifactBackend._registry[cls.backend_name] = cls
    
    @classmethod
    def from_config(cls, config):
        """
        Dispatch ``config.backend`` to its registered class and instantiate.

        Parameters
        ----------
        config : ArtifactConfig
            Backend configuration.

        Returns
        -------
        ArtifactBackend
            Concrete backend instance ready for use.

        Raises
        ------
        ValueError
            If ``config.backend`` is not a registered backend name.

        """
        try:
            backend_cls = cls._registry[config.backend]
        except KeyError:
            raise ValueError(
                f'Unknown backend {config.backend!r},'
                f'available: {sorted(cls._registry)}'
            ) from None
        return backend_cls(config)

    @abstractmethod
    def ensure_bucket(self, bucket):
        pass

    @abstractmethod
    def put(self, bucket, key, data):
        pass

    @abstractmethod
    def get(self, bucket, key):
        pass

    @abstractmethod
    def list_keys(self, bucket, prefix):
        pass

    @abstractmethod
    def delete(self, bucket, key):
        pass

    @abstractmethod
    def exists(self, bucket, key):
        pass


class MinioBackend(ArtifactBackend):

    backend_name = 'minio'

    def __init__(self, config):
        from minio import Minio
        self._client = Minio(
            endpoint=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure
        )

    def ensure_bucket(self, bucket):
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
    
    def put(self, bucket, key, data):
        buf = BytesIO(data)
        self._client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=buf,
            length=len(data),
            content_type='application/octet-stream'
        )

    def get(self, bucket, key):
        response = self._client.get_object(
            bucket_name=bucket, object_name=key
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    
    def list_keys(self, bucket, prefix):
        return [
            obj.object_name for obj in self._client.list_objects(
                bucket_name=bucket, prefix=prefix, recursive=True
            )
        ]

    def delete(self, bucket, key):
        self._client.remove_object(bucket_name=bucket, object_name=key)

    def exists(self, bucket, key):
        from minio.error import S3Error  # S3 wire protocol, not AWS specific
        try:
            self._client.stat_object(bucket_name=bucket, object_name=key)
            return True
        except S3Error:
            return False


class S3Backend(ArtifactBackend):
    
    backend_name = 's3'

    def __init__(self, config):
        import boto3
        scheme = 'https' if config.secure else 'http'
        self._client = boto3.client(
            's3',
            endpoint_url=f'{scheme}://{config.endpoint}',
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
        )

    def ensure_bucket(self, bucket):
        from botocore.exceptions import ClientError
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)
    
    def put(self, bucket, key, data):
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType='application/octet-stream'
        )

    def get(self, bucket, key):
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response['Body']
        try:
            return body.read()
        finally:
            body.close()

    def list_keys(self, bucket, prefix):
        paginator = self._client.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys

    def delete(self, bucket, key):
        self._client.delete_object(Bucket=bucket, Key=key)

    def exists(self, bucket, key):
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False


def upload_array(client, bucket, key, array):
    """
    Serialise a numpy array and upload it as a ``.npy`` object.

    Parameters
    ----------
    client : ArtifactBackend  
        Backend instance to upload through.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).
    array : np.ndarray
        Numpy array to upload.

    """
    buf = BytesIO()
    np.save(buf, array)
    client.put(bucket, key, buf.getvalue())


def download_array(client, bucket, key):
    """
    Download ``.npy`` object and deserialise it back to a numpy array.

    Parameters
    ----------
    client : ArtifactBackend
        Backend instance to download through.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).

    Returns
    -------
    np.ndarray
        Reconstructed numpy array.

    """
    return np.load(BytesIO(client.get(bucket, key)))


def upload_pickle(client, bucket, key, obj):
    """
    Pickle arbitrary Python object and upload bytes.

    Parameters
    ----------
    client : ArtifactBackend
        Backend instance to upload through.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).
    obj : object
        Any picklable Python object.

    """
    client.put(bucket, key, pickle.dumps(obj))


def download_pickle(client, bucket, key):
    """
    Download pickled object.

    Parameters
    ----------
    client : ArtifactBackend
        Backend instance to download through.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).

    Returns
    -------
    object
        Unpickled object.

    """
    return pickle.loads(client.get(bucket, key))
