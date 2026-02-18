

import os
import numpy as np
from io import BytesIO
from dataclasses import dataclass, field

import mosaic


__all__ = ['S3Config', 'get_s3_client', 'ensure_bucket',
           'upload_array', 'download_array',
           'upload_shot_data', 'download_shot_data', 'stage_shots_to_s3',
           'upload_model', 'download_model',
           'list_gradients', 'clear_iteration_gradients']


@dataclass
class S3Config:
    """
    Configuration for S3/MinIO data persistence.

    Parameters
    ----------
    endpoint : str
        S3/MinIO endpoint, e.g. "localhost:9000".
    access_key : str
        Access key for authentication.
    secret_key : str
        Secret key for authentication.
    bucket : str
        Bucket name for storing data.
    secure : bool, optional
        Whether to use HTTPS, defaults to False.
    model_prefix : str, optional
        Object prefix for model data, defaults to "models".
    shot_prefix : str, optional
        Object prefix for shot data, defaults to "shots".
    gradient_prefix : str, optional
        Object prefix for gradient data, defaults to "gradients".

    """
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    model_prefix: str = "models"
    shot_prefix: str = "shots"
    gradient_prefix: str = "gradients"

    @classmethod
    def from_env(cls, prefix='MINIO'):
        """
        Create an S3Config from environment variables.

        Reads ``{prefix}_ENDPOINT``, ``{prefix}_ACCESS_KEY``, ``{prefix}_SECRET_KEY``,
        and ``{prefix}_BUCKET`` from the environment.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix, defaults to "MINIO".

        Returns
        -------
        S3Config

        """
        return cls(
            endpoint=os.environ.get(f'{prefix}_ENDPOINT', 'localhost:9000'),
            access_key=os.environ.get(f'{prefix}_ACCESS_KEY', 'minioadmin'),
            secret_key=os.environ.get(f'{prefix}_SECRET_KEY', 'minioadmin'),
            bucket=os.environ.get(f'{prefix}_BUCKET', 'stride-data'),
            secure=os.environ.get(f'{prefix}_SECURE', 'false').lower() == 'true',
        )


def get_s3_client(config):
    """
    Create a MinIO client from an S3Config.

    Parameters
    ----------
    config : S3Config
        S3/MinIO configuration.

    Returns
    -------
    minio.Minio
        MinIO client instance.

    """
    from minio import Minio
    return Minio(
        config.endpoint,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
    )


def ensure_bucket(client, bucket):
    """
    Create a bucket if it does not already exist.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    bucket : str
        Bucket name.

    """
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def upload_array(client, bucket, key, array):
    """
    Upload a numpy array to S3 as a .npy file.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).
    array : ndarray
        Numpy array to upload.

    """
    buffer = BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    client.put_object(
        bucket, key, buffer,
        length=buffer.getbuffer().nbytes,
        content_type='application/octet-stream',
    )


def download_array(client, bucket, key):
    """
    Download a numpy array from S3.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    bucket : str
        Bucket name.
    key : str
        Object key (path within the bucket).

    Returns
    -------
    ndarray
        Downloaded numpy array.

    """
    response = client.get_object(bucket, key)
    try:
        buffer = BytesIO(response.read())
        return np.load(buffer)
    finally:
        response.close()
        response.release_conn()


def upload_shot_data(client, config, shot_id, wavelets, observed):
    """
    Upload shot data (wavelets and observed traces) to S3 as HDF5.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    shot_id : int
        Shot identifier.
    wavelets : ndarray
        Wavelet data array.
    observed : ndarray
        Observed data array.

    """
    import h5py

    buffer = BytesIO()
    with h5py.File(buffer, 'w') as f:
        f.create_dataset('wavelets', data=wavelets)
        f.create_dataset('observed', data=observed)
    buffer.seek(0)

    key = f'{config.shot_prefix}/shot_{shot_id:05d}.h5'
    client.put_object(
        config.bucket, key, buffer,
        length=buffer.getbuffer().nbytes,
        content_type='application/x-hdf5',
    )


def download_shot_data(client, config, shot_id):
    """
    Download shot data from S3.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    shot_id : int
        Shot identifier.

    Returns
    -------
    tuple
        (wavelets, observed) numpy arrays.

    """
    import h5py

    key = f'{config.shot_prefix}/shot_{shot_id:05d}.h5'
    response = client.get_object(config.bucket, key)
    try:
        buffer = BytesIO(response.read())
    finally:
        response.close()
        response.release_conn()

    with h5py.File(buffer, 'r') as f:
        wavelets = f['wavelets'][:]
        observed = f['observed'][:]

    return wavelets, observed


def stage_shots_to_s3(client, config, problem):
    """
    Upload all shot data from a Problem to S3.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    problem : Problem
        Stride Problem containing acquisitions with shot data.

    """
    logger = mosaic.logger()
    ensure_bucket(client, config.bucket)

    for shot in problem.acquisitions.shots:
        wavelets_data = shot.wavelets.data if shot.wavelets is not None else None
        observed_data = shot.observed.data if shot.observed is not None else None

        if wavelets_data is None and observed_data is None:
            continue

        if wavelets_data is None:
            wavelets_data = np.array([])
        if observed_data is None:
            observed_data = np.array([])

        upload_shot_data(client, config, shot.id, wavelets_data, observed_data)
        logger.perf('Uploaded shot %d to S3' % shot.id)


def upload_model(client, config, iteration_id, model_args):
    """
    Upload model parameter arrays to S3 for a given iteration.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    iteration_id : int or str
        Iteration identifier.
    model_args : tuple or list
        Model parameter objects (each must have a ``.data`` attribute).

    """
    logger = mosaic.logger()
    ensure_bucket(client, config.bucket)

    for idx, arg in enumerate(model_args):
        if not hasattr(arg, 'data'):
            continue
        key = f'{config.model_prefix}/iter_{iteration_id}/arg_{idx}.npy'
        upload_array(client, config.bucket, key, np.asarray(arg.data))

    logger.perf('Uploaded model to S3 for iteration %s' % iteration_id)


def download_model(client, config, iteration_id, num_args):
    """
    Download model parameter arrays from S3.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    iteration_id : int or str
        Iteration identifier.
    num_args : int
        Number of model parameter arrays to download.

    Returns
    -------
    list
        List of numpy arrays.

    """
    arrays = []
    for idx in range(num_args):
        key = f'{config.model_prefix}/iter_{iteration_id}/arg_{idx}.npy'
        arrays.append(download_array(client, config.bucket, key))
    return arrays


def list_gradients(client, config, iteration_id):
    """
    List gradient objects stored in S3 for a given iteration.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    iteration_id : int or str
        Iteration identifier.

    Returns
    -------
    list
        List of object names.

    """
    prefix = f'{config.gradient_prefix}/iter_{iteration_id}/'
    objects = client.list_objects(config.bucket, prefix=prefix)
    return [obj.object_name for obj in objects]


def clear_iteration_gradients(client, config, iteration_id):
    """
    Remove all gradient objects for a given iteration from S3.

    Parameters
    ----------
    client : minio.Minio
        MinIO client.
    config : S3Config
        S3 configuration.
    iteration_id : int or str
        Iteration identifier.

    """
    prefix = f'{config.gradient_prefix}/iter_{iteration_id}/'
    objects = client.list_objects(config.bucket, prefix=prefix)
    for obj in objects:
        client.remove_object(config.bucket, obj.object_name)
