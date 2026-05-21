"""
Tests for PR 2: Artifact Warehouse (S3/MinIO Storage).

Covers:
- ArtifactWarehouse construction and from_env()
- Key prefix computation (with and without run_prefix)
- Iteration management
- push_remote / pull_remote round-trip (mocked S3)
- write_shot_list format
- clear_iteration_gradients
- exec_remote gradient upload path
- ArtifactWarehouseObject
- Global set/get_artifact_warehouse
- Runtime.exec routing to artifact warehouse
"""

import os
import json
import pickle
import asyncio
import pytest
import numpy as np
from io import BytesIO
from unittest.mock import MagicMock, patch, call

from mosaic.runtime.artifact_warehouse import ArtifactWarehouse, ArtifactWarehouseObject


# ---------------------------------------------------------------------------
# ArtifactWarehouseObject
# ---------------------------------------------------------------------------

class TestArtifactWarehouseObject:

    def test_stores_key_and_bucket(self):
        obj = ArtifactWarehouseObject('shots/0/observed.npy', 'my-bucket')
        assert obj.key == 'shots/0/observed.npy'
        assert obj.bucket == 'my-bucket'

    def test_repr(self):
        obj = ArtifactWarehouseObject('k', 'b')
        assert 'k' in repr(obj)
        assert 'b' in repr(obj)


# ---------------------------------------------------------------------------
# ArtifactWarehouse construction
# ---------------------------------------------------------------------------

class TestArtifactWarehouseInit:

    def test_defaults(self):
        wh = ArtifactWarehouse(
            endpoint='localhost:9000',
            access_key='a',
            secret_key='s',
            bucket='test',
        )
        assert wh._endpoint == 'localhost:9000'
        assert wh._bucket == 'test'
        assert wh._secure is False
        assert wh._run_prefix == ''
        assert wh._gradient_prefix == 'gradients'
        assert wh._shot_prefix == 'shots'
        assert wh._iteration == 0
        assert wh._client is None

    def test_from_env(self):
        env = {
            'ARTIFACT_ENDPOINT': 'minio:9000',
            'ARTIFACT_ACCESS_KEY': 'admin',
            'ARTIFACT_SECRET_KEY': 'password',
            'ARTIFACT_BUCKET': 'stride-data',
            'ARTIFACT_SECURE': 'false',
            'ARTIFACT_RUN_ID': 'run-001',
        }
        with patch.dict(os.environ, env, clear=False):
            wh = ArtifactWarehouse.from_env()
        assert wh._endpoint == 'minio:9000'
        assert wh._access_key == 'admin'
        assert wh._bucket == 'stride-data'
        assert wh._run_prefix == 'run-001'

    def test_from_env_missing_endpoint_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError):
                ArtifactWarehouse.from_env()

    def test_from_env_defaults(self):
        env = {'ARTIFACT_ENDPOINT': 'localhost:9000'}
        with patch.dict(os.environ, env, clear=True):
            wh = ArtifactWarehouse.from_env()
        assert wh._access_key == 'minioadmin'
        assert wh._secret_key == 'minioadmin'
        assert wh._bucket == 'stride-data'
        assert wh._secure is False


# ---------------------------------------------------------------------------
# Key prefix computation
# ---------------------------------------------------------------------------

class TestKeyPrefixes:

    def test_no_run_prefix(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b', run_prefix='')
        assert wh.gradient_prefix == 'gradients'
        assert wh.shot_prefix == 'shots'

    def test_with_run_prefix(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b', run_prefix='run-42')
        assert wh.gradient_prefix == 'run-42/gradients'
        assert wh.shot_prefix == 'run-42/shots'

    def test_custom_gradient_prefix(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b',
                               run_prefix='r', gradient_prefix='grads')
        assert wh.gradient_prefix == 'r/grads'

    def test_custom_shot_prefix(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b',
                               run_prefix='r', shot_prefix='data')
        assert wh.shot_prefix == 'r/data'


# ---------------------------------------------------------------------------
# Iteration management
# ---------------------------------------------------------------------------

class TestIteration:

    def test_set_iteration(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        assert wh.iteration == 0
        wh.set_iteration(3)
        assert wh.iteration == 3


# ---------------------------------------------------------------------------
# push_remote / pull_remote
# ---------------------------------------------------------------------------

class TestPushPullRemote:

    def _make_warehouse(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        return wh

    def test_push_numpy_array(self):
        wh = self._make_warehouse()
        arr = np.array([1.0, 2.0, 3.0])
        result = wh.push_remote('test.npy', arr)
        assert isinstance(result, ArtifactWarehouseObject)
        assert result.key == 'test.npy'
        wh._client.put_object.assert_called_once()
        call_args = wh._client.put_object.call_args
        assert call_args[0][0] == 'b'
        assert call_args[0][1] == 'test.npy'

    def test_push_pickle_object(self):
        wh = self._make_warehouse()
        obj = {'key': 'value'}
        result = wh.push_remote('data.pkl', obj)
        assert result.key == 'data.pkl'
        wh._client.put_object.assert_called_once()

    def test_pull_npy_key(self):
        wh = self._make_warehouse()
        arr = np.array([4.0, 5.0])
        buf = BytesIO()
        np.save(buf, arr)
        raw = buf.getvalue()

        response = MagicMock()
        response.read.return_value = raw
        wh._client.get_object.return_value = response

        result = wh.pull_remote('test.npy')
        np.testing.assert_array_equal(result, arr)

    def test_pull_pkl_key(self):
        wh = self._make_warehouse()
        obj = {'foo': 42}
        raw = pickle.dumps(obj)

        response = MagicMock()
        response.read.return_value = raw
        wh._client.get_object.return_value = response

        result = wh.pull_remote('data.pkl')
        assert result == obj

    def test_pull_poll_retries(self):
        wh = self._make_warehouse()
        obj = [1, 2, 3]
        raw = pickle.dumps(obj)

        response = MagicMock()
        response.read.return_value = raw

        wh._client.get_object.side_effect = [
            Exception('not found'),
            Exception('not found'),
            response,
        ]
        with patch('time.sleep'):
            result = wh.pull_remote('data.pkl', poll=True, poll_interval=0.01)
        assert result == obj
        assert wh._client.get_object.call_count == 3

    def test_pull_no_poll_raises(self):
        wh = self._make_warehouse()
        wh._client.get_object.side_effect = Exception('not found')
        with pytest.raises(Exception, match='not found'):
            wh.pull_remote('data.pkl', poll=False)


# ---------------------------------------------------------------------------
# write_shot_list
# ---------------------------------------------------------------------------

class TestWriteShotList:

    def test_writes_correct_json(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b', run_prefix='run-1')
        uploaded = {}

        def mock_upload(key, data, content_type='application/octet-stream'):
            uploaded[key] = data

        wh._upload_bytes = mock_upload
        wh.write_shot_list(iteration=2, shot_ids=[0, 3, 5], attempt=1)

        key = 'run-1/gradients/iter_2/shots.json'
        assert key in uploaded
        payload = json.loads(uploaded[key].decode())
        assert payload['shot_ids'] == [0, 3, 5]
        assert payload['attempt'] == 1

    def test_default_attempt_zero(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        uploaded = {}

        def mock_upload(key, data, content_type='application/octet-stream'):
            uploaded[key] = data

        wh._upload_bytes = mock_upload
        wh.write_shot_list(iteration=0, shot_ids=[1, 2])

        payload = json.loads(list(uploaded.values())[0].decode())
        assert payload['attempt'] == 0


# ---------------------------------------------------------------------------
# clear_iteration_gradients
# ---------------------------------------------------------------------------

class TestClearIterationGradients:

    def test_deletes_pkl_files_only(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b', run_prefix='r')
        wh._client = MagicMock()

        pkl_obj = MagicMock()
        pkl_obj.object_name = 'r/gradients/iter_0/shot_0.pkl'
        json_obj = MagicMock()
        json_obj.object_name = 'r/gradients/iter_0/shots.json'
        final_obj = MagicMock()
        final_obj.object_name = 'r/gradients/iter_0/final.pkl'

        wh._client.list_objects.return_value = [pkl_obj, json_obj, final_obj]

        wh.clear_iteration_gradients(0)

        removed = [c[0][1] for c in wh._client.remove_object.call_args_list]
        assert 'r/gradients/iter_0/shot_0.pkl' in removed
        assert 'r/gradients/iter_0/final.pkl' in removed
        assert 'r/gradients/iter_0/shots.json' not in removed


# ---------------------------------------------------------------------------
# key_exists
# ---------------------------------------------------------------------------

class TestKeyExists:

    def test_exists(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        wh._client.stat_object.return_value = MagicMock()
        assert wh._key_exists('some/key') is True

    def test_not_exists(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        wh._client.stat_object.side_effect = Exception('NoSuchKey')
        assert wh._key_exists('missing/key') is False


# ---------------------------------------------------------------------------
# exec_remote
# ---------------------------------------------------------------------------

class TestExecRemote:

    def test_uploads_gradient_with_shot_id(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b', run_prefix='run-1')
        wh._client = MagicMock()
        wh.set_iteration(2)

        grad = MagicMock()
        grad.data = np.array([1.0, 2.0])
        grad.prec = None

        async def mock_func(rec_grads, *args, **kwargs):
            return [grad]

        loop = asyncio.new_event_loop()
        mock_runtime = MagicMock()
        mock_runtime.indices = (3,)
        with patch('mosaic.runtime', return_value=mock_runtime):
            result = loop.run_until_complete(
                wh.exec_remote('uid', mock_func,
                               func_kwargs={'iteration': 2, 'shot_id': 7}))
        loop.close()

        assert isinstance(result, ArtifactWarehouseObject)
        assert 'shot_7.pkl' in result.key
        assert 'iter_2' in result.key

    def test_uploads_gradient_without_shot_id(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        wh.set_iteration(0)

        grad = MagicMock()
        grad.data = np.array([1.0])
        grad.prec = None

        async def mock_func(rec_grads, *args, **kwargs):
            return [grad]

        loop = asyncio.new_event_loop()
        mock_runtime = MagicMock()
        mock_runtime.indices = (5,)
        with patch('mosaic.runtime', return_value=mock_runtime):
            result = loop.run_until_complete(
                wh.exec_remote('uid', mock_func))
        loop.close()

        assert 'worker_5' in result.key

    def test_uploads_preconditioner_alongside_gradient(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()

        grad = MagicMock()
        grad.data = np.array([1.0])
        grad.prec = MagicMock()
        grad.prec.data = np.array([0.5])

        async def mock_func(rec_grads, *args, **kwargs):
            return [grad]

        loop = asyncio.new_event_loop()
        mock_runtime = MagicMock()
        mock_runtime.indices = (0,)
        with patch('mosaic.runtime', return_value=mock_runtime):
            loop.run_until_complete(
                wh.exec_remote('uid', mock_func,
                               func_kwargs={'shot_id': 0}))
        loop.close()

        uploaded_keys = [c[0][1] for c in wh._client.put_object.call_args_list]
        prec_keys = [k for k in uploaded_keys if '_prec.pkl' in k]
        assert len(prec_keys) == 1


# ---------------------------------------------------------------------------
# Global set/get warehouse
# ---------------------------------------------------------------------------

class TestGlobalWarehouse:

    def test_set_and_get(self):
        import mosaic
        original = mosaic.get_artifact_warehouse()
        try:
            wh = ArtifactWarehouse('ep', 'a', 's', 'b')
            mosaic.set_artifact_warehouse(wh)
            assert mosaic.get_artifact_warehouse() is wh
        finally:
            mosaic._artifact_warehouse = original

    def test_default_is_none(self):
        import mosaic
        original = mosaic._artifact_warehouse
        try:
            mosaic._artifact_warehouse = None
            os.environ.pop('ARTIFACT_ENDPOINT', None)
            assert mosaic.get_artifact_warehouse() is None
        finally:
            mosaic._artifact_warehouse = original


# ---------------------------------------------------------------------------
# ensure_bucket
# ---------------------------------------------------------------------------

class TestEnsureBucket:

    def test_creates_bucket_if_not_exists(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        wh._client.bucket_exists.return_value = False
        wh.ensure_bucket()
        wh._client.make_bucket.assert_called_once_with('b')

    def test_skips_if_exists(self):
        wh = ArtifactWarehouse('ep', 'a', 's', 'b')
        wh._client = MagicMock()
        wh._client.bucket_exists.return_value = True
        wh.ensure_bucket()
        wh._client.make_bucket.assert_not_called()
