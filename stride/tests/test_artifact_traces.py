"""
Tests for PR 2: ArtifactTraces (stride-side).

Covers:
- ArtifactTraces stores only the artifact key, not data
- ArtifactTraces.load() fetches from the global warehouse
- ArtifactTraces.load() raises when no warehouse configured
- ArtifactTraces._data is always None (lazy)
- Legacy kwargs are discarded without error
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ArtifactTraces
# ---------------------------------------------------------------------------

class TestArtifactTraces:

    def _make_artifact_traces(self, key='shots/0/observed.npy'):
        from stride.problem.data import ArtifactTraces
        grid = MagicMock()
        grid.time = MagicMock()
        grid.time.num = 100
        grid.time.step = 0.001
        return ArtifactTraces(
            artifact_key=key,
            transducer_ids=[0, 1, 2],
            grid=grid,
        )

    def test_stores_artifact_key(self):
        traces = self._make_artifact_traces('shots/5/observed.npy')
        assert traces._artifact_key == 'shots/5/observed.npy'

    def test_data_is_none(self):
        traces = self._make_artifact_traces()
        assert traces._data is None

    def test_data_setter_is_noop(self):
        traces = self._make_artifact_traces()
        traces._data = np.array([1, 2, 3])
        assert traces._data is None

    def test_load_without_warehouse_raises(self):
        traces = self._make_artifact_traces()
        with patch('mosaic.get_artifact_warehouse', return_value=None):
            with pytest.raises(RuntimeError, match='No ArtifactWarehouse'):
                traces.load()

    def test_load_fetches_from_warehouse(self):
        traces = self._make_artifact_traces('shots/0/observed.npy')
        mock_wh = MagicMock()
        arr = np.zeros((3, 100))
        mock_wh.pull_remote.return_value = arr

        with patch('mosaic.get_artifact_warehouse', return_value=mock_wh):
            result = traces.load()

        mock_wh.pull_remote.assert_called_once_with('shots/0/observed.npy')
        assert result is not traces

    def test_legacy_kwargs_discarded(self):
        from stride.problem.data import ArtifactTraces
        grid = MagicMock()
        grid.time = MagicMock()
        grid.time.num = 10
        grid.time.step = 0.001
        traces = ArtifactTraces(
            artifact_key='k',
            transducer_ids=[0],
            grid=grid,
            artifact_config={'endpoint': 'x'},
            artifact_endpoint='y',
            artifact_access_key='z',
            artifact_secret_key='w',
            artifact_secure=False,
            artifact_backend='minio',
            artifact_bucket='b',
        )
        assert traces._artifact_key == 'k'
