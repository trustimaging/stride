"""
Tests for PR 3: Fault Tolerance for Inversions (stride-side).

Covers:
- _wait_for_workers: Phase 1 (event-driven wait for target count)
- _wait_for_workers: Phase 2 (check_node_status polling)
- _wait_for_workers: timeout behaviour
- _watch_workers: cancels task on drop threshold breach
- _watch_workers: no-op when task finishes cleanly
- _start_worker_monitor: captures baseline UIDs, wires callback
- _start_worker_monitor: disabled when drop_threshold is None
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from stride import _wait_for_workers, _watch_workers, _start_worker_monitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_runtime(num_workers=0, worker_uids=None):
    """Build a mock runtime with controllable worker pool."""
    rt = MagicMock()
    if worker_uids is None:
        worker_uids = ['worker:%d:0:aaa' % i for i in range(num_workers)]
    workers = [MagicMock(uid=uid) for uid in worker_uids]
    rt.workers = workers
    rt.num_workers = len(workers)
    rt._on_worker_count_changed = []
    rt.get_monitor.return_value = None
    return rt


def _make_mock_runtime_with_monitor(num_workers, all_ready=True):
    """Runtime with a monitor that answers check_node_status."""
    worker_uids = ['worker:%d:0:aaa' % i for i in range(num_workers)]
    rt = _make_mock_runtime(num_workers, worker_uids)

    monitor = MagicMock()
    if all_ready:
        status = {'node:%d:aaa' % i: True for i in range(num_workers)}
    else:
        status = {'node:%d:aaa' % i: (i == 0) for i in range(num_workers)}

    async def mock_check(**kwargs):
        return status

    monitor.check_node_status = mock_check
    rt.get_monitor.return_value = monitor
    return rt


# ---------------------------------------------------------------------------
# _wait_for_workers: Phase 1
# ---------------------------------------------------------------------------

class TestWaitForWorkersPhase1:

    def test_skips_when_already_enough(self):
        rt = _make_mock_runtime(4)
        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 4, timeout=1.0))
        loop.close()

    def test_waits_until_target_reached(self):
        rt = _make_mock_runtime(0)

        async def simulate():
            async def add_workers_later():
                await asyncio.sleep(0.1)
                workers = [MagicMock(uid='worker:0:0:aaa'), MagicMock(uid='worker:1:0:bbb')]
                rt.workers = workers
                rt.num_workers = 2
                for cb in rt._on_worker_count_changed:
                    cb()

            asyncio.ensure_future(add_workers_later())
            with patch('stride.mosaic.logger', return_value=MagicMock()):
                await _wait_for_workers(rt, 2, timeout=5.0)
            assert rt.num_workers >= 2

        loop = asyncio.new_event_loop()
        loop.run_until_complete(simulate())
        loop.close()

    def test_timeout_proceeds_with_fewer(self):
        rt = _make_mock_runtime(1)
        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 5, timeout=0.3, heartbeat=0.1))
        loop.close()

    def test_callback_cleaned_up(self):
        rt = _make_mock_runtime(4)
        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 4, timeout=1.0))
        loop.close()
        assert len(rt._on_worker_count_changed) == 0


# ---------------------------------------------------------------------------
# _wait_for_workers: Phase 2 (check_node_status)
# ---------------------------------------------------------------------------

class TestWaitForWorkersPhase2:

    def test_all_ready_passes(self):
        rt = _make_mock_runtime_with_monitor(3, all_ready=True)
        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 3, timeout=5.0))
        loop.close()

    def test_no_monitor_skips_phase2(self):
        rt = _make_mock_runtime(3)
        rt.get_monitor.return_value = None
        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 3, timeout=1.0))
        loop.close()

    def test_check_node_status_exception_proceeds(self):
        rt = _make_mock_runtime(2)
        monitor = MagicMock()

        async def mock_check(**kwargs):
            raise ConnectionError('RPC failed')

        monitor.check_node_status = mock_check
        rt.get_monitor.return_value = monitor

        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            loop.run_until_complete(_wait_for_workers(rt, 2, timeout=1.0))
        loop.close()


# ---------------------------------------------------------------------------
# _watch_workers
# ---------------------------------------------------------------------------

class TestWatchWorkers:

    def test_cancels_task_on_threshold_breach(self):
        rt = _make_mock_runtime(4)
        initial_uids = set(w.uid for w in rt.workers)
        task = asyncio.Future()
        event = asyncio.Event()

        async def simulate():
            watcher = asyncio.ensure_future(
                _watch_workers(rt, initial_uids, threshold=0.0, task=task, event=event))

            await asyncio.sleep(0.05)
            rt.workers = [MagicMock(uid='worker:0:0:aaa')]
            rt.num_workers = 1
            event.set()

            await asyncio.sleep(3)
            return task.cancelled()

        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            cancelled = loop.run_until_complete(simulate())
        loop.close()
        assert cancelled

    def test_no_cancel_when_no_drops(self):
        rt = _make_mock_runtime(4)
        initial_uids = set(w.uid for w in rt.workers)

        async def simulate():
            task = asyncio.ensure_future(asyncio.sleep(0.5))
            event = asyncio.Event()
            watcher = asyncio.ensure_future(
                _watch_workers(rt, initial_uids, threshold=0.0, task=task, event=event))

            await task
            event.set()
            await asyncio.sleep(0.05)
            return True

        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            result = loop.run_until_complete(simulate())
        loop.close()
        assert result

    def test_threshold_50_allows_partial_drops(self):
        rt = _make_mock_runtime(4)
        initial_uids = set(w.uid for w in rt.workers)
        task = asyncio.Future()
        event = asyncio.Event()

        async def simulate():
            watcher = asyncio.ensure_future(
                _watch_workers(rt, initial_uids, threshold=0.5, task=task, event=event))

            await asyncio.sleep(0.05)
            rt.workers = [MagicMock(uid='worker:0:0:aaa'),
                          MagicMock(uid='worker:1:0:aaa')]
            rt.num_workers = 2
            event.set()

            await asyncio.sleep(0.2)
            return not task.cancelled()

        loop = asyncio.new_event_loop()
        with patch('stride.mosaic.logger', return_value=MagicMock()):
            not_cancelled = loop.run_until_complete(simulate())
        loop.close()
        assert not_cancelled


# ---------------------------------------------------------------------------
# _start_worker_monitor
# ---------------------------------------------------------------------------

class TestStartWorkerMonitor:

    def test_disabled_when_threshold_is_none(self):
        rt = _make_mock_runtime(4)
        task = asyncio.Future()
        monitor_task, cleanup = _start_worker_monitor(rt, None, task)
        assert monitor_task is None
        cleanup()

    def test_captures_baseline_uids(self):
        rt = _make_mock_runtime(3, ['worker:0:0:aaa', 'worker:1:0:bbb', 'worker:2:0:ccc'])
        loop_task = asyncio.Future()

        with patch('stride.mosaic.logger', return_value=MagicMock()):
            monitor_task, cleanup = _start_worker_monitor(rt, 0.0, loop_task)

        assert monitor_task is not None
        assert len(rt._on_worker_count_changed) == 1
        cleanup()
        assert len(rt._on_worker_count_changed) == 0
        monitor_task.cancel()

    def test_cleanup_removes_callback(self):
        rt = _make_mock_runtime(2)
        loop_task = asyncio.Future()

        with patch('stride.mosaic.logger', return_value=MagicMock()):
            _, cleanup = _start_worker_monitor(rt, 0.0, loop_task)

        assert len(rt._on_worker_count_changed) == 1
        cleanup()
        assert len(rt._on_worker_count_changed) == 0

    def test_double_cleanup_safe(self):
        rt = _make_mock_runtime(2)
        loop_task = asyncio.Future()

        with patch('stride.mosaic.logger', return_value=MagicMock()):
            _, cleanup = _start_worker_monitor(rt, 0.0, loop_task)

        cleanup()
        cleanup()
        assert len(rt._on_worker_count_changed) == 0
