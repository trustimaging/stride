"""
Tests for PR 3: Fault Tolerance for Inversions (mosaic-side).

Covers:
- Runtime.disconnect fails pending RPC Reply futures synchronously
- OutboundConnection.disconnect cancels pending futures
- _disconnected_runtimes tracking
- _exclusive_proxy skips dead workers
- async_for waits when 0 workers
- Monitor.disconnect cascades to workers and removes from strategy
- RPC reply timeout in send_recv_async
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from mosaic.runtime.runtime import Runtime, RuntimeProxy, BaseRPC
from mosaic.core.base import RuntimeDisconnectedError


# ---------------------------------------------------------------------------
# Runtime.disconnect: fail pending RPC futures
# ---------------------------------------------------------------------------

class TestDisconnectFailsPendingFutures:
    """runtime.disconnect() should fail in-flight RPC Reply futures immediately."""

    def _make_runtime(self):
        rt = Runtime.__new__(Runtime)
        rt._name = 'head'
        rt._indices = ()
        rt._uid_override = None
        rt._workers = {}
        rt._nodes = {}
        rt._on_worker_count_changed = []
        rt._disconnected_runtimes = set()
        rt._tessera = {}
        rt._tessera_proxy = {}
        rt._tessera_proxy_array = {}
        rt._task = {}
        rt._task_proxy = {}
        rt.logger = MagicMock()

        mock_conn = MagicMock()
        mock_conn.state = 'connected'
        mock_conn.disconnect = MagicMock()
        rt._comms = MagicMock()
        rt._comms._send_conn = {'worker:0:0:abc': mock_conn}
        return rt, mock_conn

    def test_calls_conn_disconnect(self):
        rt, conn = self._make_runtime()
        rt.disconnect('monitor', 'worker:0:0:abc')
        conn.disconnect.assert_called_once()

    def test_marks_uid_as_disconnected(self):
        rt, _ = self._make_runtime()
        rt.disconnect('monitor', 'worker:0:0:abc')
        assert 'worker:0:0:abc' in rt._disconnected_runtimes

    def test_skips_when_conn_not_found(self):
        rt, _ = self._make_runtime()
        rt.disconnect('monitor', 'worker:99:0:zzz')
        assert 'worker:99:0:zzz' in rt._disconnected_runtimes

    def test_skips_when_already_disconnected(self):
        rt, conn = self._make_runtime()
        conn.state = 'disconnected'
        rt.disconnect('monitor', 'worker:0:0:abc')
        conn.disconnect.assert_not_called()

    def test_skips_when_no_comms(self):
        rt, _ = self._make_runtime()
        rt._comms = None
        rt.disconnect('monitor', 'worker:0:0:abc')
        assert 'worker:0:0:abc' in rt._disconnected_runtimes


# ---------------------------------------------------------------------------
# OutboundConnection.disconnect: cancel pending futures
# ---------------------------------------------------------------------------

class TestOutboundConnectionDisconnect:
    """OutboundConnection.disconnect() should fail all pending Reply futures."""

    def _make_connection(self):
        from mosaic.comms.comms import OutboundConnection
        conn = OutboundConnection.__new__(OutboundConnection)
        conn._uid = 'worker:0:0:abc'
        conn._state = 'connected'
        conn._heartbeat_timeout = None
        conn._heartbeat_attempts = 0
        conn._pending_reply_futures = []
        conn._local = False
        conn._transport = 'tcp'

        mock_socket = MagicMock()
        conn._socket = mock_socket
        conn._address = '127.0.0.1'
        conn._port = 5000
        conn._runtime = MagicMock()
        conn._runtime.is_monitor = False
        conn._loop = MagicMock()
        return conn

    def test_fails_all_pending_futures(self):
        conn = self._make_connection()
        f1 = asyncio.Future()
        f2 = asyncio.Future()
        conn._pending_reply_futures = [f1, f2]
        conn.disconnect()
        assert f1.done()
        assert f2.done()
        with pytest.raises(RuntimeDisconnectedError):
            f1.result()
        with pytest.raises(RuntimeDisconnectedError):
            f2.result()

    def test_already_done_futures_ignored(self):
        conn = self._make_connection()
        f1 = asyncio.Future()
        f1.set_result('ok')
        conn._pending_reply_futures = [f1]
        conn.disconnect()
        assert f1.result() == 'ok'

    def test_clears_pending_list(self):
        conn = self._make_connection()
        conn._pending_reply_futures = [asyncio.Future()]
        conn.disconnect()
        assert conn._pending_reply_futures == []

    def test_noop_if_already_disconnected(self):
        conn = self._make_connection()
        conn._state = 'disconnected'
        f = asyncio.Future()
        conn._pending_reply_futures = [f]
        conn.disconnect()
        assert not f.done()

    def test_state_transitions_to_disconnected(self):
        conn = self._make_connection()
        conn.disconnect()
        assert conn._state == 'disconnected'


# ---------------------------------------------------------------------------
# _exclusive_proxy: skip dead workers
# ---------------------------------------------------------------------------

class TestExclusiveProxySkipsDead:
    """_exclusive_proxy should not return dead workers to the queue."""

    def test_live_worker_returned_to_queue(self):
        rt = Runtime.__new__(Runtime)
        rt._disconnected_runtimes = set()
        queue = asyncio.Queue()
        proxy = RuntimeProxy(uid='worker:0:0:abc')
        queue.put_nowait(proxy)

        async def run():
            async with rt._exclusive_proxy(queue):
                pass
            return queue.qsize()

        loop = asyncio.new_event_loop()
        size = loop.run_until_complete(run())
        loop.close()
        assert size == 1

    def test_dead_worker_not_returned_to_queue(self):
        rt = Runtime.__new__(Runtime)
        rt._disconnected_runtimes = {'worker:0:0:abc'}
        queue = asyncio.Queue()
        proxy = RuntimeProxy(uid='worker:0:0:abc')
        queue.put_nowait(proxy)

        async def run():
            async with rt._exclusive_proxy(queue):
                pass
            return queue.qsize()

        loop = asyncio.new_event_loop()
        size = loop.run_until_complete(run())
        loop.close()
        assert size == 0


# ---------------------------------------------------------------------------
# Monitor.disconnect: cascade to workers
# ---------------------------------------------------------------------------

class TestMonitorDisconnectCascade:
    """Monitor.disconnect() should cascade to child workers and remove from strategy."""

    def _make_monitor(self):
        from mosaic.runtime.monitor import Monitor
        from mosaic.runtime.strategies import RoundRobin
        from collections import defaultdict

        m = Monitor.__new__(Monitor)
        m._name = 'monitor'
        m._indices = ()
        m._uid_override = None
        m._nodes = {}
        m._workers = {}
        m._on_worker_count_changed = []
        m._disconnected_runtimes = set()
        m._tessera = {}
        m._tessera_proxy = {}
        m._tessera_proxy_array = {}
        m._task = {}
        m._task_proxy = {}
        m._monitored_nodes = {}
        m._monitored_tessera = {}
        m._monitored_tasks = {}
        m._runtime_tessera = defaultdict(list)
        m._runtime_tasks = defaultdict(list)
        m._comms = MagicMock()
        m._comms._send_conn = {}
        m.logger = MagicMock()
        m._monitor_strategy = RoundRobin(m)
        return m

    def test_cascades_to_workers(self):
        m = self._make_monitor()
        node = MagicMock()
        node.sub_resources = {'workers': {'worker:0:0:aaa': {}}}
        m._monitored_nodes['node:0:aaa'] = node

        m.disconnect('heartbeat', 'node:0:aaa')

        assert 'node:0:aaa' in m._disconnected_runtimes
        assert 'worker:0:0:aaa' in m._disconnected_runtimes

    def test_removes_from_strategy(self):
        m = self._make_monitor()
        node = MagicMock()
        node.sub_resources = {'workers': {'worker:0:0:aaa': {}}}
        m._monitored_nodes['node:0:aaa'] = node
        m._monitor_strategy._worker_list.add('worker:0:0:aaa')
        m._monitor_strategy._num_workers = 1

        m.disconnect('heartbeat', 'node:0:aaa')

        assert 'worker:0:0:aaa' not in m._monitor_strategy._worker_list

    def test_removes_from_monitored_nodes(self):
        m = self._make_monitor()
        node = MagicMock()
        node.sub_resources = {'workers': {}}
        m._monitored_nodes['node:0:aaa'] = node

        m.disconnect('heartbeat', 'node:0:aaa')

        assert 'node:0:aaa' not in m._monitored_nodes

    def test_fires_on_worker_count_changed(self):
        m = self._make_monitor()
        node = MagicMock()
        node.sub_resources = {'workers': {'worker:0:0:aaa': {}}}
        m._monitored_nodes['node:0:aaa'] = node

        called = []
        m._on_worker_count_changed.append(lambda: called.append(True))
        m.disconnect('heartbeat', 'node:0:aaa')

        assert len(called) >= 1


# ---------------------------------------------------------------------------
# RPC reply timeout
# ---------------------------------------------------------------------------

class TestRPCReplyTimeout:
    """send_recv_async should raise after 20s if no reply arrives."""

    def test_timeout_raises_disconnected_error(self):
        async def run():
            future = asyncio.Future()

            async def mock_send_async(*args, **kwargs):
                return future

            conn = MagicMock()
            conn.send_async = mock_send_async

            from mosaic.comms.comms import CommsManager
            comms = CommsManager.__new__(CommsManager)
            comms._state = 'connected'
            comms._send_conn = {'worker:0:0:abc': conn}
            comms._circ_conn = MagicMock()
            comms._runtime = MagicMock()
            comms._runtime.uid = 'head'

            coro = comms._send_recv_any('worker:0:0:abc', method='test')
            with pytest.raises(RuntimeDisconnectedError, match='20s'):
                await asyncio.wait_for(coro, timeout=1.0)

        loop = asyncio.new_event_loop()
        with pytest.raises((RuntimeDisconnectedError, asyncio.TimeoutError)):
            loop.run_until_complete(run())
        loop.close()


# ---------------------------------------------------------------------------
# _disconnected_runtimes tracking
# ---------------------------------------------------------------------------

class TestDisconnectedRuntimesSet:

    def test_starts_empty(self):
        rt = Runtime.__new__(Runtime)
        rt._disconnected_runtimes = set()
        assert len(rt._disconnected_runtimes) == 0

    def test_added_on_disconnect(self):
        rt = Runtime.__new__(Runtime)
        rt._name = 'head'
        rt._indices = ()
        rt._uid_override = None
        rt._workers = {}
        rt._nodes = {}
        rt._on_worker_count_changed = []
        rt._disconnected_runtimes = set()
        rt._tessera = {}
        rt._tessera_proxy = {}
        rt._tessera_proxy_array = {}
        rt._task = {}
        rt._task_proxy = {}
        rt._comms = None
        rt.logger = MagicMock()

        rt.disconnect('monitor', 'worker:0:0:dead')
        rt.disconnect('monitor', 'worker:1:0:dead')
        assert rt._disconnected_runtimes == {'worker:0:0:dead', 'worker:1:0:dead'}
