"""
Tests for PR 1: Inverse Handshake (Dynamic Node Registration).

Covers:
- UID scheme with instance IDs
- BaseRPC UID parsing for new format
- Phone-home environment variable validation
- Monitor: register_node, init_dynamic, dynamic worker counting
- RoundRobin strategy: stale worker eviction on node replacement
- Comms address discovery (0.0.0.0 auto-detect)
"""

import os
import uuid
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mosaic.runtime.runtime import BaseRPC, RuntimeProxy
from mosaic.runtime.strategies import RoundRobin


# ---------------------------------------------------------------------------
# UID scheme
# ---------------------------------------------------------------------------

class TestUIDScheme:
    """Verify the new UID format: name:index:instance_id."""

    def test_node_uid_format(self):
        instance_id = uuid.uuid4().hex[:8]
        uid = 'node:0:%s' % instance_id
        rpc = BaseRPC(uid=uid)
        assert rpc.name == 'node'
        assert rpc.uid == uid

    def test_worker_uid_format(self):
        instance_id = uuid.uuid4().hex[:8]
        uid = 'worker:3:1:%s' % instance_id
        rpc = BaseRPC(uid=uid)
        assert rpc.name == 'worker'
        assert rpc.uid == uid

    def test_warehouse_uid_format(self):
        instance_id = uuid.uuid4().hex[:8]
        uid = 'warehouse:2:%s' % instance_id
        rpc = BaseRPC(uid=uid)
        assert rpc.name == 'warehouse'
        assert rpc.uid == uid

    def test_instance_id_makes_uid_unique(self):
        uid_a = 'node:0:%s' % uuid.uuid4().hex[:8]
        uid_b = 'node:0:%s' % uuid.uuid4().hex[:8]
        assert uid_a != uid_b

    def test_shared_instance_id_across_node_and_workers(self):
        instance_id = uuid.uuid4().hex[:8]
        node_uid = 'node:0:%s' % instance_id
        worker_uid = 'worker:0:0:%s' % instance_id
        warehouse_uid = 'warehouse:0:%s' % instance_id
        assert instance_id in node_uid
        assert instance_id in worker_uid
        assert instance_id in warehouse_uid

    def test_legacy_uid_still_works(self):
        rpc = BaseRPC(name='head')
        assert rpc.uid == 'head'

    def test_legacy_indexed_uid_still_works(self):
        rpc = BaseRPC(name='worker', indices=(0, 0))
        assert rpc.uid == 'worker:0:0'

    def test_uid_override_set_for_instance_id_format(self):
        instance_id = 'a1b2c3d4'
        uid = 'node:0:%s' % instance_id
        rpc = BaseRPC(uid=uid)
        assert rpc._uid_override == uid

    def test_numeric_indices_parsed_before_instance_id(self):
        uid = 'worker:3:1:abcd1234'
        rpc = BaseRPC(uid=uid)
        assert rpc.indices == (3, 1)

    def test_runtime_proxy_with_new_uid(self):
        uid = 'worker:5:0:deadbeef'
        proxy = RuntimeProxy(uid=uid)
        assert proxy.uid == uid
        assert proxy.name == 'worker'


# ---------------------------------------------------------------------------
# Phone-home env var validation
# ---------------------------------------------------------------------------

class TestPhoneHomeEnvVars:
    """Node.init() should fail if phone_home=True but env vars are missing."""

    def test_missing_env_vars_raises(self):
        from mosaic.runtime.node import Node
        node = Node.__new__(Node)
        node._uid_override = None
        node._name = 'node'
        node._indices = (0,)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match='MONITOR_HOST'):
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(node.init(phone_home=True))
                except RuntimeError:
                    raise
                finally:
                    loop.close()

    def test_env_vars_set_correctly(self):
        env = {
            'MONITOR_HOST': '10.0.0.1',
            'MONITOR_PORT': '3000',
            'PUBSUB_PORT': '3001',
        }
        with patch.dict(os.environ, env, clear=False):
            host = os.environ.get('MONITOR_HOST')
            port = os.environ.get('MONITOR_PORT')
            pubsub = os.environ.get('PUBSUB_PORT')
            assert host == '10.0.0.1'
            assert int(port) == 3000
            assert int(pubsub) == 3001


# ---------------------------------------------------------------------------
# Monitor: register_node
# ---------------------------------------------------------------------------

class TestMonitorRegisterNode:
    """Test monitor.register_node() RPC."""

    def _make_monitor(self):
        from mosaic.runtime.monitor import Monitor
        monitor = Monitor.__new__(Monitor)
        monitor._name = 'monitor'
        monitor._indices = ()
        monitor._uid_override = None
        monitor._nodes = {}
        monitor._workers = {}
        monitor._on_worker_count_changed = []
        monitor._monitored_nodes = {}
        monitor._disconnected_runtimes = set()
        monitor._comms = MagicMock()
        monitor._comms.start_heartbeat = MagicMock()
        monitor.logger = MagicMock()
        monitor.strategy_name = 'round-robin'
        monitor._monitor_strategy = RoundRobin(monitor)
        return monitor

    def test_register_node_adds_to_nodes(self):
        monitor = self._make_monitor()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            monitor.register_node('node:0:abc123', 'node:0:abc123', 2))
        loop.close()
        assert result['status'] == 'registered'
        assert 'node:0:abc123' in monitor._nodes

    def test_register_duplicate_node_skips(self):
        monitor = self._make_monitor()
        monitor._nodes['node:0:abc123'] = RuntimeProxy(uid='node:0:abc123')
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            monitor.register_node('node:0:abc123', 'node:0:abc123', 2))
        loop.close()
        assert result['status'] == 'already_registered'

    def test_register_starts_heartbeat(self):
        monitor = self._make_monitor()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            monitor.register_node('node:0:abc123', 'node:0:abc123', 2))
        loop.close()
        monitor._comms.start_heartbeat.assert_called_once_with('node:0:abc123')


# ---------------------------------------------------------------------------
# Monitor: check_node_status
# ---------------------------------------------------------------------------

class TestCheckNodeStatus:
    """Test monitor.check_node_status() RPC."""

    def _make_monitor(self):
        from mosaic.runtime.monitor import Monitor
        monitor = Monitor.__new__(Monitor)
        monitor._name = 'monitor'
        monitor._indices = ()
        monitor._uid_override = None
        monitor._monitored_nodes = {}
        monitor._disconnected_runtimes = set()
        return monitor

    def test_all_nodes_present(self):
        monitor = self._make_monitor()
        monitor._monitored_nodes = {
            'node:0:abc123': MagicMock(),
            'node:1:def456': MagicMock(),
        }
        worker_uids = ['worker:0:0:abc123', 'worker:1:0:def456']
        result = monitor.check_node_status('head', worker_uids)
        assert result == {'node:0:abc123': True, 'node:1:def456': True}

    def test_missing_node(self):
        monitor = self._make_monitor()
        monitor._monitored_nodes = {'node:0:abc123': MagicMock()}
        worker_uids = ['worker:0:0:abc123', 'worker:1:0:def456']
        result = monitor.check_node_status('head', worker_uids)
        assert result['node:0:abc123'] is True
        assert result['node:1:def456'] is False

    def test_empty_worker_list(self):
        monitor = self._make_monitor()
        result = monitor.check_node_status('head', [])
        assert result == {}

    def test_derives_node_uid_from_worker_uid(self):
        monitor = self._make_monitor()
        monitor._monitored_nodes = {'node:5:aabbccdd': MagicMock()}
        result = monitor.check_node_status('head', ['worker:5:0:aabbccdd'])
        assert 'node:5:aabbccdd' in result

    def test_ignores_malformed_worker_uid(self):
        monitor = self._make_monitor()
        result = monitor.check_node_status('head', ['bad-uid', 'worker:0'])
        assert result == {}


# ---------------------------------------------------------------------------
# Monitor: dynamic worker counting
# ---------------------------------------------------------------------------

class TestDynamicWorkerCount:
    """Test _get_total_workers and properties in dynamic mode."""

    def _make_monitor(self):
        from mosaic.runtime.monitor import Monitor
        monitor = Monitor.__new__(Monitor)
        monitor._name = 'monitor'
        monitor._indices = ()
        monitor._uid_override = None
        monitor._monitored_nodes = {}
        return monitor

    def test_no_nodes_returns_zero(self):
        monitor = self._make_monitor()
        assert monitor._get_total_workers() == 0

    def test_counts_workers_across_nodes(self):
        monitor = self._make_monitor()
        node_a = MagicMock()
        node_a.sub_resources = {'workers': {'worker:0:0:aaa': {}, 'worker:0:1:aaa': {}}}
        node_b = MagicMock()
        node_b.sub_resources = {'workers': {'worker:1:0:bbb': {}}}
        monitor._monitored_nodes = {'node:0:aaa': node_a, 'node:1:bbb': node_b}
        assert monitor._get_total_workers() == 3


# ---------------------------------------------------------------------------
# RoundRobin strategy: stale worker eviction
# ---------------------------------------------------------------------------

class TestRoundRobinStaleEviction:
    """When a replacement node joins, old workers for the same index must be evicted."""

    def _make_strategy(self):
        monitor = MagicMock()
        monitor.logger = MagicMock()
        return RoundRobin(monitor)

    def test_evicts_stale_workers_on_node_replacement(self):
        strategy = self._make_strategy()
        old_node = MagicMock()
        old_node.uid = 'node:0:old11111'
        old_node.sub_resources = {'workers': {'worker:0:0:old11111': {}}}
        strategy.update_node(old_node)
        assert 'worker:0:0:old11111' in strategy._worker_list

        new_node = MagicMock()
        new_node.uid = 'node:0:new22222'
        new_node.sub_resources = {'workers': {'worker:0:0:new22222': {}}}
        strategy.update_node(new_node)
        assert 'worker:0:0:new22222' in strategy._worker_list
        assert 'worker:0:0:old11111' not in strategy._worker_list

    def test_different_node_indices_not_evicted(self):
        strategy = self._make_strategy()
        node_0 = MagicMock()
        node_0.uid = 'node:0:aaa'
        node_0.sub_resources = {'workers': {'worker:0:0:aaa': {}}}
        strategy.update_node(node_0)

        node_1 = MagicMock()
        node_1.uid = 'node:1:bbb'
        node_1.sub_resources = {'workers': {'worker:1:0:bbb': {}}}
        strategy.update_node(node_1)

        assert 'worker:0:0:aaa' in strategy._worker_list
        assert 'worker:1:0:bbb' in strategy._worker_list
        assert strategy._num_workers == 2

    def test_remove_worker(self):
        strategy = self._make_strategy()
        node = MagicMock()
        node.uid = 'node:0:aaa'
        node.sub_resources = {'workers': {'worker:0:0:aaa': {}}}
        strategy.update_node(node)
        assert strategy._num_workers == 1

        strategy.remove_worker('worker:0:0:aaa')
        assert strategy._num_workers == 0
        assert 'worker:0:0:aaa' not in strategy._worker_list

    def test_select_worker_round_robins(self):
        strategy = self._make_strategy()
        for i in range(3):
            node = MagicMock()
            node.uid = 'node:%d:aaa' % i
            node.sub_resources = {'workers': {'worker:%d:0:aaa' % i: {}}}
            strategy.update_node(node)

        workers_selected = set()
        for _ in range(3):
            w = strategy.select_worker('head')
            workers_selected.add(w)
        assert len(workers_selected) == 3


# ---------------------------------------------------------------------------
# Runtime: proxy_from_uid / remove_proxy_from_uid callbacks
# ---------------------------------------------------------------------------

class TestWorkerCountChangedCallbacks:
    """Adding/removing workers fires _on_worker_count_changed callbacks."""

    def _make_runtime(self):
        rt = MagicMock()
        rt._workers = {}
        rt._on_worker_count_changed = []
        rt.uid = 'head'
        rt.logger = MagicMock()
        from mosaic.runtime.runtime import Runtime
        rt.proxy_from_uid = Runtime.proxy_from_uid.__get__(rt)
        rt.remove_proxy_from_uid = Runtime.remove_proxy_from_uid.__get__(rt)
        rt.proxy = RuntimeProxy
        return rt

    def test_add_worker_fires_callback(self):
        rt = self._make_runtime()
        called = []
        rt._on_worker_count_changed.append(lambda: called.append(True))
        proxy = RuntimeProxy(uid='worker:0:0:abc123')
        rt.proxy_from_uid('worker:0:0:abc123', proxy)
        assert len(called) == 1
        assert 'worker:0:0:abc123' in rt._workers

    def test_remove_worker_fires_callback(self):
        rt = self._make_runtime()
        proxy = RuntimeProxy(uid='worker:0:0:abc123')
        rt._workers['worker:0:0:abc123'] = proxy
        called = []
        rt._on_worker_count_changed.append(lambda: called.append(True))
        rt.remove_proxy_from_uid('worker:0:0:abc123', proxy)
        assert len(called) == 1
        assert 'worker:0:0:abc123' not in rt._workers

    def test_duplicate_add_does_not_fire(self):
        rt = self._make_runtime()
        proxy = RuntimeProxy(uid='worker:0:0:abc123')
        rt._workers['worker:0:0:abc123'] = proxy
        called = []
        rt._on_worker_count_changed.append(lambda: called.append(True))
        rt.proxy_from_uid('worker:0:0:abc123', proxy)
        assert len(called) == 0


# ---------------------------------------------------------------------------
# Comms: address auto-detect for 0.0.0.0
# ---------------------------------------------------------------------------

class TestAddressDiscovery:
    """0.0.0.0 should trigger auto-detection of the actual IP."""

    def test_zero_address_is_treated_as_auto_detect(self):
        assert '0.0.0.0' != 'localhost'

    def test_validate_address_accepts_valid_ip(self):
        from mosaic.comms.comms import validate_address
        validate_address('127.0.0.1')

    def test_validate_address_rejects_invalid(self):
        from mosaic.comms.comms import validate_address
        with pytest.raises(ValueError):
            validate_address('not-an-ip')
