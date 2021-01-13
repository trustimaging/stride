
import sys
import asyncio
import zmq
import zmq.asyncio
import tblib
import errno
import psutil
import socket
import warnings
import contextlib
import weakref
from concurrent.futures import CancelledError

import mosaic
from .compression import maybe_compress, decompress
from .serialisation import serialise, deserialise
from ..utils import Future
from ..utils.utils import sizeof


__all__ = ['CommsManager']


_protocol_version = '0.0.0'


def join_address(address, port, protocol='tcp'):
    return '%s://%s:%d' % (protocol, address, port)


def validate_address(address, port=False):
    if type(address) is not str:
        raise ValueError('Address %s is not valid' % (address,))

    if port is False:
        error_msg = 'Address %s is not valid' % (address,)
    else:
        error_msg = 'Address and port combination %s:%d is not valid' % (address, port)

    try:
        socket.inet_pton(socket.AF_INET, address)
    except AttributeError:
        try:
            socket.inet_aton(address)
        except socket.error:
            raise ValueError(error_msg)
    except socket.error:
        raise ValueError(error_msg)

    if port is not False:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError(error_msg)


class CMD:
    def __init__(self, cmd):
        self.type = cmd['type']
        self.uid = cmd['uid']
        self.method = cmd['method']
        self.args = cmd['args']
        self.kwargs = cmd['kwargs']


class Message:
    def __init__(self, sender_id, msg):
        self.method = msg['method']
        self.sender_id = sender_id
        self.runtime_id = msg['runtime_id']
        self.kwargs = msg['kwargs']
        self.reply = msg['reply']

        cmd = msg.get('cmd', {})
        self.cmd = CMD(cmd) if cmd is not None else None


class Reply(Future):
    pass


class Connection:
    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        self._runtime = runtime or mosaic.runtime()
        self._comms = comms or mosaic.get_comms()
        self._loop = loop or mosaic.get_event_loop()
        self._zmq_context = context or mosaic.get_zmq_context()

        self._uid = uid
        self._address = address
        self._port = port
        self._in_node = in_node

        self._socket = None
        self._state = 'disconnected'

    def __repr__(self):
        return "<%s object at %s, address=%s, port=%d, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.address, self.port, self.state)

    @property
    def uid(self):
        return self._uid

    @property
    def address(self):
        return self._address

    @property
    def port(self):
        return self._port

    @property
    def socket(self):
        return self._socket

    @property
    def state(self):
        return self._state

    @property
    def connect_address(self):
        if self._in_node is True:
            return join_address('127.0.0.1', self.port)

        else:
            return join_address(self.address, self.port)

    @property
    def bind_address(self):
        return join_address('*', self.port)

    @property
    def logger(self):
        return self._runtime.logger

    def disconnect(self):
        if self._state != 'connected':
            return

        self._socket.close()
        self._state = 'disconnected'


class InboundConnection(Connection):

    def __init__(self, uid, address, port=None,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        self._socket = self._zmq_context.socket(zmq.ROUTER,
                                                copy_threshold=zmq.COPY_THRESHOLD,
                                                io_loop=self._loop.get_event_loop())

    @property
    def address(self):
        if self._address is None:
            address, port = '8.8.8.8', '53'
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # This command will raise an exception if there is no internet
                # connection.
                s.connect((address, int(port)))
                self._address = s.getsockname()[0]
            except OSError as e:
                self._address = '127.0.0.1'
                # [Errno 101] Network is unreachable
                if e.errno == errno.ENETUNREACH:
                    try:
                        # try get node ip address from host name
                        host_name = socket.getfqdn(socket.gethostname())
                        self._address = socket.gethostbyname(host_name)
                    except Exception:
                        pass
            finally:
                s.close()

        return self._address

    def connect(self):
        if self._state != 'disconnected':
            return

        if self._port is None:
            self._port = 3000
            existing_ports = [each.laddr.port for each in psutil.net_connections()]
            while self._port in existing_ports:
                self._port += 1

        self._socket.bind(self.bind_address)

        self._state = 'connected'

    async def recv(self):
        if self._state == 'disconnected':
            warnings.warn('Trying to receive in a disconnected InboundConnection "%s"' % self.uid, Warning)
            return

        multipart_msg = await self._socket.recv_multipart(copy=False)

        sender_id = multipart_msg[1]
        multipart_msg = multipart_msg[2:]
        num_parts = int(multipart_msg[0])

        if len(multipart_msg) != num_parts:
            raise ValueError('Wrong number of parts')

        sender_id = str(sender_id)
        header = deserialise(multipart_msg[1], [])

        if num_parts > 3:
            compressed_msg = [multipart_msg[2], multipart_msg[3:]]
        else:
            compressed_msg = [multipart_msg[2], []]

        msg = []

        _msg = decompress(header['compression'][0], compressed_msg[0])
        msg.append(_msg)

        _msg = [decompress(compression, payload)
                for compression, payload in zip(header['compression'][1], compressed_msg[1])]
        msg.append(_msg)

        msg = deserialise(msg[0], msg[1])
        msg = Message(sender_id, msg)

        if not msg.method.startswith('log') and not msg.method.startswith('update_monitored_node'):
            if msg.method == 'cmd':
                self.logger.debug('Received cmd %s %s from %s at %s (%s)' % (msg.method, msg.cmd.method,
                                                                             sender_id, self._runtime.uid,
                                                                             msg.cmd.uid))
            else:
                self.logger.debug('Received msg %s from %s at %s' % (msg.method, sender_id, self._runtime.uid))

        return sender_id, msg


class OutboundConnection(Connection):

    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        validate_address(address, port)

        self._socket = self._zmq_context.socket(zmq.DEALER,
                                                copy_threshold=zmq.COPY_THRESHOLD,
                                                io_loop=self._loop.get_event_loop())

        self._heartbeat_timeout = None
        self._heartbeat_attempts = 0
        self._heartbeat_max_attempts = 5
        self._heartbeat_interval = 30

        self._shaken = False

    @property
    def shaken(self):
        return self._shaken

    def connect(self):
        if self._state != 'disconnected':
            return

        self._socket.connect(self.connect_address)
        self.start_heartbeat()

        self._state = 'connected'

    def shake(self):
        self._shaken = True

    def start_heartbeat(self):
        if not self._runtime.is_monitor or not self.uid.startswith('node'):
            return

        if self._heartbeat_timeout is not None:
            self._heartbeat_timeout.cancel()

        self._heartbeat_attempts = self._heartbeat_max_attempts + 1

        self._heartbeat_timeout = self._loop.timeout(self.heart, timeout=self._heartbeat_interval)

    def stop_heartbeat(self):
        if self._heartbeat_timeout is not None:
            self._heartbeat_timeout.cancel()
            self._heartbeat_timeout = None

    async def heart(self):
        self._heartbeat_attempts -= 1

        if self._heartbeat_attempts == 0:
            await self._comms.disconnect(self.uid, self.uid, notify=True)
            await self._loop.run_async(self._runtime.disconnect, args=(self.uid, self.uid))
            return

        interval = self._heartbeat_interval * self._heartbeat_max_attempts/self._heartbeat_attempts
        self._heartbeat_timeout = self._loop.timeout(self.heart, timeout=interval)

        await self.send(method='heart')

    async def beat(self):
        self._heartbeat_attempts = self._heartbeat_max_attempts + 1

        self.stop_heartbeat()
        self.start_heartbeat()

    async def send(self, method, cmd=None, reply=False, **kwargs):
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        if reply is True:
            reply_future = Reply(name=method)
            self._comms.register_reply_future(reply_future)
            reply = reply_future.uid

        else:
            reply_future = None

        msg = {
            'method': method,
            'runtime_id': self.uid,
            'kwargs': kwargs,
            'reply': reply,
            'cmd': cmd,
        }

        if not method.startswith('log') and not method.startswith('update_monitored_node'):
            if method == 'cmd':
                self.logger.debug('Sending cmd %s %s to %s (%s) from %s' % (method, cmd['method'],
                                                                            self.uid, cmd['uid'],
                                                                            self._runtime.uid))
            else:
                self.logger.debug('Sending msg %s to %s from %s' % (method, self.uid,
                                                                    self._runtime.uid))

        msg = serialise(msg)
        msg_size = sizeof(msg)

        compression = []
        compressed_msg = []

        _compression, _compressed_msg = maybe_compress(msg[0])
        compression.append(_compression)
        compressed_msg.append(_compressed_msg)

        if len(msg[1]) > 0:
            _compression, _compressed_msg = zip(*map(maybe_compress, msg[1]))
            compression.append(_compression)
            compressed_msg.append(_compressed_msg)

        else:
            compression.append([])
            compressed_msg.append([])

        header = {
            'version': _protocol_version,
            'compression': compression,
        }

        header = serialise(header)[0]

        multipart_msg = [self._runtime.uid.encode()]
        multipart_msg += [str(3 + len(compressed_msg[1])).encode()]
        multipart_msg += [header]
        multipart_msg += [compressed_msg[0]]
        multipart_msg += compressed_msg[1]

        await self._socket.send_multipart(multipart_msg, copy=msg_size < zmq.COPY_THRESHOLD)

        return reply_future


class CircularConnection(Connection):

    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        self._socket = None
        self._state = 'connected'
        self._shaken = True

    def connect(self):
        return

    async def send(self, method, cmd=None, reply=False, **kwargs):
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        if reply is True:
            reply_future = Reply(name=method)
            self._comms.register_reply_future(reply_future)
            reply = reply_future.uid

        else:
            reply_future = None

        msg = {
            'method': method,
            'runtime_id': self.uid,
            'kwargs': kwargs,
            'reply': reply,
            'cmd': cmd,
        }

        if not method.startswith('log'):
            if method == 'cmd':
                self.logger.debug('Sending cmd %s %s to %s (%s) from %s' % (method, cmd['method'],
                                                                            self.uid, cmd['uid'], self._runtime.uid))
            else:
                self.logger.debug('Sending msg %s to %s from %s' % (method, self.uid, self._runtime.uid))

        msg = Message(self._runtime.uid, msg)

        if not msg.method.startswith('log'):
            if msg.method == 'cmd':
                self.logger.debug('Received cmd %s %s from %s at %s (%s)' % (msg.method, msg.cmd.method,
                                                                             self._runtime.uid, self._runtime.uid,
                                                                             msg.cmd.uid))
            else:
                self.logger.debug('Received msg %s from %s at %s' % (msg.method, self._runtime.uid, self._runtime.uid))

        await self._comms.process_msg(self._runtime.uid, msg)

        return reply_future


class CommsManager:

    _comms_methods = ['hand', 'shake', 'heart', 'beat', 'stop', 'connect', 'disconnect', 'reply']

    def __init__(self, runtime, address=None, port=None, context=None, loop=None):
        self._runtime = runtime or mosaic.runtime()
        self._loop = loop or mosaic.get_event_loop()
        self._zmq_context = context or mosaic.get_zmq_context()

        self._recv_socket = InboundConnection(self._runtime.uid, address, port,
                                              runtime=self._runtime,
                                              comms=self,
                                              in_node=False,
                                              context=self._zmq_context,
                                              loop=self._loop)
        self._recv_socket.socket.setsockopt(zmq.IDENTITY, self._runtime.uid.encode())
        self._recv_socket.socket.setsockopt(zmq.RCVHWM, 0)

        self._send_socket = dict()
        self._circ_socket = CircularConnection(self._runtime.uid, self.address, self.port,
                                               runtime=self._runtime,
                                               comms=self,
                                               in_node=False,
                                               context=self._zmq_context,
                                               loop=self._loop)

        self._listen_future = None
        self._reply_futures = weakref.WeakValueDictionary()
        self._reply_futures = dict()

        self._state = 'disconnected'

    def __repr__(self):
        return "<CommsManager object at %s, uid=%s, address=%s, port=%d, state=%s>" % \
               (id(self), self._runtime.uid, self._recv_socket.address, self._recv_socket.port, self._state)

    def __await__(self):
        if self._listen_future is None:
            raise RuntimeError('Cannot wait for comms that has not started listening')

        future = self._loop.wrap_future(self._listen_future)
        return (yield from future.__await__())

    def wait(self):
        if self._listen_future is None:
            raise RuntimeError('Cannot wait for comms that has not started listening')

        try:
            self._listen_future.result()

        except CancelledError:
            pass

    @property
    def address(self):
        return self._recv_socket.address

    @property
    def port(self):
        return self._recv_socket.port

    @property
    def logger(self):
        return self._runtime.logger

    def uid_address(self, uid):
        return self._send_socket[uid].address

    def uid_port(self, uid):
        return self._send_socket[uid].port

    def connect_recv(self):
        if self._state != 'disconnected':
            return

        self._recv_socket.connect()
        self._circ_socket.connect()

        self._state = 'connected'

    def connect_send(self, uid, address, port):
        validate_address(address, port)

        if uid not in self._send_socket.keys() and uid != self._runtime.uid:
            self._send_socket[uid] = OutboundConnection(uid, address, port,
                                                        runtime=self._runtime,
                                                        comms=self,
                                                        in_node=False,
                                                        context=self._zmq_context,
                                                        loop=self._loop)
            self._send_socket[uid].socket.setsockopt(zmq.IDENTITY, self._runtime.uid.encode())
            self._send_socket[uid].socket.setsockopt(zmq.SNDHWM, 0)
            self._send_socket[uid].connect()

    def connected(self, uid):
        return uid in self._send_socket.keys() or uid == self._runtime.uid

    def shaken(self, uid):
        return self.connected(uid) and self._send_socket[uid].shaken

    def disconnect_recv(self):
        self._recv_socket.socket.close()

    def disconnect_send(self):
        for sender_id, connection in self._send_socket.items():
            connection.socket.close()

    def send(self, *args, **kwargs):
        wait = kwargs.pop('wait', True)

        return self._loop.run(self.send_async, args=args, kwargs=kwargs, wait=wait)

    def cmd(self, *args, **kwargs):
        wait = kwargs.pop('wait', True)

        return self._loop.run(self.cmd_async, args=args, kwargs=kwargs, wait=wait)

    def recv(self, **kwargs):
        wait = kwargs.pop('wait', True)

        return self._loop.run(self.recv_async, wait=wait)

    def send_recv(self, *args, **kwargs):
        wait = kwargs.pop('wait', True)
        kwargs['reply'] = True

        future = self._loop.run(self.send_async, args=args, kwargs=kwargs,
                                wait=True)

        if wait is True:
            return future.result()

        else:
            return future

    def cmd_recv(self, *args, **kwargs):
        wait = kwargs.pop('wait', True)
        kwargs['reply'] = True

        future = self._loop.run(self.cmd_async, args=args, kwargs=kwargs,
                                wait=True)

        if wait is True:
            return future.result()

        else:
            return future

    def reply(self, sender_id, uid, result):
        if uid not in self._reply_futures.keys():
            return

        self._reply_futures[uid].set_result(result)

    def register_reply_future(self, future):
        self._reply_futures[future.uid] = future

    def listen(self):
        if self._state != 'connected':
            return

        def done(fut):
            try:
                exception = fut.exception()

            except Exception:
                return

            if exception is not None:
                raise exception

        self._listen_future = self._loop.run(self.listen_async)
        self._listen_future.add_done_callback(done)

        return self._listen_future

    async def listen_async(self):
        if self._state != 'connected':
            return

        self._state = 'listening'

        self.logger.info('Listening at %s' % self)

        while True:
            sender_id, msg = await self.recv_async()
            await self.process_msg(sender_id, msg)

            if msg.method == 'stop':
                break

    async def process_msg(self, sender_id, msg):
        runtime = self._runtime
        method = getattr(runtime, msg.method, False)
        comms_method = getattr(self, msg.method, False)

        await self.beat(sender_id)

        if msg.method.startswith('raise') or msg.method.startswith('stop'):
            call = self.call
        else:
            call = self.call_safe

        async with self.send_exception(sender_id):
            if msg.method not in self._comms_methods:
                if method is False:
                    raise AttributeError('Class %s does not have method %s' % (runtime.__class__.__name__,
                                                                               msg.method))

                if not callable(method):
                    raise ValueError('Method %s of class %s is not callable' % (msg.method,
                                                                                runtime.__class__.__name__))

        if method is not False:
            if msg.cmd is not None:
                msg.kwargs['cmd'] = msg.cmd

            future = self._loop.run_async(call,
                                          args=(sender_id, method, msg.reply),
                                          kwargs=msg.kwargs)

            if comms_method is not False:
                await future

        if comms_method is not False and msg.method in self._comms_methods:
            self._loop.run_async(call,
                                 args=(sender_id, comms_method, False),
                                 kwargs=msg.kwargs)

    async def call(self, sender_id, method, reply, **kwargs):
        args = (sender_id,)

        await self._loop.run_async(method, args=args, kwargs=kwargs)

    async def call_safe(self, sender_id, method, reply, **kwargs):
        args = (sender_id,)

        async with self.send_exception(sender_id):
            future = self._loop.run_async(method, args=args, kwargs=kwargs)
            result = await future

            if reply is not False:
                await self.send_async(sender_id,
                                      method='reply',
                                      uid=reply, result=result)

    async def send_async(self, send_uid, *args, **kwargs):
        if send_uid == self._runtime.uid:
            return await self._circ_socket.send(*args, **kwargs)

        if send_uid not in self._send_socket.keys():
            raise KeyError('Endpoint %s is not connected' % send_uid)

        return await self._send_socket[send_uid].send(*args, **kwargs)

    async def cmd_async(self, *args, **kwargs):
        cmd = {
            'type': kwargs.pop('type'),
            'uid': kwargs.pop('uid'),
            'method': kwargs.pop('method'),
            'args': kwargs.pop('args', ()),
            'kwargs': kwargs.pop('kwargs', {}),
        }

        return await self.send_async(*args, method='cmd', cmd=cmd, **kwargs)

    async def recv_async(self):
        sender_id, msg = await self._recv_socket.recv()

        return sender_id, msg

    async def send_recv_async(self, send_uid, *args, **kwargs):
        if send_uid == self._runtime.uid:
            future = await self._circ_socket.send(*args, reply=True, **kwargs)

        else:
            if send_uid not in self._send_socket.keys():
                raise KeyError('Endpoint %s is not connected' % send_uid)

            future = await self._send_socket[send_uid].send(*args, reply=True, **kwargs)

        return await future

    async def cmd_recv_async(self, *args, **kwargs):
        cmd = {
            'type': kwargs.pop('type'),
            'uid': kwargs.pop('uid'),
            'method': kwargs.pop('method'),
            'args': kwargs.pop('args', ()),
            'kwargs': kwargs.pop('kwargs', {}),
        }

        future = await self.send_recv_async(*args, method='cmd', cmd=cmd, **kwargs)

        return future

    @contextlib.asynccontextmanager
    async def send_exception(self, uid):
        try:
            yield

        except Exception:
            et, ev, tb = sys.exc_info()
            tb = tblib.Traceback(tb)

            await self.send_async(uid,
                                  method='raise_exception',
                                  exc=(et, ev, tb))

        finally:
            pass

    async def connect(self, sender_id, uid, address, port, notify=False):
        self.connect_send(uid, address, port)

        if notify is True:
            for connected_id, connection in self._send_socket.items():
                await self.send_async(connected_id,
                                      method='connect',
                                      uid=uid, address=address, port=port)

    async def wait_for(self, uid):
        while uid not in self._send_socket.keys() and uid != self._runtime.uid:
            await asyncio.sleep(0.1)

    async def disconnect(self, sender_id, uid, notify=False):
        if uid in self._send_socket.keys():
            self._send_socket[uid].disconnect()

        if notify is True:
            for connected_id, connection in self._send_socket.items():
                await self.send_async(connected_id,
                                      method='disconnect',
                                      uid=uid)

    async def handshake(self, uid, address, port):
        validate_address(address, port)

        self.connect_send(uid, address, port)
        self._runtime.connect(uid, uid, address, port)

        await self.send_async(uid,
                              method='hand',
                              address=self._recv_socket.address, port=self._recv_socket.port)

        while True:
            sender_id, response = await self.recv_async()

            if uid == sender_id and response.method == 'shake':
                break

        await self.shake(sender_id, **response.kwargs)
        await self._loop.run_async(self._runtime.shake, args=(sender_id,), kwargs=response.kwargs)

        self._send_socket[uid].shake()

    async def hand(self, sender_id, address, port):
        for connected_id, connection in self._send_socket.items():
            await self.send_async(connected_id,
                                  method='connect',
                                  uid=sender_id, address=address, port=port)

        self.connect_send(sender_id, address, port)

        network = {}
        for connected_id, connection in self._send_socket.items():
            network[connected_id] = (connection.address, connection.port)

        await self.send_async(sender_id,
                              method='shake',
                              network=network)

    async def shake(self, sender_id, network):
        for uid, address in network.items():
            self.connect_send(uid, *address)

            if uid in self._send_socket:
                self._send_socket[uid].shake()

    async def heart(self, sender_id):
        await self.send_async(sender_id,
                              method='beat')

    async def beat(self, sender_id):
        if sender_id not in self._send_socket.keys():
            return

        await self._send_socket[sender_id].beat()

    async def stop(self, sender_id):
        self._listen_future.cancel()

        self.disconnect_send()
        self.disconnect_recv()
        self._zmq_context.term()
        self._loop.stop()

        self._state = 'disconnected'
