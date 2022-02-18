
import os
import sys
import uuid
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


__all__ = ['CommsManager', 'get_hostname']


_protocol_version = '0.1'


def join_address(address, port, interface='tcp'):
    return '%s://%s:%d' % (interface, address, port)


def validate_address(address, port=False):
    if type(address) is not str:
        raise ValueError('Address %s is not valid' % (address,))

    if port is False:
        error_msg = 'Address %s is not valid' % (address,)
    else:
        error_msg = 'Address and port combination %s:%d is not valid' % (address, port)

    # Is it an IP address?
    try:
        socket.inet_pton(socket.AF_INET, address)
    except AttributeError:
        try:
            socket.inet_aton(address)
        except socket.error:
            raise ValueError(error_msg)
    except (OSError, socket.error):
        # Could it be a hostname then?
        try:
            socket.gethostbyname(address)
        except socket.gaierror:
            raise ValueError(error_msg)

    if port is not False:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError(error_msg)


def get_hostname():
    return socket.getfqdn(socket.gethostname())


class CMD:
    """
    Container for a CMD for the runtime.

    Parameters
    ----------
    cmd : dict
        Dictionary description of the CMD.

    """

    def __init__(self, cmd):
        self.type = cmd['type']
        self.uid = cmd['uid']
        self.method = cmd['method']
        self.args = cmd['args']
        self.kwargs = cmd['kwargs']


class Message:
    """
    Container for a received message from another comms.

    Parameters
    ----------
    sender_id : str
        Identity of the message sender.
    msg : dict
        Dictionary description of the message.

    """

    def __init__(self, sender_id, msg):
        self.id = msg['id']
        self.method = msg['method']
        self.sender_id = sender_id
        self.runtime_id = msg['runtime_id']
        self.kwargs = msg['kwargs']
        self.reply = msg['reply']

        cmd = msg.get('cmd', {})
        self.cmd = CMD(cmd) if cmd is not None else None


class Reply(Future):
    """
    Future-like object to asynchronously wait for a comms reply.

    """
    pass


class Connection:
    """
    Socket connection through ZMQ.

    Parameters
    ----------
    uid : str
        UID of the current runtime.
    address : str
        IP address of the connection.
    port : int
        Port to use for the connection.
    runtime : Runtime, optional
        Current runtime, defaults to global runtime.
    comms : CommsManager, optional
        Comms to which the connection belongs, defaults to global comms.
    in_node : bool, optional
        Whether the connection is inside the node or not, defaults to False.
    context : zmq.Context, optional
        ZMQ socket context, defaults to global context.
    loop : EventLoop, optional
        Event loop to use, defaults to global event loop.


    """

    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        self._runtime = runtime or mosaic.runtime()
        self._comms = comms or mosaic.get_comms()
        self._loop = loop or mosaic.get_event_loop()
        self._zmq_context = context or mosaic.get_zmq_context()

        self._uid = uid
        self._interface = os.environ.get('MOSAIC_ZMQ_INTERFACE', 'tcp')
        self._address = address
        self._port = port
        self._in_node = in_node

        self._socket = None
        self._sync_socket = None
        self._state = 'disconnected'

    def __repr__(self):
        return "<%s object at %s, interface=%s, address=%s, port=%d, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.interface, self.address, self.port, self.state)

    @property
    def uid(self):
        """
        Runtime UID.

        """
        return self._uid

    @property
    def interface(self):
        """
        Connection interface.

        """
        return self._interface

    @property
    def address(self):
        """
        Connection address.

        """
        return self._address

    @property
    def port(self):
        """
        Connection port.

        """
        return self._port

    @property
    def socket(self):
        """
        Connection ZMQ socket.

        """
        return self._socket

    @property
    def state(self):
        """
        Connection state.

        """
        return self._state

    @property
    def connect_address(self):
        """
        Full formatted address for connection.

        """
        if self._in_node is True:
            return join_address('127.0.0.1', self.port, interface=self.interface)

        else:
            return join_address(self.address, self.port, interface=self.interface)

    @property
    def bind_address(self):
        """
        Full formatted address for binding.

        """
        return join_address('*', self.port, interface=self.interface)

    @property
    def logger(self):
        """
        Runtime logger.

        """
        return self._runtime.logger

    def disconnect(self):
        """
        Disconnect the socket.

        Returns
        -------

        """
        if self._state != 'connected':
            return

        self._socket.close()
        self._sync_socket.close()
        self._state = 'disconnected'


class InboundConnection(Connection):
    """
    Object encapsulating an incoming connection to the CommsManager.

    Parameters
    ----------
    uid : str
        UID of the current runtime.
    address : str
        IP address of the connection.
    port : int
        Port to use for the connection.
    runtime : Runtime, optional
        Current runtime, defaults to global runtime.
    comms : CommsManager, optional
        Comms to which the connection belongs, defaults to global comms.
    in_node : bool, optional
        Whether the connection is inside the node or not, defaults to False.
    context : zmq.Context, optional
        ZMQ socket context, defaults to global context.
    loop : EventLoop, optional
        Event loop to use, defaults to global event loop.

    """

    def __init__(self, uid, address, port=None,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        self._socket = self._zmq_context.socket(zmq.ROUTER,
                                                copy_threshold=zmq.COPY_THRESHOLD,
                                                io_loop=self._loop.get_event_loop())

        self._socket.setsockopt(zmq.IDENTITY, self._runtime.uid.encode())
        self._socket.setsockopt(zmq.RCVHWM, 0)

    @property
    def address(self):
        """
        Connection address.

        If no address is set, it will try to discover it.

        """
        if self._address is None:
            # Try using a hostname first
            self._address = get_hostname()

            try:
                validate_address(self._address)
            except ValueError:
                # Try to find an IP address otherwise
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
                            host_name = get_hostname()
                            self._address = socket.gethostbyname(host_name)
                        except Exception:
                            pass
                finally:
                    s.close()

        return self._address

    def connect(self):
        """
        Connect the socket.

        Returns
        -------

        """
        if self._state != 'disconnected':
            return

        # OSX does not like looking into other processes connections
        try:
            existing_ports = [each.laddr.port for each in psutil.net_connections()]
        except psutil.AccessDenied:
            existing_ports = []

        if self._port is None:
            self._port = 3000

        while self._port in existing_ports:
            self._port += 1

        # If no existing ports were retrieved, we might need to do
        # some trial and error
        while True:
            try:
                self._socket.bind(self.bind_address)
                break
            except zmq.error.ZMQError:
                self._port += 1

        self._sync_socket = zmq.Socket.shadow(self._socket.underlying)

        self._state = 'connected'

    async def recv_async(self):
        """
        Asynchronously receive on the socket.

        Returns
        -------
        str
            Sender UID.
        Message
            Message object.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to receive in a disconnected InboundConnection "%s"' % self.uid, Warning)
            return

        multipart_msg = await self._socket.recv_multipart(copy=False)

        return self._process_rcv(multipart_msg)

    def recv_sync(self):
        """
        Synchronously receive on the socket.

        Returns
        -------
        str
            Sender UID.
        Message
            Message object.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to receive in a disconnected InboundConnection "%s"' % self.uid, Warning)
            return

        multipart_msg = self._sync_socket.recv_multipart(copy=False)

        return self._process_rcv(multipart_msg)

    def _process_rcv(self, multipart_msg):
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
    """
    Object encapsulating an outgoing connection from the CommsManager.

    Parameters
    ----------
    uid : str
        UID of the connected runtime.
    address : str
        IP address of the connection.
    port : int
        Port to use for the connection.
    runtime : Runtime, optional
        Current runtime, defaults to global runtime.
    comms : CommsManager, optional
        Comms to which the connection belongs, defaults to global comms.
    in_node : bool, optional
        Whether the connection is inside the node or not, defaults to False.
    context : zmq.Context, optional
        ZMQ socket context, defaults to global context.
    loop : EventLoop, optional
        Event loop to use, defaults to global event loop.

    """

    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        validate_address(address, port)

        self._socket = self._zmq_context.socket(zmq.DEALER,
                                                copy_threshold=zmq.COPY_THRESHOLD,
                                                io_loop=self._loop.get_event_loop())

        self._socket.setsockopt(zmq.IDENTITY, self._runtime.uid.encode())
        self._socket.setsockopt(zmq.SNDHWM, 0)

        self._heartbeat_timeout = None
        self._heartbeat_attempts = 0
        self._heartbeat_max_attempts = 5
        self._heartbeat_interval = 30

        self._shaken = False

    @property
    def shaken(self):
        """
        Whether or not the handshake has happened.

        """
        return self._shaken

    def connect(self):
        """
        Connect the socket.

        Returns
        -------

        """
        if self._state != 'disconnected':
            return

        self._socket.connect(self.connect_address)
        self._sync_socket = zmq.Socket.shadow(self._socket.underlying)
        self.start_heartbeat()

        self._state = 'connected'

    def shake(self):
        """
        Complete the handshake.

        Returns
        -------

        """
        self._shaken = True

    def start_heartbeat(self):
        """
        Start the heartbeat procedure with the remote endpoint.

        After 5 failed heartbeat attempts, the endpoint is considered disconnected.

        The heartbeat only operates if this is the monitor runtime.

        Returns
        -------

        """
        if not self._runtime.is_monitor or not self.uid.startswith('node'):
            return

        if self._heartbeat_timeout is not None:
            self._heartbeat_timeout.cancel()

        self._heartbeat_attempts = self._heartbeat_max_attempts + 1

        self._heartbeat_timeout = self._loop.timeout(self.heart, timeout=self._heartbeat_interval)

    def stop_heartbeat(self):
        """
        Stop the heartbeat.

        Returns
        -------

        """
        if self._heartbeat_timeout is not None:
            self._heartbeat_timeout.cancel()
            self._heartbeat_timeout = None

    async def heart(self):
        """
        Send heart signal

        Returns
        -------

        """
        self._heartbeat_attempts -= 1

        if self._heartbeat_attempts == 0:
            await self._comms.disconnect(self.uid, self.uid, notify=True)
            await self._loop.run(self._runtime.disconnect, self.uid, self.uid)
            return

        interval = self._heartbeat_interval * self._heartbeat_max_attempts/self._heartbeat_attempts
        self._heartbeat_timeout = self._loop.timeout(self.heart, timeout=interval)

        await self.send(method='heart')

    async def beat(self):
        """
        Process beat signal

        Returns
        -------

        """
        self._heartbeat_attempts = self._heartbeat_max_attempts + 1

        self.stop_heartbeat()
        self.start_heartbeat()

    async def send_async(self, method, cmd=None, reply=False, **kwargs):
        """
        Send message through the connection.

        Parameters
        ----------
        method : str
            Remote method.
        cmd : dict, optional
            If the method is ``cmd`` a description of the command has to be provided.
        reply : bool, optional
            Whether the connection should wait for a reply, defaults to False.
        kwargs : optional
            Keywird arguments for the remote method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        reply_future, msg_size, multipart_msg = self._process_send(method, cmd=cmd, reply=reply, **kwargs)

        await self._socket.send_multipart(multipart_msg, copy=msg_size < zmq.COPY_THRESHOLD)

        return reply_future

    def send_sync(self, method, cmd=None, reply=False, **kwargs):
        """
        Send synchronous message through the connection.

        Parameters
        ----------
        method : str
            Remote method.
        cmd : dict, optional
            If the method is ``cmd`` a description of the command has to be provided.
        reply : bool, optional
            Whether the connection should wait for a reply, defaults to False.
        kwargs : optional
            Keywird arguments for the remote method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        reply_future, msg_size, multipart_msg = self._process_send(method, cmd=cmd, reply=reply, **kwargs)

        self._sync_socket.send_multipart(multipart_msg, copy=msg_size < zmq.COPY_THRESHOLD)

        return reply_future

    def _process_send(self, method, cmd=None, reply=False, **kwargs):
        if reply is True:
            reply_future = Reply(name=method)
            self._comms.register_reply_future(reply_future)
            reply = reply_future.uid

        else:
            reply_future = None

        msg_id = '%s.%s' % (self._runtime.uid, uuid.uuid4().hex)

        msg = {
            'id': msg_id,
            'method': method,
            'runtime_id': self.uid,
            'kwargs': kwargs,
            'reply': reply,
            'cmd': cmd,
        }

        if not method.startswith('log') and not method.startswith('update_monitored_node'):
            if method == 'cmd':
                method = '%s:%s.%s' % (method, cmd['type'], cmd['method'])

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

        return reply_future, msg_size, multipart_msg


class CircularConnection(Connection):
    """
    Object encapsulating a circular connection to itself.

    Parameters
    ----------
    uid : str
        UID of the current runtime.
    address : str
        IP address of the connection.
    port : int
        Port to use for the connection.
    runtime : Runtime, optional
        Current runtime, defaults to global runtime.
    comms : CommsManager, optional
        Comms to which the connection belongs, defaults to global comms.
    in_node : bool, optional
        Whether the connection is inside the node or not, defaults to False.
    context : zmq.Context, optional
        ZMQ socket context, defaults to global context.
    loop : EventLoop, optional
        Event loop to use, defaults to global event loop.

    """

    def __init__(self, uid, address, port,
                 runtime=None, comms=None, in_node=False, context=None, loop=None):
        super().__init__(uid, address, port,
                         runtime=runtime, comms=comms, in_node=in_node, context=context, loop=loop)

        self._socket = None
        self._state = 'connected'
        self._shaken = True

    def connect(self):
        """
        Connect the socket.

        Returns
        -------

        """
        return

    async def send_async(self, method, cmd=None, reply=False, **kwargs):
        """
        Send message through the connection.

        Parameters
        ----------
        method : str
            Remote method.
        cmd : dict, optional
            If the method is ``cmd`` a description of the command has to be provided.
        reply : bool, optional
            Whether the connection should wait for a reply, defaults to False.
        kwargs : optional
            Keywird arguments for the remote method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        reply_future, msg = self._process_send(method, cmd=cmd, reply=reply, **kwargs)

        await self._comms.process_msg(self._runtime.uid, msg)

        return reply_future

    def send_sync(self, method, cmd=None, reply=False, **kwargs):
        """
        Send synchronous message through the connection.

        Parameters
        ----------
        method : str
            Remote method.
        cmd : dict, optional
            If the method is ``cmd`` a description of the command has to be provided.
        reply : bool, optional
            Whether the connection should wait for a reply, defaults to False.
        kwargs : optional
            Keywird arguments for the remote method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        if self._state == 'disconnected':
            warnings.warn('Trying to send in a disconnected OutboundConnection "%s"' % self.uid, Warning)
            return

        reply_future, msg = self._process_send(method, cmd=cmd, reply=reply, **kwargs)
        process_future = self._loop.run(self._comms.process_msg, self._runtime.uid, msg)

        return reply_future

    def _process_send(self, method, cmd=None, reply=False, **kwargs):
        if reply is True:
            reply_future = Reply(name=method)
            self._comms.register_reply_future(reply_future)
            reply = reply_future.uid

        else:
            reply_future = None

        msg_id = '%s.%s' % (self._runtime.uid, uuid.uuid4().hex)

        msg = {
            'id': msg_id,
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

        return reply_future, msg


class CommsManager:
    """
    Objects of this type manage the connections and message passing between different
    runtimes.

    Parameters
    ----------
    runtime : Runtime, optional
        Current runtime, defaults to global runtime.
    address : str
        IP address of the connection.
    port : int
        Port to use for the connection.
    context : zmq.Context, optional
        ZMQ socket context, defaults to global context.
    loop : EventLoop, optional
        Event loop to use, defaults to global event loop.

    """

    _comms_methods = ['hand', 'shake', 'heart', 'beat', 'stop', 'connect', 'disconnect', 'reply']

    def __init__(self, runtime=None, address=None, port=None, context=None, loop=None):
        self._runtime = runtime or mosaic.runtime()
        self._loop = loop or mosaic.get_event_loop()
        self._zmq_context = context or mosaic.get_zmq_context()

        self._recv_socket = InboundConnection(self._runtime.uid, address, port,
                                              runtime=self._runtime,
                                              comms=self,
                                              in_node=False,
                                              context=self._zmq_context,
                                              loop=self._loop)

        self._send_socket = dict()
        self._circ_socket = CircularConnection(self._runtime.uid, self.address, self.port,
                                               runtime=self._runtime,
                                               comms=self,
                                               in_node=False,
                                               context=self._zmq_context,
                                               loop=self._loop)

        self._listen_future = None
        self._reply_futures = weakref.WeakValueDictionary()

        self._state = 'disconnected'

    def __repr__(self):
        return "<CommsManager object at %s, uid=%s, interface=%s, address=%s, port=%d, state=%s>" % \
               (id(self), self._runtime.uid, self._recv_socket.interface,
                self._recv_socket.address, self._recv_socket.port, self._state)

    def __await__(self):
        if self._listen_future is None:
            raise RuntimeError('Cannot wait for comms that has not started listening')

        future = self._loop.wrap_future(self._listen_future)
        return (yield from future.__await__())

    def wait(self):
        """
        Wait until the listening loop of the comms is done.

        Returns
        -------

        """
        if self._listen_future is None:
            raise RuntimeError('Cannot wait for comms that has not started listening')

        try:
            self._listen_future.result()

        except CancelledError:
            pass

    @property
    def address(self):
        """
        Connection address.

        """
        return self._recv_socket.address

    @property
    def port(self):
        """
        Connection port.

        """
        return self._recv_socket.port

    @property
    def logger(self):
        """
        Runtime logger.

        """
        return self._runtime.logger

    def uid_address(self, uid):
        """
        Find remote address given UID.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------
        str
            Address.

        """
        return self._send_socket[uid].address

    def uid_port(self, uid):
        """
        Find remote port given UID.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------
        int
            Port.

        """
        return self._send_socket[uid].port

    def connect_recv(self):
        """
        Connect inbound connection.

        Returns
        -------

        """
        if self._state != 'disconnected':
            return

        self._recv_socket.connect()
        self._circ_socket.connect()

        self._state = 'connected'

    def connect_send(self, uid, address, port):
        """
        Create and connect outbound connection for a remote runtime,
        with a given address and port.

        Parameters
        ----------
        uid : str
            Remote UID.
        address : str
            Remote address.
        port : int
            Remote port.

        Returns
        -------

        """
        validate_address(address, port)

        if uid not in self._send_socket.keys() and uid != self._runtime.uid:
            self._send_socket[uid] = OutboundConnection(uid, address, port,
                                                        runtime=self._runtime,
                                                        comms=self,
                                                        in_node=False,
                                                        context=self._zmq_context,
                                                        loop=self._loop)
            self._send_socket[uid].connect()

    def connected(self, uid):
        """
        Check whether remote UID is connected.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------

        """
        return uid in self._send_socket.keys() or uid == self._runtime.uid

    def shaken(self, uid):
        """
        Check whether remote UID has completed handshake.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------

        """
        return self.connected(uid) and self._send_socket[uid].shaken

    def disconnect_recv(self):
        """
        Disconnect inbound connection.

        Returns
        -------

        """
        self._recv_socket.disconnect()

    def disconnect_send(self):
        """
        Connect all outbound connections.

        Returns
        -------

        """
        for sender_id, connection in self._send_socket.items():
            connection.disconnect()

    def send(self, *args, **kwargs):
        """
        Synchronously send message to remote runtime.

        For arguments and return values check ``Comms.send_async``.

        """
        return self._send_any(*args, **kwargs, sync=True)

    def cmd(self, *args, **kwargs):
        """
        Synchronously send command to remote runtime.

        For arguments and return values check ``Comms.cmd_async``.

        """
        return self._cmd_any(*args, **kwargs, sync=True)

    def recv(self, **kwargs):
        """
        Synchronously receive message from remote runtime.

        For arguments and return values check ``Comms.recv_async``.

        """
        return self._recv_any(sync=True)

    def send_recv(self, *args, **kwargs):
        """
        Synchronously send message to remote runtime and wait for reply.

        For arguments and return values check ``Comms.send_async``.

        """
        wait = kwargs.pop('wait', True)

        future = self._send_recv_any(*args, **kwargs, sync=True)

        if wait is True:
            return future.result()

        else:
            return future

    def cmd_recv(self, *args, **kwargs):
        """
        Synchronously send command to remote runtime and wait for reply.

        For arguments and return values check ``Comms.cmd_async``.

        """
        wait = kwargs.pop('wait', True)

        future = self._cmd_recv_any(*args, **kwargs, sync=True)

        if wait is True:
            return future.result()

        else:
            return future

    def reply(self, sender_id, uid, result):
        """
        Process reply from remote runtime.

        Parameters
        ----------
        sender_id : str
            UID of the remote endpoint.
        uid : str
            UID of the associated Reply.
        result : object
            Result of the reply.

        Returns
        -------

        """
        if uid not in self._reply_futures.keys():
            return

        self._reply_futures[uid].set_result(result)

    def register_reply_future(self, future):
        """
        Register a Reply to be accessible later on.

        Parameters
        ----------
        future : Reply

        Returns
        -------

        """
        self._reply_futures[future.uid] = future

    def listen(self):
        """
        Start the listening loop.

        Returns
        -------
        concurrent.futures.Future
            Future associated with the running loop.

        """
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
        """
        Asynchronous listening loop.

        The loop waits on messages from the incoming connection, then
        processes them and, if necessary, passes them to the runtime.

        Returns
        -------

        """
        if self._state != 'connected':
            return

        self._state = 'listening'

        self.logger.info('Listening at %s' % self)

        while self._state != 'disconnected':
            sender_id, msg = await self.recv_async()
            await self.process_msg(sender_id, msg)

            if msg.method == 'stop':
                break

    async def process_msg(self, sender_id, msg):
        """
        Process a received message to decide what to do with it.

        Parameters
        ----------
        sender_id : str
            UID of the remote endpoint.
        msg : Message
            Message object.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

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

            method_name = msg.method
            if method_name == 'cmd':
                method_name = '%s:%s.%s' % (method_name, msg.cmd.type, msg.cmd.method)

            future = self._loop.run(call,
                                    sender_id, method, msg.reply,
                                    **msg.kwargs)

            if comms_method is not False:
                await future

        if comms_method is not False and msg.method in self._comms_methods:
            self._loop.run(call,
                           sender_id, comms_method, False,
                           **msg.kwargs)

    async def call(self, sender_id, method, reply, **kwargs):
        """
        Run method in the loop.

        Parameters
        ----------
        sender_id : str
            UID of the remote endpoint.
        method : callable
            Method to execute
        reply : False or str
            Whether a reply is needed and, if so, the UID of the reply.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        args = (sender_id,)

        await self._loop.run(method, *args, **kwargs)

    async def call_safe(self, sender_id, method, reply, **kwargs):
        """
        Run method in the loop, and within an exception handler that
        will process exceptions and send them back to the sender.

        Parameters
        ----------
        sender_id : str
            UID of the remote endpoint.
        method : callable
            Method to execute
        reply : False or str
            Whether a reply is needed and, if so, the UID of the reply.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        args = (sender_id,)

        async with self.send_exception(sender_id):
            future = self._loop.run(method, *args, **kwargs)

            if future is None:
                return

            result = await future

            if reply is not False:
                await self.send_async(sender_id,
                                      method='reply',
                                      uid=reply, result=result)

    async def send_async(self, send_uid, *args, **kwargs):
        """
        Send message to ``sender_id`` with given arguments and keyword arguments.

        Parameters
        ----------
        send_uid : str
            UID of the remote runtime.
        args : tuple, optional
            Any arguments for the message.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        return await self._send_any(send_uid, *args, **kwargs, sync=False)

    async def cmd_async(self, *args, **kwargs):
        """
        Send command with given arguments and keyword arguments.

        Parameters
        ----------
        args : tuple, optional
            Any arguments for the message.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------
        Reply or None
            Depending on whether a reply is expected or not.

        """
        return await self._cmd_any(*args, **kwargs, sync=False)

    async def recv_async(self):
        """
        Wait for received message from the inbound socket.

        Returns
        -------
        str
            Sender UID.
        Message
            Received message.

        """
        return await self._recv_any(sync=False)

    async def send_recv_async(self, send_uid, *args, **kwargs):
        """
        Send message to ``sender_id`` with given arguments and keyword arguments,
        and then wait for the reply.

        Parameters
        ----------
        send_uid : str
            UID of the remote runtime.
        args : tuple, optional
            Any arguments for the message.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------
        object
            Result of the reply

        """
        return await self._send_recv_any(send_uid, *args, **kwargs, sync=False)

    async def cmd_recv_async(self, *args, **kwargs):
        """
        Send command with given arguments and keyword arguments,
        and then wait for the reply.

        Parameters
        ----------
        args : tuple, optional
            Any arguments for the message.
        kwargs : optional
            Keyword arguments for the method.

        Returns
        -------
        object
            Result of the reply

        """
        return await self._cmd_recv_any(*args, **kwargs, sync=False)

    @contextlib.asynccontextmanager
    async def send_exception(self, uid):
        """
        Context manager that handles exceptions by sending them
        back to the ``uid``.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------

        """
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
        """
        Create and connect outbound connection for a remote runtime,
        with a given address and port.

        Parameters
        ----------
        uid : str
            Remote UID.
        address : str
            Remote address.
        port : int
            Remote port.
        notify : bool, optional
            Whether or not to notify others of a new connection, defaults to False.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        self.connect_send(uid, address, port)

        if notify is True:
            for connected_id, connection in self._send_socket.items():
                await self.send_async(connected_id,
                                      method='connect',
                                      uid=uid, address=address, port=port)

    async def wait_for(self, uid):
        """
        Wait until remote endpoint has connected.

        Parameters
        ----------
        uid : str
            Remote UID.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        while uid not in self._send_socket.keys() and uid != self._runtime.uid:
            await asyncio.sleep(0.1)

    async def disconnect(self, sender_id, uid, notify=False):
        """
        Disconnect a remote endpoint.

        Parameters
        ----------
        sender_id : str
            Sender UID.
        uid : str
            Remote UID to disconnect.
        notify : bool, optional
            Whether or not to notify others of the disconnection, defaults to False.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        if uid in self._send_socket.keys():
            self._send_socket[uid].disconnect()

        if notify is True:
            for connected_id, connection in self._send_socket.items():
                await self.send_async(connected_id,
                                      method='disconnect',
                                      uid=uid)

    async def handshake(self, uid, address, port):
        """
        Start handshake with remote ``uid``, located at a certain ``address`` and ``port``.

        Parameters
        ----------
        uid : str
            Remote UID.
        address : str
            Remote address.
        port : int
            Remote port.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

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
        await self._loop.run(self._runtime.shake, sender_id, **response.kwargs)

        self._send_socket[uid].shake()

    async def hand(self, sender_id, address, port):
        """
        Handle incoming handshake.

        Parameters
        ----------
        sender_id : str
            Remote UID.
        address : str
            Remote address.
        port : int
            Remote port.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

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
        """
        Handle confirmation of complete handshake.

        Parameters
        ----------
        sender_id : str
            Remote UID.
        network : dict
            Existing topology of connected sockets.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        for uid, address in network.items():
            self.connect_send(uid, *address)

            if uid in self._send_socket:
                self._send_socket[uid].shake()

    async def heart(self, sender_id):
        """
        Received ``heart`` message, respond with ``beat``.

        Parameters
        ----------
        sender_id : str
            Remote UID.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        await self.send_async(sender_id,
                              method='beat')

    async def beat(self, sender_id):
        """
        Received ``beat`` message, the remote endpoint is alive.

        Parameters
        ----------
        sender_id : str
            Remote UID.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        if sender_id not in self._send_socket.keys():
            return

        await self._send_socket[sender_id].beat()

    async def stop(self, sender_id):
        """
        Stop the CommsManager.

        Parameters
        ----------
        sender_id : str
            Remote UID.

        Returns
        -------

        """
        if self._state == 'disconnected':
            return

        self._listen_future.cancel()

        self.disconnect_send()
        self.disconnect_recv()

        self._state = 'disconnected'

    def _send_any(self, send_uid, *args, **kwargs):
        if self._state == 'disconnected':
            return

        sync = kwargs.pop('sync', False)

        def send_sync():
            if send_uid == self._runtime.uid:
                return self._circ_socket.send_sync(*args, **kwargs)

            if send_uid not in self._send_socket.keys():
                raise KeyError('Endpoint %s is not connected' % send_uid)

            return self._send_socket[send_uid].send_sync(*args, **kwargs)

        async def send_async():
            if send_uid == self._runtime.uid:
                return await self._circ_socket.send_async(*args, **kwargs)

            if send_uid not in self._send_socket.keys():
                raise KeyError('Endpoint %s is not connected' % send_uid)

            return await self._send_socket[send_uid].send_async(*args, **kwargs)

        if sync:
            return send_sync()
        else:
            return send_async()

    def _cmd_any(self, *args, **kwargs):
        if self._state == 'disconnected':
            return

        sync = kwargs.pop('sync', False)

        cmd = {
            'type': kwargs.pop('type'),
            'uid': kwargs.pop('uid'),
            'method': kwargs.pop('method'),
            'args': kwargs.pop('args', ()),
            'kwargs': kwargs.pop('kwargs', {}),
        }

        def cmd_sync():
            return self.send(*args, method='cmd', cmd=cmd, **kwargs)

        async def cmd_async():
            return await self.send_async(*args, method='cmd', cmd=cmd, **kwargs)

        if sync:
            return cmd_sync()
        else:
            return cmd_async()

    def _recv_any(self, sync=False):
        if self._state == 'disconnected':
            return None, None

        def recv_sync():
            sender_id, msg = self._recv_socket.recv_sync()
            return sender_id, msg

        async def recv_async():
            sender_id, msg = await self._recv_socket.recv_async()
            return sender_id, msg

        if sync:
            return recv_sync()
        else:
            return recv_async()

    def _send_recv_any(self, send_uid, *args, **kwargs):
        if self._state == 'disconnected':
            return

        sync = kwargs.pop('sync', False)

        def send_recv_sync():
            if send_uid == self._runtime.uid:
                future = self._circ_socket.send_sync(*args, reply=True, **kwargs)

            else:
                if send_uid not in self._send_socket.keys():
                    raise KeyError('Endpoint %s is not connected' % send_uid)

                future = self._send_socket[send_uid].send_sync(*args, reply=True, **kwargs)

            return future

        async def send_recv_async():
            if send_uid == self._runtime.uid:
                future = await self._circ_socket.send_async(*args, reply=True, **kwargs)

            else:
                if send_uid not in self._send_socket.keys():
                    raise KeyError('Endpoint %s is not connected' % send_uid)

                future = await self._send_socket[send_uid].send_async(*args, reply=True, **kwargs)

            return await future

        if sync:
            return send_recv_sync()
        else:
            return send_recv_async()

    def _cmd_recv_any(self, *args, **kwargs):
        if self._state == 'disconnected':
            return

        sync = kwargs.pop('sync', False)

        cmd = {
            'type': kwargs.pop('type'),
            'uid': kwargs.pop('uid'),
            'method': kwargs.pop('method'),
            'args': kwargs.pop('args', ()),
            'kwargs': kwargs.pop('kwargs', {}),
        }

        def cmd_recv_sync():
            future = self.send_recv(*args, method='cmd', cmd=cmd, **kwargs)
            return future

        async def cmd_recv_async():
            future = await self.send_recv_async(*args, method='cmd', cmd=cmd, **kwargs)
            return future

        if sync:
            return cmd_recv_sync()
        else:
            return cmd_recv_async()
