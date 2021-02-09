
import sys
import logging
from cached_property import cached_property

import mosaic


__all__ = ['LoggerManager', 'clear_logger']


log_level = 'info'


_stdout = sys.stdout
_stderr = sys.stderr


_local_log_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'error': logging.ERROR,
    'warning': logging.WARNING,
}


_remote_log_levels = {
    'info': 'log_info',
    'debug': 'log_debug',
    'error': 'log_error',
    'warning': 'log_warning',
}


class LoggerBase:

    @property
    def runtime(self):
        return mosaic.runtime()

    @property
    def comms(self):
        return mosaic.get_comms()

    @property
    def loop(self):
        return mosaic.get_event_loop()

    def isatty(self):
        return False


class LocalLogger(LoggerBase):
    def __init__(self, logger, log_level=logging.INFO):
        self._logger = logger
        self._log_level = log_level
        self._linebuf = ''

    def write(self, buf, uid=None):
        if uid is None:
            uid = self.runtime.uid
        uid = uid.upper()

        temp_linebuf = self._linebuf + buf
        self._linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self._logger.log(self._log_level, line.rstrip(), extra={'runtime_id': uid})
            else:
                self._linebuf += line

    def flush(self, uid=None):
        if uid is None:
            uid = self.runtime.uid
        uid = uid.upper()

        if self._linebuf != '':
            self._logger.log(self._log_level, self._linebuf.rstrip(), extra={'runtime_id': uid})
        self._linebuf = ''


class RemoteLogger(LoggerBase):
    def __init__(self, runtime_id, log_level='log_info'):
        self._runtime_id = runtime_id
        self._log_level = log_level
        self._linebuf = ''

    @cached_property
    def remote_runtime(self):
        return self.runtime.proxy(self._runtime_id)

    def write(self, buf, uid=None):
        if buf == '\n':
            return

        temp_linebuf = self._linebuf + buf
        self._linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                continue

            elif line.rstrip() == '':
                continue

            else:
                self._linebuf += line
                self._linebuf += '\n'

        self.send(self._linebuf)
        self._linebuf = ''

    def flush(self):
        if self._linebuf != '':
            self.send(self._linebuf)

        self._linebuf = ''

    def send(self, buf):
        if not self.comms.shaken(self._runtime_id):
            _stdout.write(buf)
            _stdout.flush()

        else:
            self.remote_runtime[self._log_level](buf=buf, as_async=False)


class LoggerManager:
    """
    Class that manages the creation loggers and the interface with them. It creates
    local or remote loggers and handles the communication with loggers at different
    levels ``info``, ``debug``, ``error`` and ``warning``.

    """

    def __init__(self):
        self._info_logger = None
        self._debug_logger = None
        self._error_logger = None
        self._warn_logger = None

        self._stdout = _stdout
        self._stderr = _stderr

        self._log_level = 'info'
        self._log_location = None

    def set_local(self):
        """
        Set up local loggers.

        Returns
        -------

        """
        self._log_location = 'local'

        sys.stdout = self._stdout
        sys.stderr = self._stderr

        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)-10s %(runtime_id)-15s %(message)s'))

        logger = logging.getLogger('STDOUT')
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(handler)

        self._info_logger = LocalLogger(logger, log_level=_local_log_levels['info'])
        self._debug_logger = LocalLogger(logger, log_level=_local_log_levels['debug'])
        self._error_logger = LocalLogger(logger, log_level=_local_log_levels['error'])
        self._warn_logger = LocalLogger(logger, log_level=_local_log_levels['warning'])

        sys.stdout.flush()
        sys.stdout = self._info_logger
        sys.stderr = self._error_logger

        logging.basicConfig(
            stream=sys.stdout,
            level=_local_log_levels[log_level],
            format='%(message)s',
            # format='%(asctime)s - %(levelname)-26s %(message)s',
        )

    def set_remote(self, runtime_id='monitor'):
        """
        Set up remote loggers.

        Parameters
        ----------
        runtime_id : str, optional
            Runtime to which logging will be directed, defaults to ``monitor``.

        Returns
        -------

        """
        self._log_location = 'remote'

        sys.stdout = self._stdout
        sys.stderr = self._stderr

        self._info_logger = RemoteLogger(runtime_id=runtime_id, log_level=_remote_log_levels['info'])
        self._debug_logger = RemoteLogger(runtime_id=runtime_id, log_level=_remote_log_levels['debug'])
        self._error_logger = RemoteLogger(runtime_id=runtime_id, log_level=_remote_log_levels['error'])
        self._warn_logger = RemoteLogger(runtime_id=runtime_id, log_level=_remote_log_levels['warning'])

        sys.stdout.flush()
        sys.stdout = self._info_logger
        sys.stderr = self._error_logger

    @staticmethod
    def set_level(level):
        """
        Set log level from options ``info``, ``debug``, ``error`` and ``warning``.

        Parameters
        ----------
        level : str
            Log level

        Returns
        -------

        """
        global log_level
        log_level = level

        logger = logging.getLogger('STDOUT')
        logger.setLevel(_local_log_levels[level])

    def info(self, buf, uid=None):
        """
        Log message with level ``info``.

        Parameters
        ----------
        buf : str
            Message to log.
        uid : str, optional
            UID of the runtime from which the message originates, defaults to
            current runtime.

        Returns
        -------

        """
        if self._info_logger is None:
            return

        if log_level in ['error']:
            return

        self._info_logger.write(buf, uid=uid)

        if self._log_location == 'local':
            self._info_logger.flush(uid=uid)

    def debug(self, buf, uid=None):
        """
        Log message with level ``debug``.

        Parameters
        ----------
        buf : str
            Message to log.
        uid : str, optional
            UID of the runtime from which the message originates, defaults to
            current runtime.

        Returns
        -------

        """
        if self._debug_logger is None:
            return

        if log_level in ['info', 'error']:
            return

        self._debug_logger.write(buf, uid=uid)

        if self._log_location == 'local':
            self._debug_logger.flush(uid=uid)

    def error(self, buf, uid=None):
        """
        Log message with level ``error``.

        Parameters
        ----------
        buf : str
            Message to log.
        uid : str, optional
            UID of the runtime from which the message originates, defaults to
            current runtime.

        Returns
        -------

        """
        if self._error_logger is None:
            return

        self._error_logger.write(buf, uid=uid)

        if self._log_location == 'local':
            self._error_logger.flush(uid=uid)

    def warning(self, buf, uid=None):
        """
        Log message with level ``warning``.

        Parameters
        ----------
        buf : str
            Message to log.
        uid : str, optional
            UID of the runtime from which the message originates, defaults to
            current runtime.

        Returns
        -------

        """
        if self._warn_logger is None:
            return

        self._warn_logger.write(buf, uid=uid)

        if self._log_location == 'local':
            self._warn_logger.flush(uid=uid)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'runtime_id'):
            record.runtime_id = ''

        return super().format(record)


def clear_logger():
    sys.stdout.flush()
    sys.stderr.flush()

    sys.stdout = _stdout
    sys.stderr = _stderr
