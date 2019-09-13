"""RPC channel for multi-process communication."""
import os
import subprocess
import sys
import weakref

from myia.utils import serializable
from myia.utils.serialize import LoadedError, MyiaDumper, MyiaLoader

_local_handle_table = {}
_remote_handle_table = weakref.WeakValueDictionary()


def _dead_handle(id):
    if id in _local_handle_table:
        del _local_handle_table[id]


def _delete_remote(id, channel):
    channel._send_msg('dead_handle', id)


@serializable('channel-lhandle', scalar=True)
class LocalHandle:
    """Part of the handle that lives in the local process.

    This is serialized specially so that deserialization gets you a
    RemoteHandle linked to this one.  When sent back it will be
    associated with the passed-in object, not this handle.
    """

    def __init__(self, value):
        """Create a handle for an object."""
        self.value = value
        self._id = id(value)
        _local_handle_table[self._id] = self

    def _serialize(self):
        return str(self._id)

    @classmethod
    def _construct(self, data):
        data = int(data)
        res = _remote_handle_table.get(id, None)
        if res is None:
            res = RemoteHandle(data)
        return res


@serializable('channel-rhandle', scalar=True)
class RemoteHandle:
    """Remote part of a LocalHandle.

    Interacting with this will send messages back through the channel
    it was recieved from to communicate with the real object.  Not
    everything is supported.
    """

    current_channel = None

    def __init__(self, id):
        """Private constructor, you never need to call this.

        `id` is the id of the associated handle in the other process.
        """
        self._id = id
        _remote_handle_table[id] = self
        self.channel = self.current_channel
        weakref.finalize(self, _delete_remote, self._id, self.channel)

    def _serialize(self):
        return str(self._id)

    @classmethod
    def _construct(self, data):
        data = int(data)
        handle = _local_handle_table[data]
        return handle.value

    def __call__(self, *args, **kwargs):
        """Call the remote object associated with this handle."""
        return self.channel.call_handle(self, args, kwargs)


def handle(value):
    """Create a handle for the given value.

    This should be done just before serialization to mark the value as
    local only.  It will be proxied using the handle mechanism.
    """
    h = _local_handle_table.get(id(value), None)
    if h is None:
        h = LocalHandle(value)
    return h


class RPCProcess:
    """Object that represents a remote class in another process."""

    def __init__(self, module, cls, init_args, *, interpreter=sys.executable):
        """Create a subprocess and the object specified by the arguments in it.

        The object can then be interacted remotely with by using call_method.

        Remote debugging is also enabled for the process so that
        breakpoints trigger a remote debugger.
        """
        env = os.environ.copy()
        env.update(dict(REMOTE_PDB='1',
                        PYTHONBREAKPOINT='rpdb.set_trace'))
        self.proc = subprocess.Popen(
            [interpreter, '-m', 'myia.compile.channel'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            env=env)
        self.dumper = MyiaDumper(self.proc.stdin.fileno())
        self.loader = MyiaLoader(self.proc.stdout.fileno())
        self.dumper.open()
        self.dumper.represent((module, cls, init_args))
        try:
            resp = self._read_msg()
        except Exception:
            os.waitpid(-1, 0)
            raise
        assert resp == 'ready'

    def call_method(self, name, *args, **kwargs):
        """Call a method on the remote object."""
        self.dumper.represent((name, args, kwargs))
        return self._read_msg()

    def call_handle(self, handle, args, kwargs):
        """Call the object associated with a handle."""
        self._send_msg('handle_call', (handle, args, kwargs))
        return self._read_msg()

    def _send_msg(self, msg, args):
        self.dumper.represent([msg, args])

    def _read_msg(self):
        RemoteHandle.current_channel = self
        try:
            res = self.loader.get_data()
        finally:
            RemoteHandle.current_channel = None
        if isinstance(res, LoadedError):
            raise res
        return res
