"""RPC channel for multi-process communication."""
import subprocess
from myia.utils.serialize import MyiaDumper, MyiaLoader
from myia.utils import serializable
import sys
import os
import weakref


_local_handle_table = {}
_remote_handle_table = weakref.WeakValueDictionary()


def _dead_handle(id):
    if id in _local_handle_table:
        del _local_handle_table[id]


def _delete_remote(id, channel):
    channel._send_msg('dead_handle', id)


@serializable('channel-lhandle', scalar=True)
class LocalHandle:
    def __init__(self, value):
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
    current_channel = None

    def __init__(self, id):
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
        return self.channel.call_handle(self, args, kwargs)


def handle(value):
    h = _local_handle_table.get(id(value), None)
    if h is None:
        h = LocalHandle(value)
    return h


class RPCProcess:
    def __init__(self, module, cls, init_args, *, interpreter=sys.executable):
        env = os.environ.copy()
        env.update(dict(REMOTE_PDB='1',
                        PYTHONBREAKPOINT='rpdb.set_trace'))
        self.proc = subprocess.Popen(
            [interpreter, '-m', 'myia.compile.channel'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            env=env)
        self.dumper = MyiaDumper(self.proc.stdin)
        self.loader = MyiaLoader(self.proc.stdout)
        self.dumper.open()
        self.dumper.represent((module, cls, init_args))

    def call_method(self, name, *args, **kwargs):
        self.dumper.represent((name, args, kwargs))
        return self._read_msg()

    def call_handle(self, handle, args, kwargs):
        self._send_msg('handle_call', (handle, args, kwargs))
        return self._read_msg()

    def _send_msg(self, msg, args):
        try:
            self.dumper.represent([msg, args])
        except BrokenPipeError:
            pass

    def _read_msg(self):
        RemoteHandle.current_channel = self
        try:
            res = self.loader.get_data()
        finally:
            RemoteHandle.current_channel = None
        return res

    def close(self):
        self.proc.terminate()
        self.dumper.close()
        self.loader.close()
        self.dumper.dispose()
        self.loader.dispose()
