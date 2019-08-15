"""RPC channel for multi-process communication."""
import subprocess
from myia.utils.serialize import MyiaDumper, MyiaLoader
from myia.utils import serializable
import sys
import os
import weakref


_local_handle_table = {}
_remote_handle_table = weakref.WeakKeyDictionary()


def _dead_handle(id):
    _local_handle_table[id]._remove_remote()


def _delete_remote(id, channel):
    channel._send_ood(('dead_handle', id))


@serializable('channel-lhandle', scalar=True)
class LocalHandle:
    def __init__(self, value):
        self.value = weakref.ref(value, self._gone)
        self._id = id(value)
        _local_handle_table[self._id] = self

    def _remove_remote(self):
        del self.strong_value

    def _gone(self, *ignored):
        del _handle_table[self._id]
        self._id = None
        self.value = None

    def _serialize(self):
        assert self._id is not None
        if not hasattr(self, 'strong_value'):
            self.strong_value = self.value()
        return self._id

    @classmethod
    def _construct(self, data):
        res = _remote_handle_table.get(id, None)
        if res is None:
            res = RemoteHandle(data)
        return res


@serializable('channel-rhandle', scalar=True)
class RemoteHandle:
    def __init__(self, id):
        self._id = id
        _remote_handle_table[id] = self

    def _serialize(self):
        return self._id

    @classmethod
    def _construct(self, data):
        handle = _local_handle_table[id]
        return handle.strong_value


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
        res = self.loader.get_data()
        if isinstance(res, RemoteHandle):
            weakref.finalize(res, _delete_remote, res._id, self)
        return res

    def _send_oob(self, msg):
        self.dumper.represent({'oob': msg})

    def close(self):
        self.proc.terminate()
        self.dumper.close()
        self.loader.close()
        self.dumper.dispose()
        self.loader.dispose()
