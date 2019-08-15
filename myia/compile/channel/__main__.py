"""Entry point for RPC processes."""

from myia.utils.serialize import MyiaDumper, MyiaLoader
import sys
import importlib

from .handle import LocalHandle, _dead_handle


def _handle_oob(msg):
    if msg[0] == 'dead_handle':
        _dead_handle(msg[1])


def _rpc_server():
    loader = MyiaLoader(sys.stdin.buffer)
    dumper = MyiaDumper(sys.stdout.buffer)
    dumper.open()
    pkg, name, init_args = loader.get_data()
    mod = importlib.import_module(pkg)
    cls = getattr(mod, name)
    iface = cls(**init_args)

    while loader.check_data():
        data = loader.get_data()
        if isinstance(data, tuple):
            name, args, kwargs = data
            meth = getattr(iface, name)
            res = meth(*args, **kwargs)
            dumper.represent(res)
        elif isinstance(data, dict):
            data = data['oob']
            _handle_oob(data)
        else:
            raise TypeError(f"bad message {data}")


_rpc_server()
