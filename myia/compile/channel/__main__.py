"""Entry point for RPC processes."""

from myia.utils.serialize import MyiaDumper, MyiaLoader
import sys
import importlib

from . import LocalHandle, _dead_handle


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
        elif isinstance(data, list):
            msg, arg = data
            if msg == 'dead_handle':
                _dead_handle(arg)
            elif msg == 'handle_call':
                res = arg[0](*arg[1], **arg[2])
                dumper.represent(res)
            else:
                raise ValueError(f"Unknown message: {msg}")
        else:
            raise TypeError(f"bad data {data}")


_rpc_server()
