"""Entry point for RPC processes."""

import importlib
import sys

from myia.utils.serialize import MyiaDumper, MyiaLoader

from . import _dead_handle


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
            try:
                meth = getattr(iface, name)
                res = meth(*args, **kwargs)
            except Exception as e:
                res = e
            dumper.represent(res)
        elif isinstance(data, list):
            msg, arg = data
            if msg == 'dead_handle':
                _dead_handle(arg)
            elif msg == 'handle_call':
                try:
                    res = arg[0](*arg[1], **arg[2])
                except Exception as e:
                    res = e
                dumper.represent(res)
            elif msg == 'send_obj':
                try:
                    dumper.represent(arg)
                except Exception as e:
                    dumper.represent(e)
            else:
                raise ValueError(f"Unknown message: {msg}")
        else:
            raise TypeError(f"bad data {data}")


_rpc_server()
