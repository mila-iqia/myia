"""Entry point for RPC processes."""

import importlib
import sys

from myia.utils.serialize import MyiaDumper, MyiaLoader

from . import _dead_handle


def _rpc_server():
    # Try to prevent other libs from using stdout
    sys.stdout = sys.stderr
    dumper = MyiaDumper(1)
    dumper.open()
    loader = MyiaLoader(0)
    pkg, name, init_args = loader.get_data()
    try:
        mod = importlib.import_module(pkg)
        cls = getattr(mod, name)
        iface = cls(**init_args)
        dumper.represent('ready')
    except Exception as e:
        dumper.represent(e)
        return 1

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
                except Exception as e:  # pragma: no cover
                    res = e
                dumper.represent(res)
            else:
                raise ValueError(f"Unknown message: {msg}")  # pragma: no cover
        else:  # pragma: no cover
            raise TypeError(f"bad data {data}")
    return 0


sys.exit(_rpc_server())
