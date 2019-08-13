"""Entry point for RPC processes."""

from myia.utils.serialize import MyiaDumper, MyiaLoader
import sys
import importlib


def _rpc_server():
    loader = MyiaLoader(sys.stdin.buffer)
    dumper = MyiaDumper(sys.stdout.buffer)
    dumper.open()
    pkg, name, init_args = loader.get_data()
    mod = importlib.import_module(pkg)
    cls = getattr(mod, name)
    iface = cls(**init_args)

    while loader.check_data():
        name, args, kwargs = loader.get_data()
        meth = getattr(iface, name)
        res = meth(*args, **kwargs)
        dumper.represent(res)


_rpc_server()
