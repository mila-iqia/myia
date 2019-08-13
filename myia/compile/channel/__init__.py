"""RPC channel for multi-process communication."""
import subprocess
from myia.utils.serialize import MyiaDumper, MyiaLoader
import sys


class RPCProcess:
    def __init__(self, module, cls, init_args, *, interpreter=sys.executable):
        self.proc = subprocess.Popen(
            [interpreter, '-m', 'myia.compile.channel'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
            #stderr=subprocess.PIPE)
        self.dumper = MyiaDumper(self.proc.stdin)
        self.loader = MyiaLoader(self.proc.stdout)
        self.dumper.open()
        self.dumper.represent((module, cls, init_args))

    def call_method(self, name, *args, **kwargs):
        self.dumper.represent((name, args, kwargs))
        return self.loader.get_data()

    def close(self):
        self.proc.terminate()
        self.dumper.close()
        self.loader.close()
        self.dumper.dispose()
        self.loader.dispose()