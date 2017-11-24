
from typing import Tuple as TupleT
import inspect
import textwrap
import ast
from .parse import Parser, Locator, parse_function
from .stx import Symbol, _Assign, python_universe
from .lib import \
    BackedUniverse, StructuralMap, is_struct, \
    UniverseGenerator, UniversePipelineGenerator
from .stx import PythonUniverse
from .ir import SymbolicUniverse, IRUniverse
from .interpret import VMFunction, VMUniverse
from .symbols import object_map
from .impl.main import impl_bank


class CallableVMFunction:
    def __init__(self, vmf, vmu, eu):
        self.argnames = vmf.argnames
        self.vmf = vmf
        self.__myia_vmfunction__ = vmf
        self.vm_universe = vmu
        self.eval_universe = eu

    def __call__(self, *args):
        assert self.eval_universe
        result = self.vm_universe.run(self.vmf, args)
        return self.eval_universe.export_value(result)


class EvaluationUniverse(BackedUniverse):
    def acquire(self, x):
        x = self.parent[x]
        return self.export_value(x)

    def export_value(self, x):
        if isinstance(x, VMFunction):
            return CallableVMFunction(x, self.parent, self)
        elif is_struct(x):
            return StructuralMap(self.export_value)(x)
        else:
            return x


pipeline = UniversePipelineGenerator(
    # TODO: permit future customization of python_universe
    # {'name': 'py', 'generator': UniverseGenerator(PythonUniverse)},
    {'name': 'py', 'generator': lambda: python_universe},
    {'name': 'sy', 'generator': UniverseGenerator(SymbolicUniverse)},
    {'name': 'ir', 'generator': UniverseGenerator(IRUniverse)},
    {'name': 'vm', 'generator': UniverseGenerator(VMUniverse)},
    {'name': 'ev', 'generator': UniverseGenerator(EvaluationUniverse)}
)


standard_configuration = dict(
    sy_object_map = object_map,
    vm_primitives = impl_bank['interp']
)


standard_universe = pipeline.get_universe(**standard_configuration)


class MyiaFunction:
    def __init__(self, fn, **options):
        self.fn = fn
        self.mfn = None
        self.options = {**standard_configuration, **options}
        self.universe = None
        self.__myia_base__ = fn

    def __call__(self, *args):
        if not self.universe:
            self.universe = pipeline.get_universe(**self.options)
        if not self.mfn:
            self.mfn = self.universe[self.fn]
        assert isinstance(self.mfn, CallableVMFunction)
        return self.mfn(*args)

    def configure(self, **config):
        self.options = {**self.options, **config}
        self.mfn = None
        self.universe = None


def myia(fn, **options):
    return MyiaFunction(fn=fn, **options)


def compile(node):
    return standard_universe[node]
