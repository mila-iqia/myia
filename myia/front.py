
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
from .ir import \
    SymbolicUniverse, IRUniverse, OptimizedUniverse, \
    ResolveGlobalsPass  # , ClosureUnconversionPass, ClosureConversionPass
from .ir.pattern import EquilibriumPass, drop_copy
from .interpret import VMFunction, VMUniverse
from .symbols import object_map
from .impl.main import impl_bank


class CallableVMFunction:
    def __init__(self, vmf, vmu, eu):
        self.argnames = vmf.argnames
        self.vmf = vmf
        self.__myia_graph__ = vmf.graph
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


standard_pipeline = UniversePipelineGenerator(
    const_prop='py->sy->ir->vm->ev',
    full='py->sy->ir->irg->opt->vm->ev',
    # TODO: permit future customization of python_universe
    # py=UniverseGenerator(PythonUniverse)
    py=lambda: python_universe,
    sy=UniverseGenerator(SymbolicUniverse),
    ir=UniverseGenerator(IRUniverse),
    irg=UniverseGenerator(OptimizedUniverse),
    opt=UniverseGenerator(OptimizedUniverse),
    vm=UniverseGenerator(VMUniverse),
    ev=UniverseGenerator(EvaluationUniverse)
)


standard_configuration = dict(
    sy_object_map = object_map,
    vm_primitives = impl_bank['interp'],
    irg_duplicate = True,
    irg_passes = [ResolveGlobalsPass()],
    opt_passes = [
        EquilibriumPass(
            drop_copy
        )
    ]
)


standard_universe = standard_pipeline \
    .get_universes(**standard_configuration)['full']


class MyiaFunction:
    def __init__(self, fn, **options):
        self.fn = fn
        self.mfn = None
        self.options = {**standard_configuration, **options}
        self.universe = None
        self.__myia_base__ = fn

    def __call__(self, *args):
        if not self.universe:
            self.universe = standard_pipeline \
                .get_universes(**self.options)['full']
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
