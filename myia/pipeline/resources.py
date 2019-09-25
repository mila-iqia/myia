"""Pipeline resources."""

from types import FunctionType

import numpy as np

from .. import parser, xtype
from ..abstract import InferenceEngine, type_to_abstract
from ..compile import load_backend
from ..ir import Graph, clone
from ..monomorphize import monomorphize
from ..operations.utils import Operation
from ..utils import (
    MyiaConversionError,
    Partial,
    Partializable,
    overload,
    tracer,
)
from ..vm import VM

#####################
# ConverterResource #
#####################


@overload
def default_convert(env, fn: FunctionType):
    """Default converter for Python types."""
    g = parser.parse(fn)
    if isinstance(g, Graph):
        g = clone(g)
    env.object_map[fn] = g
    return g


@overload  # noqa: F811
def default_convert(env, g: Graph):
    mng = env.resources.manager
    if g._manager is not mng:
        g2 = clone(g)
        env.object_map[g] = g2
        return g2
    else:
        return g


@overload  # noqa: F811
def default_convert(env, seq: (tuple, list)):
    return type(seq)(env(x) for x in seq)


@overload  # noqa: F811
def default_convert(env, x: Operation):
    dflt = x.defaults()
    if 'mapping' in dflt:
        return default_convert(env, dflt['mapping'])
    else:
        raise MyiaConversionError(f"Cannot convert '{x}'")


@overload  # noqa: F811
def default_convert(env, x: object):
    return x


@overload  # noqa: F811
def default_convert(env, x: type):
    try:
        return type_to_abstract(x)
    except KeyError:
        return x


@overload  # noqa: F811
def default_convert(env, x: np.dtype):
    return default_convert(env, xtype.np_dtype_to_type(x.name))


class _Unconverted:
    # This is just used by Converter to delay conversion of graphs associated
    # to operators or methods until they are actually needed.
    def __init__(self, value):
        self.value = value


class ConverterResource(Partializable):
    """Convert a Python object into an object that can be in a Myia graph."""

    def __init__(self, resources, object_map):
        """Initialize a Converter."""
        self.resources = resources
        self.object_map = {}
        for k, v in object_map.items():
            seen = set()
            while v in object_map:
                idv = id(v)
                if idv in seen:  # pragma: no cover
                    raise Exception(f'Operation {v} maps to itself.')
                seen.add(idv)
                v = object_map[v]
            self.object_map[k] = _Unconverted(v)

    def __call__(self, value):
        """Convert a value."""
        try:
            v = self.object_map[value]
            if isinstance(v, _Unconverted):
                v = default_convert(self, v.value)
                self.object_map[value] = v
        except (TypeError, KeyError):
            v = default_convert(self, value)

        if isinstance(v, Graph):
            self.resources.manager.add_graph(v)
        return v


class InferenceResource(Partializable):
    """Performs inference and monomorphization."""

    def __init__(self, resources, constructors, context_class):
        """Initialize an InferenceResource."""
        self.manager = resources.manager
        self.context_class = context_class
        self.constructors = constructors
        self.engine = InferenceEngine(
            resources,
            constructors=self.constructors,
            context_class=self.context_class,
        )

    def infer(self, graph, argspec, outspec=None, clear=False):
        """Perform inference."""
        with tracer('infer',
                    graph=graph,
                    argspec=argspec,
                    outspec=outspec) as tr:
            if clear:
                self.engine.reset()
            rval = self.engine.run(
                graph,
                argspec=tuple(arg['abstract'] if isinstance(arg, dict)
                              else arg for arg in argspec),
                outspec=outspec,
            )
            tr.set_results(output=rval)
            return rval

    def monomorphize(self, context):
        """Perform monomorphization."""
        with tracer('monomorphize',
                    engine=self.engine,
                    context=context) as tr:
            rval = monomorphize(self.engine, context, reuse_existing=True)
            tr.set_results(output=rval)
            return rval

    def renormalize(self, graph, argspec, outspec=None):
        """Perform inference and specialization."""
        _, context = self.infer(graph, argspec, outspec, clear=True)
        return self.monomorphize(context)


class NumpyChecker:
    """Dummy backend used for debug mode."""

    def to_backend_value(self, v, t):
        """Returns v."""
        return v

    def from_backend_value(self, v, t):
        """Returns v."""
        return v


class BackendResource(Partializable):
    """Contains the backend."""

    def __init__(self, resources, name=None, options=None):
        """Initialize a BackendResource.

        Arguments:
            name: (str) the name of the backend to use
            options: (dict) options for the backend

        """
        self.resources = resources
        if name is False:
            self.backend = NumpyChecker()
        else:
            self.backend = load_backend(name, options)

    def compile(self, graph, argspec, outspec):
        """Compile the graph."""
        return self.backend.compile(graph, argspec, outspec)


class DebugVMResource(Partializable):
    """Contains the DebugVM."""

    def __init__(self, resources, implementations):
        """Initialize a DebugVMResource."""
        self.vm = VM(resources.convert,
                     resources.manager,
                     resources.py_implementations,
                     implementations)


class Resources(Partializable):
    """Defines a set of resources shared by the steps of a Pipeline."""

    def __init__(self, **members):
        """Initialize the Resources."""
        self._members = members
        self._inst = {}

    def __getattr__(self, attr):
        if attr in self._inst:
            return self._inst[attr]

        if attr in self._members:
            inst = self._members[attr]
            if isinstance(inst, Partial):
                try:
                    inst = inst.partial(resources=self)
                except TypeError:
                    pass
                inst = inst()
            self._inst[attr] = inst
            return inst

        raise AttributeError(f'No resource named {attr}.')

    def __call__(self):
        """Run the Resources as a pipeline step."""
        return {'resources': self}


__consolidate__ = True
__all__ = [
    'BackendResource',
    'ConverterResource',
    'DebugVMResource',
    'InferenceResource',
    'NumpyChecker',
    'Resources',
    'default_convert',
]
