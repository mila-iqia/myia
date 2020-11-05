"""Pipeline resources."""

from types import FunctionType

import numpy as np
from ovld import ovld

from .. import parser, xtype
from ..abstract import InferenceEngine, LiveInferenceEngine, type_to_abstract
from ..compile import load_backend
from ..ir import Graph, clone
from ..monomorphize import Monomorphizer
from ..operations.utils import Operation
from ..opt import LocalPassOptimizer, lib as optlib
from ..utils import MyiaConversionError, Partial, Partializable, tracer
from ..vm import VM
from .pipeline import Pipeline

#####################
# ConverterResource #
#####################


@ovld
def default_convert(env, fn: FunctionType):
    """Default converter for Python types."""
    g = parser.parse(fn)
    rval = env(g)
    env.object_map[fn] = rval
    return rval


@ovld  # noqa: F811
def default_convert(env, g: Graph):
    mng = env.resources.infer_manager
    if g._manager is not mng:
        g2 = clone(g)
        env.object_map[g] = g2
        if env.resources.preresolve:
            opt_pass = Pipeline(
                LocalPassOptimizer(optlib.resolve_globals, name="resolve")
            )
            opt_pass(
                graph=g2,
                resources=env.resources,
                manager=env.resources.infer_manager,
            )
        return g2
    else:
        return g


@ovld  # noqa: F811
def default_convert(env, seq: (tuple, list)):
    return type(seq)(env(x) for x in seq)


@ovld  # noqa: F811
def default_convert(env, x: Operation):
    dflt = x.defaults()
    if "mapping" in dflt:
        return env(dflt["mapping"])
    else:
        raise MyiaConversionError(f"Cannot convert '{x}'")


@ovld  # noqa: F811
def default_convert(env, x: object):
    if hasattr(x, "__to_myia__"):
        return x.__to_myia__()
    else:
        return x


@ovld  # noqa: F811
def default_convert(env, x: type):
    try:
        return type_to_abstract(x)
    except TypeError:
        return x


@ovld  # noqa: F811
def default_convert(env, x: np.dtype):
    return env(xtype.np_dtype_to_type(x.name))


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
                    raise Exception(f"Operation {v} maps to itself.")
                seen.add(idv)
                v = object_map[v]
            self.object_map[k] = _Unconverted(v)

    def __call__(self, value, manage=True):
        """Convert a value."""
        if isinstance(value, Graph) and value.abstract is not None:
            return value

        try:
            v = self.object_map[value]
            if isinstance(v, _Unconverted):
                v = default_convert(self, v.value)
                self.object_map[value] = v
        except (TypeError, KeyError):
            v = default_convert(self, value)

        if manage and isinstance(v, Graph):
            self.resources.infer_manager.add_graph(v)
        return v


class Tracker(Partializable):
    """Track new nodes that require type inference."""

    def __init__(self, resources):
        """Initialize a Tracker."""
        self.todo = set()
        self.manager = resources.opt_manager
        self.activated = False

    def activate(self, force=False):
        """Activate the tracker.

        If the tracker is already activated, this does nothing.
        """
        if force or not self.activated:
            self.manager.events.add_node.register(self._on_add_node)
            self.manager.events.drop_node.register(self._on_drop_node)
            self.manager.events.post_reset.register(
                lambda evt: self.activate(force=True)
            )
            self.activated = True

    def _on_add_node(self, event, node):
        if node.abstract is None:
            self.todo.add(node)

    def _on_drop_node(self, event, node):
        self.todo.discard(node)


class InferenceResource(Partializable):
    """Performs inference and monomorphization."""

    def __init__(self, resources, constructors, max_stack_depth):
        """Initialize an InferenceResource."""
        self.resources = resources
        self.manager = resources.infer_manager
        self.constructors = constructors
        self.max_stack_depth = max_stack_depth
        self.engine = InferenceEngine(
            resources,
            manager=self.manager,
            constructors=self.constructors,
            max_stack_depth=self.max_stack_depth,
        )

    def __call__(self, graph, argspec, outspec=None):
        """Perform inference."""
        with tracer(
            "infer", graph=graph, argspec=argspec, outspec=outspec
        ) as tr:
            with tracer("engine", profile=False) as tr:
                rval = self.engine.run(
                    graph,
                    argspec=tuple(
                        arg["abstract"] if isinstance(arg, dict) else arg
                        for arg in argspec
                    ),
                    outspec=outspec,
                )
            tr.set_results(output=rval)
            return rval


class LiveInferenceResource(Partializable):
    """Performs live inference."""

    def __init__(self, resources, constructors):
        """Initialize a LiveInferenceResource."""
        self.resources = resources
        self.manager = resources.opt_manager
        self.constructors = constructors
        self.live = LiveInferenceEngine(
            resources, constructors=self.constructors, manager=self.manager
        )

    def __call__(self):
        """Perform live inference."""
        tracker = self.resources.tracker
        if not tracker.activated:
            return
        mng = self.resources.opt_manager
        todo = tracker.todo
        while todo:
            nodes = [
                node
                for node in todo
                if node.abstract is None and node in mng.all_nodes
            ]
            todo.clear()
            self.live.run(nodes)


class MonomorphizationResource(Partializable):
    """Performs monomorphization."""

    def __init__(self, resources):
        """Initialize a MonomorphizationResource."""
        self.resources = resources
        self.engine = resources.inferrer.engine
        self.manager = resources.opt_manager
        self.mono = Monomorphizer(resources, self.engine)

    def __call__(self, context):
        """Perform monomorphization."""
        with tracer("monomorphize", engine=self.engine, context=context) as tr:
            rval = self.mono.run(context)
            tr.set_results(output=rval)
            return rval


class Incorporator(Partializable):
    """Resource to integrate a new graph during optimization."""

    def __init__(self, resources):
        """Initialize an Incorporator."""
        self.infer_manager = resources.infer_manager
        self.engine = resources.inferrer.engine
        self.mono = resources.monomorphizer

    def opaque_to_inference(self, node):
        """A graph is opaque to inference if it has a type."""
        g = node.value if node.is_constant_graph() else node.graph
        return g and g.abstract is not None

    def __call__(self, graph, argspec, outspec):
        """Run inferrer and monomorphizer on graph.

        The graph's input types are given in argspec, and the expected output
        type in outspec.
        """
        self.infer_manager.set_opaque_condition(self.opaque_to_inference)
        if not isinstance(graph, Graph):
            sig = graph.make_signature(argspec)
            graph = graph.generate_graph(sig)
        self.infer_manager.add_graph(graph)
        context = self.engine.context_class.empty().add(graph, tuple(argspec))
        self.engine.run_coroutine(
            self.engine.infer_function(graph, argspec, outspec)
        )
        return self.mono(context)


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
            resources: The resources object of the pipeline
            name (str): the name of the backend to use
            options (dict): options for the backend

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
        self.vm = VM(
            resources.convert,
            resources.opt_manager,
            resources.py_implementations,
            implementations,
        )


class Resources(Partializable):
    """Defines a set of resources shared by the steps of a Pipeline."""

    def __init__(self, **members):
        """Initialize the Resources."""
        for attr, inst in members.items():
            if isinstance(inst, Partial):
                try:
                    inst = inst.partial(resources=self)
                except TypeError:
                    pass
                inst = inst()
            setattr(self, attr, inst)


__consolidate__ = True
__all__ = [
    "BackendResource",
    "ConverterResource",
    "DebugVMResource",
    "Incorporator",
    "InferenceResource",
    "NumpyChecker",
    "Resources",
    "Tracker",
    "default_convert",
]
