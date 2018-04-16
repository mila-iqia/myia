"""Type inference."""

from typing import Any, Dict as DictT, Set as SetT

from myia.anf_ir import Constant, Parameter, Apply, Graph, ANFNode
from myia.anf_ir_utils import is_constant_graph
from myia.prim import Primitive, SIGNATURES
from myia.dtype import Type, Bool, Float, Int, Tuple, Function
from myia.unify import var as _var, UnificationError, Var

from .graph import Plugin, is_return
from .value import ValuePlugin


class TypeVar(Type):
    """Hide a variable as a type."""

    var: Var

    @property
    def __var__(self):
        return self.var

    def __visit__(self, fn):
        return TypeVar(fn(self.__var__))


def var(*args, **kwargs) -> TypeVar:
    """Proxy to wrap in TypeVar."""
    return TypeVar(_var(*args, **kwargs))


TYPE_MAP: DictT[type, Type] = {
    bool: Bool(),
    float: Float(64),
    int: Int(64),
}


class TypeInferenceError(Exception):
    """Error raised to indicate that there is a type error in the graph."""


def typeof(value) -> Type:
    """Return the Type for a python constant.

    This doesn't support deeply nested or recursive values.
    """
    tt = type(value)
    if tt in TYPE_MAP:
        return TYPE_MAP[tt]
    if tt == tuple:
        etts = tuple(typeof(e) for e in value)
        return Tuple(etts)
    if tt is Primitive:
        return SIGNATURES[value]
    raise TypeError(f"Cannot assign type to: {value}")


class TypePlugin(Plugin):
    """Unification-based type inference."""

    NAME = "type"

    def __init__(self):
        """Create a TypePlugin."""
        self._clones: DictT[Graph, SetT] = dict()

    def on_attach(self):
        """Attach shortcuts."""
        analyzer = self.analyzer
        # Register dependencies
        analyzer.add_plugin(ValuePlugin())

        analyzer.add_shortcut('infer_type', self.infer_type)
        analyzer.add_shortcut('infer_args', self.infer_args)

    def infer_type(self, graph: Graph, *args: Type):
        """Get the return type of a call of `graph`.

        args should be the types of the arguments passed in.
        """
        if graph not in self.analyzer.graphs:
            raise ValueError("Unknown graph")

        equiv = dict(self.analyzer.equiv)
        fn_t = self.analyzer.graphs[graph][self.NAME]
        U = self.analyzer.U
        equiv = U.unify(Tuple(fn_t.arguments), Tuple(args), equiv)

        if equiv is None:
            raise TypeInferenceError("Incompatible apply")

        return U.reify(fn_t.retval, equiv)

    def infer_args(self, graph: Graph, *args: Any):
        """Get the return type of a call of `graph`.

        args here should be a constant value for each argument.
        """
        return self.infer_type(graph, *(typeof(a)
                                        for a in args))

    def get_type(self, node: ANFNode):
        """Get the inferred type for a node."""
        return self.analyzer._info_map[node][self.NAME]

    def on_graph(self, graph: Graph):
        """Return the type of the graph."""
        return Function((self.get_type(p) for p in graph.parameters),
                        TypeVar(var()))

    def on_node(self, node: ANFNode):
        """Compute the type of the node."""
        if isinstance(node, Constant):
            if isinstance(node.value, Graph):
                return self.analyzer.graphs[node.value][self.NAME]
            return typeof(node.value)

        elif isinstance(node, Parameter):
            return var()

        elif isinstance(node, Apply):
            node_t = self.unify_apply(node, self.analyzer.equiv)

            if is_return(node):
                graph_t = Function((self.get_type(p)
                                    for p in node.graph.parameters), node_t)
                self.analyzer.graphs[node.graph][self.NAME] = graph_t
            return node_t

        else:
            raise AssertionError("Unknown node type")

    def unify_apply(self, node: Apply, equiv):
        """Unify an Apply with its function type."""
        U = self.analyzer.U
        fn = node.inputs[0]
        args = node.inputs[1:]
        fn_t = self.get_type(fn)

        if not isinstance(fn_t, Function):
            c_fn_t = Function((var() for a in args), var())
            try:
                U.unify_raw(fn_t, c_fn_t, equiv)
            except UnificationError:
                raise TypeInferenceError("Apply with non-callable")
            fn_t = c_fn_t

        if not isinstance(fn, Parameter):
            fn_t = U.clone(fn_t)
            assert isinstance(fn_t, Function)
            if is_constant_graph(fn):
                self._clones.setdefault(fn.value, set()).add(fn_t)

        args_t = Tuple(self.get_type(a) for a in args)
        equiv = U.unify(Tuple(fn_t.arguments), args_t, equiv)
        if equiv is None:
            raise TypeInferenceError("Apply with incompatible types")

        return U.reify(fn_t.retval, equiv)

    def on_postprocess(self):
        """Refine the graph types until equilibrium."""
        graphs_bak: DictT[Graph, Function] = dict()
        loops = 0
        U = self.analyzer.U

        # Make a copy of the graph type information.
        graphs = dict((k, v[self.NAME])
                      for k, v in self.analyzer.graphs.items())

        while graphs != graphs_bak:
            graphs_bak = graphs.copy()
            loops += 1
            if loops > 100:  # pragma: no cover
                raise Exception("Possible bug in graph type stabilization")

            for g in list(graphs.keys()):
                graphs[g] = U.reify(graphs[g], self.analyzer.equiv)

            for g in self._clones.keys():
                for c in self._clones[g]:
                    res = U.unify(U.clone(graphs[g]), c, self.analyzer.equiv)
                    if res is None:
                        raise TypeInferenceError("Imcompatible applies")

        # Write back the changes to type information in the global map
        for g, v in graphs.items():
            self.analyzer.graphs[g][self.NAME] = v

        for n in list(self.analyzer._info_map.keys()):
            if is_constant_graph(n):
                self.analyzer._info_map[n][self.NAME] = \
                        self.analyzer.graphs[n.value][self.NAME]
            else:
                self.analyzer._info_map[n][self.NAME] = U.reify(
                    self.analyzer._info_map[n][self.NAME],
                    self.analyzer.equiv)
