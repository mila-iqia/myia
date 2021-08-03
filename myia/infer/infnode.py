"""Implementation of type inference on Myia graphs."""

import types
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from myia.ir.node import SEQ

from ..abstract import data, utils as autils
from ..abstract.to_abstract import to_abstract, type_to_abstract
from ..ir import Constant, Graph, Node
from ..ir.graph_utils import dfs, succ_deeper
from ..parser import parse
from ..utils.info import enable_debug
from .algo import Inferrer, Require, RequireAll, Unify

inferrers = {}


class Virtual(Node):
    """Node that only has an abstract field."""

    def __init__(self, abs):
        super().__init__()
        self.abstract = abs


@dataclass(frozen=True)
class InferenceFunction:
    """Encapsulates a function to use for inference."""

    fn: types.FunctionType


def inference_function(fn):
    """Create an abstract type with the given function as its interface.

    Arguments:
        fn: A generator function that performs inference from a node. The
            function should take a list of arg nodes, and a Unificator.
    """
    return data.AbstractStructure((), {"interface": InferenceFunction(fn)})


class SpecializedGraph:
    """Represents a graph specialized on a set of input types.

    Arguments:
        base_graph: The canonical graph.
    """

    def __init__(self, base_graph):
        self.base_graph = base_graph
        self.graph = None

    def commit(self, sig):
        """Commit a specialization to the given input type signature.

        A SpecializedGraph can only be committed for a single signature.
        Using the same SpecializedGraph with multiple signatures is an
        error.

        Arguments:
            sig: Input type signature.
        """
        graph = self.base_graph.specialize(sig)
        if self.graph is None:
            self.graph = graph
        assert self.graph is graph


def signature(*arg_types, ret):
    """Create an inference function from a type signature."""
    arg_types = [
        InferenceDefinition.type_to_abstract(argt) for argt in arg_types
    ]

    return_type = InferenceDefinition.type_to_abstract(ret)

    def _infer(node, args, unif, inferrers):
        inp_types = []
        for inp in args:
            inp_types.append((yield Require(inp)))
        assert len(inp_types) == len(arg_types), (
            f"wrong number of arguments, "
            f"expected {len(arg_types)}, got {len(inp_types)}"
        )
        for inp_type, expected_type in zip(inp_types, arg_types):
            autils.unify(expected_type, inp_type, U=unif)
        return autils.reify(return_type, unif=unif.canon)

    return inference_function(_infer)


class InferenceDefinition:
    """Helper class to represent a signature."""

    __slots__ = "arg_types", "ret_type"

    def __init__(self, *arg_types, ret_type):
        self.arg_types = tuple(
            self.type_to_abstract(arg_type) for arg_type in arg_types
        )
        self.ret_type = self.type_to_abstract(ret_type)

    def __str__(self):
        return f"{list(self.arg_types)} -> {self.ret_type}"

    @classmethod
    def type_to_abstract(cls, el):
        """Convert given object to an abstract type."""
        if isinstance(el, (data.AbstractValue, data.Generic)):
            return el
        if not isinstance(el, type):
            el = type(el)
        if isinstance(el, (tuple, list)):
            return data.AbstractStructure(
                [cls.type_to_abstract(typ) for typ in el],
                {"interface": type(el)},
            )
        return type_to_abstract(el)


def dispatch_inferences(*signatures: Sequence):
    """Create an inference function from many type signatures.

    Arguments:
        signatures: a sequence of type signatures.
            Each signature is a sequence of types or abstract values.
            First sequence values are the argument types.
            Last sequence value is the return type.
            Each sequence must contain at least one element (the return type).
    """
    def_map = {}  # type: Dict[int, Dict[Tuple, InferenceDefinition]]
    for sig in signatures:
        if not isinstance(sig, InferenceDefinition):
            *arg_types, ret_type = sig
            sig = InferenceDefinition(*arg_types, ret_type=ret_type)
        def_map.setdefault(len(sig.arg_types), {})[sig.arg_types] = sig

    def inference(node, args, unif, inferrers):
        inp_types = []
        for inp in args:
            inp_types.append((yield Require(inp)))
        inp_types = tuple(inp_types)
        inf_def = def_map.get(len(inp_types), {}).get(inp_types, None)
        if not inf_def:
            raise RuntimeError(
                f"No inference for node: {node}, signature: {list(inp_types)}\n"
                f"Available signatures:\n"
                + (
                    "\n".join(
                        str(dfn)
                        for dct in def_map.values()
                        for dfn in dct.values()
                    )
                )
            )
        for inp_type, expected_type in zip(inp_types, inf_def.arg_types):
            autils.unify(inp_type, expected_type, U=unif)
        return autils.reify(inf_def.ret_type, unif=unif.canon)

    return inference_function(inference)


class Replace:
    """Declares that the current node should be replaced by a new node."""

    def __init__(self, new_node):
        self.new_node = new_node


class InferenceEngine:
    """Inferrer for graph nodes."""

    def __init__(self, inferrers):
        self.inferrers = inferrers

    def __call__(self, node, unif):
        """Infer the type of a node."""
        assert node is not None
        assert not isinstance(node, (data.AbstractValue, data.GenericBase))

        if node.is_constant(Graph):
            spc = SpecializedGraph(node.value)
            node.replace(Constant(spc))
            return inference_function(spc)

        elif node.is_constant() and node.value in self.inferrers:
            return self.inferrers[node.value]

        elif node.is_constant(types.FunctionType):
            with enable_debug():
                spc = SpecializedGraph(parse(node.value))
                ct = Constant(spc)
            node.replace(ct)
            return inference_function(spc)

        elif node.is_constant(SpecializedGraph):
            return inference_function(node.value)

        elif node.is_constant(
            (
                types.BuiltinFunctionType,
                types.MethodWrapperType,
                types.WrapperDescriptorType,
            )
        ):
            raise TypeError(f"No inferrer for {node.value}")

        elif node.is_constant():
            value = node.value
            assert value not in self.inferrers
            return to_abstract(value)

        else:
            fn = yield Require(node.fn)

            # Graph may contain nodes that are not involved in return node.
            # e.g:
            # def f(h, v):
            #     global_universe_setitem(h, v)  # <-- unused in return value
            #     return v
            # Make sure to infer types for such nodes by visiting SEQ nodes.
            if SEQ in node.edges:
                yield Require(node.edges[SEQ].node)

            # if isinstance(fn, data.AbstractFunction):
            #     inp_types = []
            #     for inp in node.inputs:
            #         inp_types.append((yield Require(inp)))
            #     for inp_type, expected_type in zip(inp_types, fn.args):
            #         autils.unify(expected_type, inp_type, U=unif)

            #     return autils.reify(fn.out, unif=unif.canon)

            # fn type inference don't go through inferer logic if
            # node.fn.abstract was already defined, but there may be
            # inference function associated to this already-existing abstract.
            # So, we use a specific function to do all work about getting
            # associated infererrer.
            inf = self._get_inference_function(node.fn, fn)
            if inf:
                partial_types = fn.elements
                if isinstance(inf, SpecializedGraph):
                    arg_types = yield RequireAll(*node.inputs)
                    inf.commit((*partial_types, *arg_types))
                    res = yield Require(inf.graph.return_)
                    return res
                else:
                    partial_args = [Virtual(a) for a in partial_types]
                    res = inf(node, [*partial_args, *node.inputs], unif, self.inferrers)
                    # TODO: Return res if not generator, if we create
                    # inferrers that are not generators
                    assert isinstance(res, types.GeneratorType)
                    curr = None
                    try:
                        while True:
                            instruction = res.send(curr)
                            if isinstance(instruction, Replace):
                                node.replace(instruction.new_node)
                                curr = None
                            else:
                                curr = yield instruction
                    except StopIteration as stop:
                        return stop.value

            elif isinstance(fn, data.AbstractUnion):

                def _ct(opt):
                    ct = Constant(None)
                    ct.abstract = opt
                    return ct

                optnodes = [
                    node.graph.apply(_ct(opt), *node.inputs)
                    for opt in fn.options
                ]
                return (yield Unify(*optnodes))

            elif self.is_abstract_type(fn):
                # Got abstract type, return interface
                return fn.elements[0]

            else:
                raise TypeError("Unknown function", fn)

    @classmethod
    def is_abstract_type(cls, fn: data.AbstractValue):
        """Return True if given abstract is an abstract type."""
        return (
            isinstance(fn, data.AbstractStructure)
            and fn.tracks.interface is type
            and len(fn.elements) == 1
            # and isinstance(fn.elements[0], data.AbstractAtom)
        )

    def _get_inference_function(self, node, fn: data.AbstractValue):
        """Get inference function associated to given node and abstract."""
        if node.is_constant() and node.value in self.inferrers:
            return self.inferrers[node.value].tracks.interface.fn
        if isinstance(fn.tracks.interface, InferenceFunction):
            return fn.tracks.interface.fn
        elif (
            self.is_abstract_type(fn)
            and (typ := fn.elements[0].tracks.interface) in self.inferrers
        ):
            # Got abstract type.
            # If an inference function is set for given type, return it.
            # We may pass here for e.g. if node is a parameter
            # and inferred parameter type is an abstract type:
            # def f(param_type, value):
            #     return param_type(value)
            return self.inferrers[typ].tracks.interface.fn
        return None


def infer_graph(graph, input_types):
    """Run type inference on a graph given a list of input types.

    Arguments:
        graph: The graph to infer through.
        input_types: A tuple of input types.

    Returns:
        A new, specialized graph. The return type is in g.return_.abstract
    """

    eng = InferenceEngine(inferrers)
    g = graph.specialize(input_types)
    inf = Inferrer(eng)

    try:
        inf.run(g.return_)
    except Exception as exc:
        tr = getattr(exc, "myia_trace", None)
        while tr:
            tr.node.abstract = None
            tr = tr.origin
        raise

    for node in dfs(g.return_, succ=succ_deeper):
        if node.is_constant(SpecializedGraph):
            node.replace(Constant(node.value.graph))

        a = node.abstract
        if a is None:
            assert node.is_constant()
        else:
            node.abstract = autils.reify(a, unif=inf.unif.canon)
        assert not autils.get_generics(node.abstract)

    return g
