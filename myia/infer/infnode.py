import operator
import types
from collections import defaultdict
from dataclasses import dataclass

from .. import basics
from ..abstract import data, utils as autils
from ..abstract.to_abstract import to_abstract, type_to_abstract
from ..ir import Constant, Graph
from ..parser import parse
from ..utils.info import enable_debug
from ..utils.misc import ModuleNamespace
from .algo import Require, RequireAll, Unify, infer

X = data.Generic("x")


@dataclass(frozen=True)
class InferenceFunction:
    fn: types.FunctionType


def inference_function(fn):
    return data.AbstractAtom({"interface": InferenceFunction(fn)})


class SpecializedGraph:
    def __init__(self, base_graph):
        self.base_graph = base_graph
        self.graph = None

    def commit(self, sig):
        graph = self.base_graph.specialize(sig)
        if self.graph is None:
            self.graph = graph
        elif self.graph is not graph:
            raise TypeError()


def resolve(args, unif):
    ns, name = args
    assert ns.is_constant(ModuleNamespace)
    assert name.is_constant(str)
    resolved = ns.value[name.value]
    ct = Constant(resolved)
    yield Replace(ct)
    res = yield Require(ct)
    return res


def user_switch(args, unif):
    cond, ift, iff = args
    _ = yield Require(cond)  # TODO: check bool
    ift_t = yield Require(ift)
    iff_t = yield Require(iff)
    return data.AbstractUnion([ift_t, iff_t], tracks={})


def signature(*arg_types, ret):
    arg_types = [
        type_to_abstract(argt) if isinstance(argt, type) else argt
        for argt in arg_types
    ]

    return_type = type_to_abstract(ret) if isinstance(ret, type) else ret

    def _infer(inputs, unif):
        inp_types = []
        for inp in inputs:
            inp_types.append((yield Require(inp)))
        for inp_type, expected_type in zip(inp_types, arg_types):
            autils.unify(expected_type, inp_type, U=unif)
        return autils.reify(return_type, unif=unif.canon)

    return inference_function(_infer)


class Handler:
    pass


inferrers = {
    operator.mul: signature(X, X, ret=X),
    operator.add: signature(X, X, ret=X),
    operator.sub: signature(X, X, ret=X),
    operator.neg: signature(X, ret=X),
    operator.le: signature(X, X, ret=bool),
    operator.truth: signature(X, ret=bool),
    basics.return_: signature(X, ret=X),
    basics.resolve: inference_function(resolve),
    basics.user_switch: inference_function(user_switch),
    type: signature(
        X,
        ret=data.AbstractStructure([X], tracks={"interface": type}),
    ),
    basics.make_handle: signature(
        data.AbstractStructure([X], tracks={"interface": type}),
        ret=data.AbstractStructure([X], tracks={"interface": Handler}),
    ),
    basics.global_universe_getitem: signature(
        data.AbstractStructure([X], tracks={"interface": Handler}),
        ret=X,
    ),
}


class Replace:
    def __init__(self, new_node):
        self.new_node = new_node


class InferenceEngine:
    def __init__(self):
        self.replacements = defaultdict(dict)
        # Make sure None entry is present.
        self.replacements[None] = {}

    def __call__(self, node, unif):
        if repl := self.replacements.get((None, None, node), None):
            return (yield Require(repl))

        assert node is not None
        assert not isinstance(node, (data.AbstractValue, data.GenericBase))

        if node.is_constant(Graph):
            spc = SpecializedGraph(node.value)
            self.replacements[None][None, None, node] = Constant(spc)
            return inference_function(spc)

        elif node.is_constant() and node.value in inferrers:
            return inferrers[node.value]

        elif node.is_constant(types.FunctionType):
            with enable_debug():
                spc = SpecializedGraph(parse(node.value))
                ct = Constant(spc)
            self.replacements[None][None, None, node] = ct
            return inference_function(spc)

        elif node.is_constant():
            value = node.value
            assert value not in inferrers
            return to_abstract(value)

        else:
            fn = yield Require(node.fn)

            # if isinstance(fn, data.AbstractFunction):
            #     inp_types = []
            #     for inp in node.inputs:
            #         inp_types.append((yield Require(inp)))
            #     for inp_type, expected_type in zip(inp_types, fn.args):
            #         autils.unify(expected_type, inp_type, U=unif)

            #     return autils.reify(fn.out, unif=unif.canon)

            if isinstance(fn.tracks.interface, InferenceFunction):
                inf = fn.tracks.interface.fn
                if isinstance(inf, SpecializedGraph):
                    arg_types = yield RequireAll(*node.inputs)
                    inf.commit(arg_types)
                    res = yield Require(inf.graph.return_)
                    return res
                else:
                    res = inf(node.inputs, unif)
                    if isinstance(res, types.GeneratorType):
                        curr = None
                        try:
                            while True:
                                instruction = res.send(curr)
                                if isinstance(instruction, Replace):
                                    self.replacements[node.graph][
                                        None, None, node
                                    ] = instruction.new_node
                                    curr = None
                                else:
                                    curr = yield instruction
                        except StopIteration as stop:
                            return stop.value
                    else:
                        return res

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

            else:
                raise TypeError("Unknown function", fn)


def infer_graph(graph, input_types):
    eng = InferenceEngine()
    g = graph.specialize(input_types)
    eng.replacements[g] = {}
    res = infer(eng, g.return_)

    for gx, repl in eng.replacements.items():
        if gx is not None:
            repl = repl | eng.replacements[None]
            for a, b in repl.items():
                origin, lbl, to_replace = a
                assert origin is None
                assert lbl is None
                if b.is_constant(SpecializedGraph):
                    b = Constant(b.value.graph)
                gx.replace_node(to_replace, None, b)

    return g, res
