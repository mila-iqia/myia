"""Specialize graphs according to the types of their arguments."""

import numpy
from itertools import count

from .dtype import Function, TypeMeta
from .infer import ANYTHING, Context, concretize_type, \
    GraphInferrer, PartialInferrer, Inferrer, Unspecializable, \
    DEAD, INACCESSIBLE
from .ir import GraphCloner, Constant, Graph
from .prim import ops as P, Primitive
from .utils import Overload, overload, Namespace, SymbolicKeyInstance, \
    EnvInstance


_count = count(1)


def _const(v, t):
    ct = Constant(v)
    ct.type = t
    return ct


class TypeSpecializer:
    """Specialize a graph using inferred type information."""

    def __init__(self, engine):
        """Initialize a TypeSpecializer."""
        self.engine = engine
        self.mng = self.engine.mng
        self.specializations = {Context.empty(): None}

    def run(self, graph, context):
        """Run the specializer on the given graph in the given context."""
        ginf = GraphInferrer(self.engine.tracks['type'],
                             graph, Context.empty(),
                             broaden=False)

        argrefs = [self.engine.ref(p, context)
                   for p in graph.parameters]

        return self.engine.run_coroutine(
            self._specialize(ginf, argrefs)
        )

    async def _specialize(self, ginf, argrefs):
        g = await ginf.make_graph(argrefs)
        ctx = await ginf.make_context(argrefs)

        ctxkey = ctx  # TODO: Reify ctx to collapse multiple ctx into one
        if ctxkey in self.specializations:
            return self.specializations[ctxkey].new_graph

        gspec = _GraphSpecializer(self, g, ctx)
        g2 = gspec.new_graph
        self.specializations[ctxkey] = gspec
        await gspec.run()
        return g2


_legal = (int, float, numpy.number, numpy.ndarray,
          str, Namespace, SymbolicKeyInstance, TypeMeta)


def _visible(g, node):
    if isinstance(node, Graph):
        g2 = node.parent
    else:
        g2 = node.graph
    while g and g2 is not g:
        g = g.parent
    return g2 is g


@overload
def _is_concrete_value(v: (tuple, list)):
    return all(_is_concrete_value(x) for x in v)


@overload  # noqa: F811
def _is_concrete_value(v: EnvInstance):
    return all(_is_concrete_value(x) for x in v._contents.values())


@overload  # noqa: F811
def _is_concrete_value(v: _legal):
    return True


@overload  # noqa: F811
def _is_concrete_value(v: object):
    return False


class _GraphSpecializer:
    """Helper class for TypeSpecializer."""

    def __init__(self, specializer, graph, context):
        self.parent = specializer.specializations[context.parent]
        self.specializer = specializer
        self.engine = specializer.engine
        self.graph = graph
        self.context = context
        cl = GraphCloner(
            self.graph,
            total=False,
            clone_children=False,
            graph_relation=next(_count)
        )
        cl.run()
        self.repl = cl.repl
        self.new_graph = self.repl[self.graph]
        self.todo = [self.graph.return_] + list(self.graph.parameters)
        self.marked = set()

    def get(self, node):
        g = node.graph
        sp = self
        while g is not None and g is not sp.graph:
            sp = sp.parent
        return sp.repl.get(node, node)

    async def run(self):
        while self.todo:
            node = self.todo.pop()
            if node.graph is None:
                continue
            if node.graph is not self.graph:
                self.parent.todo.append(node)
                await self.parent.run()
                continue
            if node in self.marked:
                continue
            self.marked.add(node)
            await self.process_node(node)

    def ref(self, node):
        return self.engine.ref(node, self.context)

    #########
    # Build #
    #########

    async def build(self, ref, argrefs=None):
        t = await ref['type']
        v = await ref['value']
        if isinstance(t, Inferrer) and v is not ANYTHING:
            if argrefs is None:
                argrefs = await t.get_unique_argrefs()
            return await self._build_inferrer(t, argrefs)
        elif _is_concrete_value(v):
            return _const(v, await concretize_type(t))
        elif not _visible(self.graph, ref.node):
            raise Unspecializable(INACCESSIBLE)
        else:
            self.todo.append(ref.node)
            new_node = self.get(ref.node)
            new_node.type = await concretize_type(t)
            return new_node

    _build_inferrer = Overload()

    @_build_inferrer.register
    async def _build_inferrer(self, inf: GraphInferrer, argrefs):
        g = await inf.make_graph(argrefs)
        if _visible(self.graph, g):
            v = await self.specializer._specialize(inf, argrefs)
            return _const(v, await inf.as_function_type(argrefs))
        else:
            raise Unspecializable(INACCESSIBLE)

    @_build_inferrer.register  # noqa: F811
    async def _build_inferrer(self, inf: PartialInferrer, argrefs):
        all_argrefs = None if argrefs is None else [*inf.args, *argrefs]
        sub_build = await self._build_inferrer(inf.fn, all_argrefs)
        ptl_args = [await self.build(ref) for ref in inf.args]
        res_t = await inf.as_function_type(argrefs)
        ptl = _const(P.partial, Function[
            [sub_build.type, *[a.type for a in ptl_args]],
            res_t
        ])
        res = self.new_graph.apply(
            ptl,
            sub_build,
            *ptl_args
        )
        res.type = res_t
        return res

    @_build_inferrer.register  # noqa: F811
    async def _build_inferrer(self, inf: Inferrer, argrefs):
        v = inf.identifier
        if isinstance(v, Primitive):
            return _const(v, await inf.as_function_type(argrefs))
        else:
            raise Unspecializable(DEAD)

    ###########
    # Process #
    ###########

    async def fill_inferred(self, new_node, ref):
        # Fill in inferred properties like shape, etc.
        # Inference for 'type' and 'value' is ignored here because
        # they are processed specifically by the rest of the code.
        for name, track in self.engine.tracks.items():
            if name not in ('type', 'value'):
                res = await ref[name]
                if not isinstance(res, Inferrer):
                    new_node.inferred[name] = res

    async def process_node(self, node):
        ref = self.ref(node)
        new_node = self.get(node)
        if new_node.graph is not self.new_graph:
            raise AssertionError('Error in specializer [A]')

        new_node.type = await concretize_type(await ref['type'])

        await self.fill_inferred(new_node, ref)

        if node.is_apply():
            new_inputs = new_node.inputs
            irefs = list(map(self.ref, node.inputs))
            for i, iref in enumerate(irefs):
                argrefs = irefs[1:] if i == 0 else None
                try:
                    repl = await self.build(ref=iref, argrefs=argrefs)
                    await self.fill_inferred(repl, iref)
                except Unspecializable as e:
                    if new_inputs[i].is_constant_graph():
                        # Graphs that cannot be specialized are replaced
                        # by a constant with the associated Problem type.
                        # We can't keep references to unspecialized graphs.
                        repl = _const(e.problem.kind, e.problem)
                    else:
                        self.todo.append(node.inputs[i])
                        it = await iref['type']
                        repl = self.get(node.inputs[i])
                        repl.type = await concretize_type(it)
                if repl is not new_inputs[i]:
                    new_inputs[i] = repl
                else:
                    self.todo.append(node.inputs[i])
