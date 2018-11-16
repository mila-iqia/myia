"""Specialize graphs according to the types of their arguments."""

import numpy
from itertools import count

from .dtype import Type, Function, TypeMeta
from .infer import ANYTHING, Context, concretize_type, \
    GraphInferrer, MetaGraphInferrer, PartialInferrer, Inferrer, \
    Unspecializable, DEAD, INACCESSIBLE
from .ir import GraphCloner, Constant, Graph
from .prim import ops as P, Primitive
from .utils import Overload, overload, Namespace, SymbolicKeyInstance, \
    EnvInstance


_count = count()


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
        self.node_map = self.mng.nodes
        self.originals = {}
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
        self.originals[g2] = g
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
        self.nodes = specializer.node_map[self.graph]

        g = self.graph
        if self.parent:
            g = self.parent.get(g)
        self.cl = GraphCloner(g, total=False, graph_relation=next(_count))
        self.new_graph = self.cl[g]

    def get(self, node):
        if self.parent:
            node = self.parent.get(node)
        return self.cl[node]

    async def run(self):
        for node in self.nodes:
            await self.process_node(node)

    def ref(self, node):
        return self.engine.ref(node, self.context)

    #########
    # Build #
    #########

    async def build(self, ref, argrefs=None, t=None):
        if t is None:
            t = await ref['type']
        if ref is not None and isinstance(t, Inferrer):
            if (await ref['value']) is ANYTHING:
                t = await t.as_function_type(argrefs)
                return await self._build[Type](ref, argrefs, t)
        return await self._build(ref, argrefs, t)

    _build = Overload()

    @_build.register
    async def _build(self, ref, argrefs, inf: GraphInferrer):
        if isinstance(inf, MetaGraphInferrer):
            g = None
        else:
            g = await inf.make_graph(None)
        if g is None or _visible(self.graph, g):
            if argrefs is None:
                argrefs = await inf.get_unique_argrefs()
            v = await self.specializer._specialize(inf, argrefs)
            return _const(v, await inf.as_function_type(argrefs))
        else:
            raise Unspecializable(INACCESSIBLE)

    @_build.register  # noqa: F811
    async def _build(self, ref, argrefs, inf: PartialInferrer):
        all_argrefs = None if argrefs is None else [*inf.args, *argrefs]
        sub_build = await self._build(None, all_argrefs, inf.fn)
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

    @_build.register  # noqa: F811
    async def _build(self, ref, argrefs, inf: Inferrer):
        v = inf.identifier
        if isinstance(v, Primitive):
            return _const(v, await inf.as_function_type(argrefs))
        else:
            raise Unspecializable(DEAD)

    @_build.register  # noqa: F811
    async def _build(self, ref, argrefs, t: object):
        v = await ref['value']
        if _is_concrete_value(v):
            return _const(v, await concretize_type(t))
        elif not _visible(self.graph, ref.node):
            raise Unspecializable(INACCESSIBLE)
        else:
            new_node = self.get(ref.node)
            new_node.type = await concretize_type(t)
            return new_node

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
                    prev = new_inputs[i]
                    if repl.graph and prev.graph \
                            and repl.graph is not prev.graph:
                        raise AssertionError('Error in specializer [B]')
                    new_inputs[i] = repl
                except Unspecializable as e:
                    if new_inputs[i].is_constant_graph():
                        # Graphs that cannot be specialized are replaced
                        # by a constant with the associated Problem type.
                        # We can't keep references to unspecialized graphs.
                        new_inputs[i] = _const(e.problem.kind, e.problem)
                    else:
                        it = await iref['type']
                        new_inputs[i].type = await concretize_type(it)
