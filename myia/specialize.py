"""Specialize graphs according to the types of their arguments."""

from collections import Counter

from .dtype import Type, Function, Number, Bool, Problem, TypeType
from .infer import ANYTHING, Context, reify, \
    GraphInferrer, MetaGraphInferrer, PartialInferrer, Inferrer
from .ir import GraphCloner, Constant
from .prim import ops as P, Primitive
from .utils import Named, TypeMap


UNKNOWN = Named('UNKNOWN')
DEAD = Named('DEAD')
POLY = Named('POLY')
INACCESSIBLE = Named('INACCESSIBLE')


class _Unspecializable(Exception):
    def __init__(self, problem):
        problem = Problem[problem]
        super().__init__(problem)
        self.problem = problem


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
        self.specializations = {}
        self.counts = Counter()

    def run(self, graph, context):
        """Run the specializer on the given graph in the given context."""
        ginf = GraphInferrer(self.engine.tracks['type'],
                             graph, Context.empty(),
                             broaden=False)

        argrefs = [self.engine.ref(p, context)
                   for p in graph.parameters]

        return self.engine.run_coroutine(
            self._specialize(None, ginf, argrefs)
        )

    async def _specialize(self, parent, ginf, argrefs):
        g = await ginf.make_graph(argrefs)
        ctx = await ginf.make_context(argrefs)

        ctxkey = await reify(ctx)
        if ctxkey in self.specializations:
            return self.specializations[ctxkey]

        self.counts[g] += 1
        gspec = _GraphSpecializer(parent, self, g, ctx)
        g2 = gspec.new_graph
        self.originals[g2] = g
        self.specializations[ctxkey] = g2
        await gspec.run()
        return g2


async def _find_argrefs(inf):
    # The cache works using References, but if two references have
    # the same inferred type/value/etc., we can merge their entries.
    cache = {}
    for x, y in inf.cache.items():
        y = await reify(y)
        key = tuple([await arg[track] for arg in x
                     for track in inf.engine.tracks])
        if key in cache:
            assert cache[key] == y
        cache[key] = y

    if len(cache) == 0:
        raise _Unspecializable(DEAD)
    elif len(cache) == 1:
        (argrefs, res), *_ = inf.cache.items()
        return argrefs
    else:
        raise _Unspecializable(POLY)


async def _concretize_type(t, argrefs=None):
    if isinstance(t, Inferrer):
        if argrefs is None:
            try:
                argrefs = await _find_argrefs(t)
            except _Unspecializable as e:
                return e.args[0]
        return Function[[await _extract_type(argref)
                         for argref in argrefs],
                        await _concretize_type(t.cache[tuple(argrefs)])]
    else:
        return await reify(t)


async def _extract_type(ref, argrefs=None):
    return await _concretize_type(await ref['type'], argrefs)


def _parents(g):
    rval = set()
    while g.parent:
        g = g.parent
        rval.add(g)
    return rval


class _GraphSpecializer:
    """Helper class for TypeSpecializer."""

    def __init__(self, parent, specializer, graph, context):
        par = _parents(graph)
        while parent and parent.graph not in par:
            parent = parent.parent

        self.parent = parent
        self.specializer = specializer
        self.engine = specializer.engine
        self.graph = graph
        self.context = context
        self.nodes = specializer.node_map[self.graph]

        rel = f'{self.specializer.counts[self.graph]}'
        g = self.graph
        if self.parent:
            g = self.parent.get(g)
        self.cl = GraphCloner(g, total=False, graph_relation=rel)
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

    build_map = TypeMap()

    async def build(self, ref, argrefs=None, t=None):
        if t is None:
            t = await ref['type']
        if ref is not None and isinstance(t, Inferrer):
            if (await ref['value']) is ANYTHING:
                t = await _concretize_type(t, argrefs)
                return await self.build_generic(ref, argrefs, t)
        handler = self.build_map[type(t)]
        return await handler(self, ref, argrefs, t)

    @build_map.register(GraphInferrer)
    async def build_GraphInferrer(self, ref, argrefs, inf):
        if isinstance(inf, MetaGraphInferrer):
            g = None
        else:
            g = await inf.make_graph(None)
        if not g or g.parent is None or ref and ref.node.is_constant_graph():
            if argrefs is None:
                argrefs = await _find_argrefs(inf)
            v = await self.specializer._specialize(
                self, inf, argrefs
            )
            return _const(v, await _concretize_type(inf, argrefs))
        else:
            raise _Unspecializable(INACCESSIBLE)

    @build_map.register(PartialInferrer)
    async def build_PartialInferrer(self, ref, argrefs, inf):
        all_argrefs = None if argrefs is None else [*inf.args, *argrefs]
        sub_build = await self.build(None, all_argrefs, inf.fn)
        ptl_args = [await self.build(ref) for ref in inf.args]
        res_t = await _concretize_type(inf, argrefs)
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

    @build_map.register(Inferrer)
    async def build_Inferrer(self, ref, argrefs, inf):
        v = inf.identifier
        assert isinstance(v, Primitive)
        return _const(v, await _concretize_type(inf, argrefs))

    @build_map.register(Number)
    @build_map.register(Bool)
    @build_map.register(TypeType)
    @build_map.register(type)
    async def build_atom(self, ref, argrefs, t):
        v = await ref['value']
        if v is ANYTHING:
            return await self.build_generic(ref, argrefs, t)
        else:
            return _const(v, t)

    @build_map.register(Type)
    async def build_generic(self, ref, argrefs, t):
        new_node = self.get(ref.node)
        new_node.type = t
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
                if not isinstance(ref, Inferrer):
                    new_node.inferred[name] = res

    async def process_node(self, node):
        ref = self.ref(node)
        new_node = self.get(node)
        if new_node.graph is not self.new_graph:
            raise AssertionError('Error in specializer [A]')

        t = await _extract_type(ref)
        new_node.type = t

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
                except _Unspecializable as e:
                    if new_inputs[i].is_constant_graph():
                        # Graphs that cannot be specialized are replaced
                        # by a constant with the associated Problem type.
                        # We can't keep references to unspecialized graphs.
                        new_inputs[i] = _const(e.problem.kind, e.problem)
                    else:
                        new_inputs[i].type = await _extract_type(iref)
