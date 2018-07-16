"""Specialize graphs according to the types of their arguments."""

from collections import defaultdict, Counter

from .dtype import Type, Function, Dead, Unknown
from .infer import Context, GraphInferrer, ANYTHING, Inferrer
from .ir import GraphCloner, is_apply, is_constant, Constant, \
    succ_deeper, Graph
from .prim import Primitive
from .graph_utils import dfs
from .utils import Named, TypeMap


UNKNOWN = Named('UNKNOWN')
POLY = Named('POLY')


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

        empty_ctx = Context.empty()
        ginf = GraphInferrer(self.engine.tracks['type'],
                             engine.graph,
                             empty_ctx)
        argrefs = self.engine.argrefs

        self.result = self.engine.run_coroutine(
            self._specialize(None, ginf, argrefs)
        )

    async def _specialize(self, parent, ginf, argrefs):
        g = ginf.graph

        ctx = await ginf.make_context(argrefs)

        if ctx in self.specializations:
            return self.specializations[ctx]

        self.counts[g] += 1
        gspec = _GraphSpecializer(parent, self, ginf.graph, ctx)
        g2 = gspec.new_graph
        self.originals[g2] = g
        self.specializations[ctx] = g2
        await gspec.run()
        return g2


_concretize_map = TypeMap()


class _NotConcrete(Exception):
    pass


@_concretize_map.register(tuple)
def _concretize_tuple(xs):
    return tuple(map(_concretize, xs))


@_concretize_map.register(list)
def _concretize_list(xs):
    return list(map(_concretize, xs))


@_concretize_map.register(Named)
def _concretize_Named(x):
    if x is ANYTHING:
        raise _NotConcrete()
    else:
        raise ValueError(f'Invalid value: {x}')


@_concretize_map.register(object)
def _concretize_object(x):
    return x


@_concretize_map.register(GraphInferrer)
def _concretize_GraphInferrer(v):
    g = v.graph
    if g.parent is None:
        return g
    else:
        raise _NotConcrete()


@_concretize_map.register(Inferrer)
def _concretize_Inferrer(v):
    v = v.identifier
    if isinstance(v, Primitive):
        return v
    else:
        # This can't happen at the moment
        raise _NotConcrete()  # pragma: no cover


def _concretize(x):
    return _concretize_map[type(x)](x)


class _GraphSpecializer:
    """Helper class for TypeSpecializer."""

    def __init__(self, parent, specializer, graph, context):
        self.parent = parent
        self.specializer = specializer
        self.engine = specializer.engine
        self.graph = graph
        self.context = context
        self.nodes = specializer.node_map[self.graph]

        # rel = 'copy'
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

    async def value_of(self, ref):
        node = ref.node
        if is_constant(node):
            return node.value
        else:
            v = await ref['value']
            try:
                return _concretize(v)
            except _NotConcrete:
                return UNKNOWN

    async def extract_type_and_argrefs(self, ref, argrefs=None):
        if isinstance(ref, (Type, Inferrer)):
            t = ref
        else:
            t = await ref['type']

        if not isinstance(t, Inferrer):
            return t, None

        if argrefs is None:
            # The cache works using References, but if two references have
            # the same inferred type/value/etc., we can merge their entries.
            cache = {}
            for x, y in t.cache.items():
                key = tuple([await self.extract_type(arg) for arg in x])
                if key in cache:
                    assert cache[key] == y
                cache[key] = y

            if len(cache) == 0:
                return Dead(), None
            elif len(cache) == 1:
                (argrefs, res), *_ = t.cache.items()
            else:
                return POLY, None

        else:
            res = t.cache[tuple(argrefs)]

        ftype = Function([await self.extract_type(argref)
                          for argref in argrefs],
                         await self.extract_type(res))

        return ftype, argrefs

    async def extract_type(self, ref, argrefs=None):
        t, _ = await self.extract_type_and_argrefs(ref, argrefs)
        return t

    async def make_constant(self, ref, argrefs=None):
        v = await self.value_of(ref)
        t = await ref['type']

        if v is UNKNOWN:
            return None

        elif isinstance(v, (Graph, Primitive)):

            ftype, argrefs = \
                await self.extract_type_and_argrefs(ref, argrefs)

            if argrefs is None:
                return _const(ftype, ftype)

            if isinstance(v, Graph):
                v = await self.specializer._specialize(
                    self, t, argrefs
                )

            return _const(v, ftype)

        else:
            assert not isinstance(v, (Graph, Primitive))
            return _const(v, t)

    async def process_node(self, node):
        ref = self.ref(node)

        self.get(node).type = await self.extract_type(ref)

        if is_apply(node):
            irefs = list(map(self.ref, node.inputs))
            for i, iref in enumerate(irefs):
                if i == 0:
                    ct = await self.make_constant(iref, irefs[1:])
                else:
                    ct = await self.make_constant(iref)
                if ct:
                    self.get(node).inputs[i] = ct


def validate(g):  # pragma: no cover
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    and every application must be compatible with its argument types.
    """
    errors = defaultdict(set)
    for node in dfs(g.return_, succ_deeper):
        if node.type is None or node.type == Unknown():
            errors[node].add('notype')
        elif isinstance(node.type, Inferrer):
            errors[node].add('inferrer')
        elif not isinstance(node.type, Type):
            errors[node].add(f'{node.type}')
        elif is_apply(node):
            expected = Function([i.type for i in node.inputs[1:]], node.type)
            if node.inputs[0].type != expected:
                errors[node].add('mismatch')

    return errors
