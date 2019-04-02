"""Specialize graphs according to the types of their arguments."""

from itertools import count
from dataclasses import replace as dc_replace

from .abstract import GraphFunction, concretize_abstract, \
    AbstractFunction, AbstractError, build_value, MyiaTypeError, \
    TypedPrimitive, BaseGraphInferrer, broaden, \
    TrackedInferrer, PrimitiveFunction, MetaGraphFunction
from .abstract import Context, Unspecializable, \
    DEAD, POLY, VirtualReference
from .ir import GraphCloner, Constant, Graph, MetaGraph
from .prim import ops as P, Primitive


_count = count(1)


def _const(v, t):
    ct = Constant(v)
    ct.abstract = t
    return ct


async def concretize_cache(cache):
    """Complete a cache with concretized versions of its keys.

    If an entry in the cache has a key that contains a Pending, a new key
    is created where the Pending is resolved, and it is entered in the cache
    so that it can be found more easily.
    """
    for k, v in list(cache.items()):
        kc = await concretize_abstract(k)
        cache[kc] = v


class TypeSpecializer:
    """Specialize a graph using inferred type information."""

    def __init__(self, engine):
        """Initialize a TypeSpecializer."""
        self.engine = engine
        self.seen = set()
        self.mng = self.engine.mng
        self.specializations = {Context.empty(): None}
        self.infcaches = {}

    def run(self, graph, context):
        """Run the specializer on the given graph in the given context."""
        ginf = self.engine.get_inferrer_for(GraphFunction(graph, context))

        argrefs = [self.engine.ref(p, context)
                   for p in graph.parameters]

        return self.engine.run_coroutine(
            self._specialize_helper(ginf, argrefs)
        )

    async def _specialize_helper(self, ginf, argrefs):
        await concretize_cache(self.engine.cache.cache)
        await concretize_cache(self.engine.reference_map)
        g = ginf._graph
        argvals = [await self.engine.get_inferred(ref)
                   for ref in argrefs]
        ctx = ginf.make_context(self.engine, argvals)
        return await self._specialize(g, ctx, argrefs)

    async def _specialize(self, g, ctx, argrefs):
        if ctx in self.specializations:
            return self.specializations[ctx].new_graph

        gspec = _GraphSpecializer(self, g, await concretize_abstract(ctx))
        g2 = gspec.new_graph
        self.specializations[ctx] = gspec
        await gspec.run()
        return g2


def _visible(g, parentg):
    while g and parentg is not g:
        g = g.parent
    return parentg is g


class _GraphSpecializer:
    """Helper class for TypeSpecializer."""

    def __init__(self, specializer, graph, context):
        self.parent = specializer.specializations[context.parent]
        self.specializer = specializer
        self.engine = specializer.engine
        self.graph = graph
        self.context = context
        self.cl = GraphCloner(
            self.graph,
            total=False,
            clone_children=False,
            graph_relation=next(_count)
        )
        self.cl.run()
        self.repl = self.cl.repl
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
        await self.first_pass()
        await self.second_pass()

    def ref(self, node):
        return self.engine.ref(node, self.context)

    #########
    # Build #
    #########

    async def build_inferrer(self, a, fn, argvals):
        if isinstance(fn, TypedPrimitive):
            return _const(fn.prim, a)

        inf = self.specializer.engine.get_inferrer_for(fn)
        argvals = argvals and inf.normalize_args(argvals)
        argvals, outval = await self._find_unique_argvals(a, inf, argvals)

        if isinstance(inf, TrackedInferrer):
            fn = dc_replace(fn, tracking_id=None)
            inf = self.specializer.engine.get_inferrer_for(fn)

        if isinstance(fn, PrimitiveFunction):
            a = AbstractFunction(TypedPrimitive(fn.prim, argvals, outval))
            return _const(fn.prim, a)

        assert isinstance(inf, BaseGraphInferrer)

        assert _visible(self.graph, fn.context.graph)

        ctx = inf.make_context(self.specializer.engine, argvals)
        v = await self.specializer._specialize(ctx.graph, ctx, None)
        return _const(v, a)

    async def _find_choices(self, inf):
        if inf not in self.specializer.infcaches:
            rcache = {}
            for k, v in list(inf.cache.items()):
                kc = tuple([await concretize_abstract(x) for x in k])
                inf.cache[kc] = v
                rcache[kc] = v
            self.specializer.infcaches[inf] = rcache
        return self.specializer.infcaches[inf]

    async def _find_generalized(self, inf):
        choices = set()
        for argvals in self.specializer.infcaches[inf]:
            argvals = tuple([broaden(v, None) for v in argvals])
            choices.add(argvals)
        if len(choices) == 1:
            choice, = choices
            argrefs = [VirtualReference(v) for v in choice]
            res = await inf.run(self.specializer.engine, None, argrefs)
            self.specializer.infcaches[inf] = {choice: res}
            return choice, res
        else:
            return None, None

    async def _find_unique_argvals(self, a, inf, argvals):
        if argvals is not None:
            argvals = tuple(argvals)
        if argvals in inf.cache:
            # We do this first because it's inexpensive
            return argvals, inf.cache[argvals]
        choices = await self._find_choices(inf)
        if argvals in inf.cache:
            # We do this a second time because _find_choices may have
            # added what we are looking for.
            return argvals, inf.cache[argvals]
        elif len(choices) == 1:
            for choice in choices.items():
                # Return the only element
                return choice
        elif len(choices) == 0:
            raise Unspecializable(DEAD)
        else:
            generalized, outval = await self._find_generalized(inf)
            if generalized is not None:
                return generalized, outval
            raise Unspecializable(POLY)

    ###########
    # Process #
    ###########

    async def first_pass(self):
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

    async def second_pass(self):
        # Specialize constant graphs
        from .graph_utils import dfs
        from .ir.utils import succ_incoming
        for node in dfs(self.new_graph.return_, succ=succ_incoming):
            if node.is_apply():
                await self.process_apply(node)

    async def build(self, ref, a):
        if isinstance(a, AbstractFunction):
            try:
                fn = a.get_unique()
                if isinstance(fn, PrimitiveFunction):
                    g = fn.prim
                elif isinstance(fn, MetaGraphFunction):
                    g = fn.metagraph
                elif isinstance(fn, GraphFunction):
                    g = fn.graph
                else:
                    return None
                if not isinstance(g, Graph) \
                        or g.parent is None \
                        or (ref.node.is_constant_graph()
                            and _visible(self.graph, g.parent)):
                    return _const(g, a)
                else:
                    return None
            except MyiaTypeError:
                return None
        else:
            try:
                v = build_value(a)
            except ValueError:
                return None
            else:
                return _const(v, a)

    async def process_node(self, node):
        ref = self.ref(node)
        new_node = self.get(node)
        if new_node.graph is not self.new_graph:
            raise AssertionError('Error in specializer [A]')

        new_node.abstract = await concretize_abstract(await ref.get())

        if node.is_apply():
            new_inputs = new_node.inputs
            irefs = list(map(self.ref, node.inputs))
            ivals = [await concretize_abstract(await iref.get())
                     for iref in irefs]
            for i, ival in enumerate(ivals):
                iref = irefs[i]
                repl = await self.build(iref, ival)
                if repl is None:
                    _iref = self.specializer.engine.get_actual_ref(iref)
                    if _iref is not iref:
                        self.cl.clone_disconnected(_iref.node)
                        iref = _iref
                    self.todo.append(iref.node)
                    repl = self.get(iref.node)
                    repl.abstract = await concretize_abstract(ival)
                if repl is not new_inputs[i]:
                    new_inputs[i] = repl

    async def process_apply(self, new_node):
        if new_node in self.specializer.seen:
            return
        self.specializer.seen.add(new_node)

        async def _helper(i, node, a, argvals):
            if node.is_constant_graph() \
                    or node.is_constant(MetaGraph) \
                    or node.is_constant(Primitive):
                # There should only be one possibility here
                fn = a.get_unique()
                try:
                    repl = await self.build_inferrer(a, fn, argvals)
                except Unspecializable as e:
                    repl = _const(e.problem, AbstractError(e.problem))
                new_inputs[i] = repl

        new_inputs = new_node.inputs
        fn, *args = new_inputs
        while fn.is_apply(P.partial):
            args = fn.inputs[2:] + args
            fn = fn.inputs[1]
            new_inputs = [fn, *args]
        fnval, *argvals = [x.abstract for x in new_inputs]
        await _helper(0, fn, fnval, argvals)
        for i, val in enumerate(argvals):
            await _helper(i + 1, args[i], val, None)

        new_node.inputs = new_inputs
