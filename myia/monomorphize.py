"""Monomorphization algorithm.

Monomorphization creates a separate function for each type signature the
function may be called with.
"""

from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from itertools import chain, count
from typing import Optional
from warnings import warn

from .abstract import (
    DEAD,
    POLY,
    AbstractError,
    AbstractFunction,
    AbstractValue,
    Context,
    DummyFunction,
    GraphFunction,
    GraphInferrer,
    MetaGraphFunction,
    PrimitiveFunction,
    Reference,
    TrackedInferrer,
    TypedPrimitive,
    VirtualReference,
    abstract_check,
    abstract_clone,
    broaden,
    build_value,
    concretize_abstract,
)
from .abstract.utils import CheckState, CloneState
from .graph_utils import dfs
from .info import About
from .ir import (
    CloneRemapper,
    Constant,
    Graph,
    GraphCloner,
    MetaGraph,
    succ_incoming,
)
from .operations import Primitive
from .utils import InferenceError, MyiaTypeError, overload


class Unspecializable(Exception):
    """Raised when it is impossible to specialize an inferrer."""

    def __init__(self, problem, data=None):
        """Initialize Unspecializable."""
        super().__init__(problem)
        self.problem = problem
        self.data = data


@abstract_clone.variant
def _fix_type(self, a: GraphFunction, spc):
    if a.graph in spc.ctcache:
        ctx = spc.ctcache[a.graph]
        return GraphFunction(spc.results[ctx], Context.empty(), None)
    else:
        assert a.graph not in spc.results
        return DummyFunction()


@overload  # noqa: F811
def _fix_type(self, a: PrimitiveFunction, spc):
    try:
        return spc.analyze_function(None, a, None)[0].abstract.get_unique()
    except Unspecializable:
        return a


@overload  # noqa: F811
def _fix_type(self, a: TypedPrimitive, spc):
    return TypedPrimitive(a.prim, tuple(self(ar, spc) for ar in a.args),
                          self(a.output, spc))


@abstract_check.variant(
    initial_state=lambda: CheckState({}, '_no_track')
)
def _check_no_tracking_id(self, x: GraphFunction):
    return x.tracking_id is None


@concretize_abstract.variant(
    initial_state=lambda: CloneState(
        cache={},
        prop='_no_track',
        check=_check_no_tracking_id,
    )
)
def _no_tracking_id(self, x: GraphFunction):
    return dc_replace(x, tracking_id=None)


@overload(bootstrap=True)
def _refmap(self, fn, x: Context):
    return Context(
        self(fn, x.parent),
        x.graph,
        tuple(fn(arg) for arg in x.argkey)
    )


@overload  # noqa: F811
def _refmap(self, fn, x: Reference):
    return Reference(
        x.engine,
        x.node,
        self(fn, x.context),
    )


@overload  # noqa: F811
def _refmap(self, fn, x: tuple):
    return tuple(self(fn, y) for y in x)


@overload  # noqa: F811
def _refmap(self, fn, x: AbstractValue):
    return fn(x)


@overload  # noqa: F811
def _refmap(self, fn, x: object):
    return x


def concretize_cache(cache):
    """Complete a cache with concretized versions of its keys.

    If an entry in the cache has a key that contains a Pending, a new key
    is created where the Pending is resolved, and it is entered in the cache
    so that it can be found more easily.
    """
    for k, v in list(cache.items()):
        kc = _refmap(concretize_abstract, k)
        cache[kc] = v
        kc2 = _refmap(_no_tracking_id, kc)
        cache[kc2] = v


_count = count(1)


def _const(v, t):
    ct = Constant(v)
    ct.abstract = t
    ct.force_abstract = True
    return ct


def _build(a):
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
            if not isinstance(g, Graph) or g.parent is None:
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


@dataclass(frozen=True)
class _Placeholder:
    context: Context


@dataclass(frozen=True)
class _TodoEntry:
    ref: Reference
    argvals: Optional[tuple]
    link: Optional[tuple]


def _normalize_context(ctx):
    return _refmap(_no_tracking_id, ctx)


class Monomorphizer:
    """Monomorphize graphs using inferred type information.

    Monomorphization creates a separate function for each type signature the
    function may be called with.

    Arguments:
        engine: The InferenceEngine containing the type information.
        reuse_existing: Whether to reuse existing graphs when possible.
    """

    def __init__(self, engine, reuse_existing=True):
        """Initialize the monomorphizer."""
        self.engine = engine
        self.specializations = {}
        self.replacements = defaultdict(dict)
        self.results = {}
        self.ctcache = {}
        self.infcaches = {}
        self.reuse_existing = reuse_existing

    def run(self, context):
        """Run monomorphization."""
        self.manager = context.graph.manager
        concretize_cache(self.engine.cache.cache)
        concretize_cache(self.engine.reference_map)
        self.collect(context)
        self.order_tasks()
        self.create_graphs()
        self.monomorphize()
        self.fill_placeholders()
        result = self.results[context]
        self.manager.keep_roots(result)
        return result

    #########
    # Build #
    #########

    def analyze_function(self, a, fn, argvals):
        """Analyze a function for the collect phase.

        Arguments:
            a: The abstract value for the function.
            fn: The Function object, equivalent a.get_unique().
            argvals: The abstract arguments given to the function.

        Returns:
            ct: A Constant to use for this call.
            ctx: The context for this call, or None
            norm_ctx: The normalized context for this call, or None

        """
        inf = self.engine.get_inferrer_for(fn)
        argvals = argvals and inf.normalize_args_sync(argvals)
        argvals, outval = self._find_unique_argvals(a, inf, argvals)

        if isinstance(inf, TrackedInferrer):
            fn = dc_replace(fn, tracking_id=None)
            inf = self.engine.get_inferrer_for(fn)

        if isinstance(fn, PrimitiveFunction):
            a = AbstractFunction(TypedPrimitive(fn.prim, argvals, outval))
            return _const(fn.prim, a), None

        assert isinstance(inf, GraphInferrer)
        concretize_cache(inf.graph_cache)

        ctx = inf.make_context(self.engine, argvals)
        norm_ctx = _normalize_context(ctx)
        if norm_ctx not in self.specializations:
            self.specializations[norm_ctx] = ctx
        new_ct = _const(_Placeholder(norm_ctx), None)
        return new_ct, norm_ctx

    def _find_choices(self, inf):
        if inf not in self.infcaches:
            rcache = {}
            for k, v in list(inf.cache.items()):
                kc = tuple(concretize_abstract(x) for x in k)
                inf.cache[kc] = v
                rcache[kc] = v
            self.infcaches[inf] = rcache
        return self.infcaches[inf]

    def _find_generalized(self, inf):
        choices = set()
        for argvals in self.infcaches[inf]:
            argvals = tuple(broaden(v) for v in argvals)
            choices.add(argvals)
        if len(choices) == 1:
            choice, = choices
            argrefs = [VirtualReference(v) for v in choice]
            try:
                res = self.engine.run_coroutine(
                    inf.run(self.engine, None, argrefs)
                )
            except InferenceError:
                return None, None
            self.infcaches[inf] = {choice: res}
            return choice, res
        else:
            return None, None

    def _find_unique_argvals(self, a, inf, argvals):
        if argvals is not None:
            argvals = tuple(argvals)
            # Let's try to get broader/more general argvals to avoid
            # specializing on values, if we can.
            broad_argvals = tuple(broaden(v) for v in argvals)
            if argvals != broad_argvals:
                currinf = inf
                while hasattr(currinf, 'subinf'):
                    currinf = currinf.subinf
                try:
                    res = self._find_unique_argvals_helper(
                        a, currinf, broad_argvals, False
                    )
                    eng = self.engine
                    if isinstance(currinf, GraphInferrer):
                        # Have to check that this graph was processed by
                        # the inferrer. It should have been, but sometimes
                        # it isn't, not sure why. Hopefully a rewrite of the
                        # specializer should fix everything.
                        concretize_cache(currinf.graph_cache)
                        try:
                            g = currinf.get_graph(eng, broad_argvals)
                        except InferenceError:
                            return res
                        except Exception as e:  # pragma: no cover
                            warn(f'{currinf} failed with a {type(e)}, but it'
                                 f' should only fail with an InferenceError.')
                            return res
                        else:
                            ctx = currinf.make_context(eng, broad_argvals)
                            if eng.ref(g.output, ctx) in eng.cache.cache:
                                return res
                    else:
                        return res
                except Unspecializable:
                    pass
        return self._find_unique_argvals_helper(a, inf, argvals, True)

    def _find_unique_argvals_helper(self, a, inf, argvals,
                                    try_generalize=True):
        if argvals in inf.cache:
            # We do this first because it's inexpensive
            return argvals, inf.cache[argvals]
        choices = self._find_choices(inf)
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
        elif try_generalize:
            generalized, outval = self._find_generalized(inf)
            if generalized is not None:
                return generalized, outval
            raise Unspecializable(POLY, (a, *choices.keys()))
        else:
            raise Unspecializable(POLY, (a, *choices.keys()))

    ###########
    # Collect #
    ###########

    def _special_array_map(self, todo, ref, irefs, argvals):
        todo.append(_TodoEntry(irefs[0], tuple(argvals), (ref, 0)))
        am_argvals = [a.element for a in argvals[1:]]
        todo.append(_TodoEntry(irefs[1], tuple(am_argvals), (ref, 1)))
        # Iterate through the rest of the inputs
        for i, iref in enumerate(irefs[2:]):
            todo.append(_TodoEntry(iref, None, (ref, i + 2)))

    def _special_array_reduce(self, todo, ref, irefs, argvals):
        todo.append(_TodoEntry(irefs[0], tuple(argvals), (ref, 0)))
        elem_t = argvals[1].element
        todo.append(_TodoEntry(irefs[1], (elem_t, elem_t), (ref, 1)))
        todo.append(_TodoEntry(irefs[2], None, (ref, 2)))
        todo.append(_TodoEntry(irefs[3], None, (ref, 3)))

    def collect(self, root_context):
        """Collect all the available contexts.

        Sets self.specializations to a dict from a normalized context (which we
        must generate a graph for) to the original context (cached in the
        inferrer in a possibly unnormalized form that contains Pendings).

        Also sets self.replacements to a context->(node,key)->new_node dict.
        When an inferrer's reroute function tells the inference engine that
        some node is equivalent to another, and to use that other node to
        resume inference, this is reflected in self.replacements.
        """
        root = root_context.graph
        todo = [_TodoEntry(self.engine.ref(root.return_, root_context),
                           None, None)]
        seen = set()

        self.specializations[root_context] = root_context

        while todo:
            entry = todo.pop()
            if entry in seen:
                continue
            seen.add(entry)

            # Get the proper reference
            ref = self.engine.get_actual_ref(entry.ref)
            a = concretize_abstract(ref.get_resolved())
            if entry.link is not None:
                with About(ref.node.debug, 'equiv'):
                    ct = _build(a)
                    if ct is not None:
                        ref = self.engine.ref(ct, ref.context)

            new_node = ref.node

            if ref.node.is_apply():
                # Grab the argvals
                irefs = [self.engine.ref(inp, entry.ref.context)
                         for inp in ref.node.inputs]
                absfn = concretize_abstract(irefs[0].get_resolved())
                argvals = [concretize_abstract(iref.get_resolved())
                           for iref in irefs[1:]]

                prim = absfn.get_prim()
                if prim is not None:
                    method = getattr(self, f'_special_{prim}', None)
                    if method is not None:
                        method(todo, ref, irefs, argvals)
                        continue

                # Keep traversing the graph. Element 0 is special.
                todo.append(_TodoEntry(irefs[0], tuple(argvals),
                                       (ref, 0)))
                # Iterate through the rest of the inputs
                for i, iref in enumerate(irefs[1:]):
                    todo.append(_TodoEntry(iref, None, (ref, i + 1)))

            elif ref.node.is_constant_graph() \
                    or ref.node.is_constant(MetaGraph) \
                    or ref.node.is_constant(Primitive):

                fn = a.get_unique()
                with About(ref.node.debug, 'equiv'):
                    try:
                        new_node, norm_ctx = self.analyze_function(
                            a, fn, entry.argvals)
                    except Unspecializable as e:
                        aerr = AbstractError(e.problem, e.data)
                        new_node = _const(e.problem, aerr)
                    else:
                        if (isinstance(fn, GraphFunction)
                                and entry.argvals is None):
                            self.ctcache[ref.node.value] = norm_ctx

                        if norm_ctx is not None:
                            retref = self.engine.ref(norm_ctx.graph.return_,
                                                     norm_ctx)
                            todo.append(_TodoEntry(retref, None, None))

            if new_node is not entry.ref.node:
                if entry.link is None:
                    raise AssertionError('Cannot replace a return node.')
                else:
                    ref, _ = entry.link
                    nctx = _normalize_context(ref.context)
                    self.replacements[nctx][entry.link] = new_node

    ###############
    # Order tasks #
    ###############

    def order_tasks(self):
        """Create an ordered list of "tasks" to perform into self.tasks.

        Each task is a context/original_context pair. They are ordered such
        that context.parent comes before context. That way, when copying
        children graphs, their parent graphs will have also been copied, so we
        can access their free variables.

        self.last is a dictionary mapping each of the original graph to the
        last context that uses it.
        """
        seen = set()
        self.last = {}
        self.tasks = []

        def _process_ctx(ctx, orig_ctx):
            if ctx in seen:
                return
            self.manager.add_graph(ctx.graph, root=True)
            seen.add(ctx)
            if ctx.parent != Context.empty():
                orig_parent_ctx = self.specializations[ctx.parent]
                _process_ctx(ctx.parent, orig_parent_ctx)
            self.last[ctx.graph] = ctx
            self.tasks.append([ctx, orig_ctx])

        for ctx, orig_ctx in self.specializations.items():
            _process_ctx(ctx, orig_ctx)

    #################
    # Create graphs #
    #################

    def create_graphs(self):
        """Create the (empty) graphs associated to the contexts.

        If self.reuse_existing is True, the last context that uses a graph will
        be monomorphized in place, that is to say, into the original graph. The
        exception to that is any graph that has the 'reference' flag. This flag
        indicates the graph is a "reference graph" that may be used elsewhere
        or reintroduced and we should not modify it.
        """
        for entry in self.tasks:
            ctx, orig_ctx = entry
            if (self.reuse_existing
                    and self.last[ctx.graph] is ctx
                    and not ctx.graph.has_flags('reference')):
                newgraph = ctx.graph
            else:
                newgraph = ctx.graph.make_new(relation=next(_count))
                newgraph.set_flags(reference=False)
            self.results[ctx] = newgraph
            entry.append(newgraph)

    ################
    # Monomorphize #
    ################

    def monomorphize(self):
        """Create the monomorphized graphs.

        For each context in the computed order:

        1. Rewire the original graph according to the reroutings of various
           nodes suggested by the inferrer.
        2. If monomorphizing the original, set node.abstract for all of its
           nodes and reroute the free variables to the right monomorphized
           parent. Get the next context and goto 1.
        3. If not monomorphizing the original, clone it using _MonoRemapper.
           _MonoRemapper will clone the graph as normal, except for its free
           variables which will be connected to those of the right parent.
        4. Set node.abstract for all of the cloned graph's nodes.
        5. Undo the modifications on the original graph.
        """
        m = self.manager
        cloners = {}

        for ctx, orig_ctx, newgraph in self.tasks:

            def fix_node(node, new_node):
                if getattr(node, 'force_abstract', False):
                    new_node.abstract = node.abstract
                else:
                    ref = self.engine.ref(node, orig_ctx)
                    if ref in self.engine.cache.cache:
                        res = ref.get_resolved()
                        new_node.abstract = concretize_abstract(res)
                    else:
                        new_node.abstract = AbstractError(DEAD)
                new_node.abstract = _fix_type(new_node.abstract, self)

            def fv_function(fv, ctx=ctx):
                fv_ctx = ctx.filter(fv.graph)
                if fv_ctx in cloners:
                    return cloners[fv_ctx][fv]
                else:
                    return fv

            # Rewire the graph to clone
            with m.transact() as tr:
                for (ref, key), repl in self.replacements[ctx].items():
                    tr.set_edge(ref.node, key, repl)

            if newgraph is ctx.graph:
                for node in chain(ctx.graph.nodes, ctx.graph.constants):
                    fix_node(node, node)
                with m.transact() as tr_fv:
                    for node in ctx.graph.free_variables_direct:
                        new_node = fv_function(node)
                        if new_node is not node:
                            for user, key in m.uses[node]:
                                if user.graph is ctx.graph:
                                    tr_fv.set_edge(user, key, new_node)
                continue

            # Clone the graph
            cl = GraphCloner(
                ctx.graph,
                total=False,
                clone_children=False,
                clone_constants=True,
                graph_repl={ctx.graph: newgraph},
                remapper_class=_MonoRemapper.partial(
                    engine=self.engine,
                    fv_function=fv_function,
                )
            )
            assert cl[ctx.graph] is newgraph
            cloners[ctx] = cl

            # Populate the abstract field
            for node in chain(ctx.graph.nodes, ctx.graph.constants):
                fix_node(node, cl[node])

            # Undo changes to the original graph
            tr.undo()

    #####################
    # Fill placeholders #
    #####################

    def fill_placeholders(self):
        """Replace all placeholder constants with monomorphized graphs.

        The placeholders were created during the collect phase, since the
        monomorphized graphs are unavailable at that stage. They contain the
        context.

        This procedure will not work on a managed graph because it changes
        constants directly, therefore the manager is cleared entirely before
        doing the procedure.
        """
        self.manager.clear()
        for ctx, g in self.results.items():
            for node in dfs(g.return_, succ=succ_incoming):
                if node.is_constant(_Placeholder):
                    node.value = self.results[node.value.context]
                    node.abstract = AbstractFunction(
                        GraphFunction(node.value, Context.empty())
                    )


class _MonoRemapper(CloneRemapper):
    """Special remapper used by Monomorphizer to clone graphs.

    Arguments:
        graph_repl: Should map the graph to clone to the graph to use for
            the clone. The remapper will not create it.
        fv_function: A function that takes a free variable and returns a
            replacement free variable pointing to the right parent graph.

    """

    def __init__(self,
                 graphs,
                 inlines,
                 manager,
                 relation,
                 graph_relation,
                 clone_constants,
                 engine,
                 graph_repl,
                 fv_function):
        """Initialize the _MonoRemapper."""
        super().__init__(
            graphs=graphs,
            inlines=inlines,
            manager=manager,
            relation=relation,
            graph_repl=graph_repl,
            graph_relation=graph_relation,
            clone_constants=clone_constants,
            set_abstract=False,
        )
        self.engine = engine
        self.fv_function = fv_function

    def generate(self):
        """Does nothing, because all the graphs were already generated."""
        pass

    def gen_constant_graph(self, g, ng, ct):
        """Constant graphs that get through here cannot be reachable."""
        with About(ct.debug, self.relation):
            new = _const(DEAD, AbstractError(DEAD))
            self.remap_node(ct, g, ct, ng, new)

    def gen_fv_direct(self, g, ng, fv):
        """Remap the free variables we want to remap."""
        new = self.fv_function(fv)
        if new is not None:
            self.remap_node((g, fv), g, fv, ng, new, link=False)


def monomorphize(engine, root_context, reuse_existing=True):
    """Monomorphize all graphs starting with the given root context."""
    mono = Monomorphizer(engine, reuse_existing=reuse_existing)
    return mono.run(root_context)


__all__ = [
    'Monomorphizer',
    'Unspecializable',
    'concretize_cache',
    'monomorphize',
]
