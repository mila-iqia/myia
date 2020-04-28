"""Monomorphization algorithm.

Monomorphization creates a separate function for each type signature the
function may be called with.
"""

from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from functools import reduce
from itertools import count
from typing import Optional
from warnings import warn

from .abstract import (
    DEAD,
    DUMMY,
    POLY,
    AbstractError,
    AbstractFunction,
    AbstractJTagged,
    AbstractTuple,
    CheckState,
    CloneState,
    Context,
    DummyFunction,
    GraphFunction,
    GraphInferrer,
    JTransformedFunction,
    MetaGraphFunction,
    PartialApplication,
    PrimitiveFunction,
    Reference,
    TrackedInferrer,
    TypedPrimitive,
    VirtualFunction,
    VirtualFunction2,
    VirtualReference,
    abstract_check,
    abstract_clone,
    amerge,
    broaden,
    build_value,
    compute_bprop_type,
    concretize_abstract,
    concretize_cache,
    no_tracking_id,
    refmap,
)
from .abstract.infer import VirtualInferrer
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
from .utils import InferenceError, MyiaTypeError, OrderedSet, overload


class Unspecializable(Exception):
    """Raised when it is impossible to specialize an inferrer."""

    def __init__(self, problem, data=None):
        """Initialize Unspecializable."""
        super().__init__(problem)
        self.problem = problem
        self.data = data


@abstract_check.variant(initial_state=lambda: CheckState({}, "_fixed"))
def _chk(
    self,
    a: (
        GraphFunction,
        PartialApplication,
        JTransformedFunction,
        PrimitiveFunction,
        MetaGraphFunction,
        TypedPrimitive,
    ),
    finder,
    monomorphizer,
):
    return False


@abstract_clone.variant(
    initial_state=lambda: CloneState(cache={}, prop="_fixed", check=_chk)
)
def _fix_type(self, a: GraphFunction, finder, monomorphizer):
    assert a.graph.abstract is None
    if a.tracking_id in monomorphizer.ctcache:
        ctx = monomorphizer.ctcache[a.tracking_id]
        g = monomorphizer.results[ctx]
        return VirtualFunction2(
            tuple(
                self(p.abstract, finder, monomorphizer) for p in g.parameters
            ),
            self(g.return_.abstract, finder, monomorphizer),
        )
    else:
        # return DummyFunction()
        return AbstractError(DUMMY)


@overload  # noqa: F811
def _fix_type(self, a: PartialApplication, finder, monomorphizer):
    vfn = self(a.fn, finder, monomorphizer)
    if isinstance(vfn, AbstractError) and vfn.xvalue() is DUMMY:
        return vfn
    assert isinstance(vfn, VirtualFunction2)
    vfn = VirtualFunction2(vfn.args[len(a.args) :], vfn.output)
    return vfn


@overload  # noqa: F811
def _fix_type(self, a: VirtualFunction, finder, monomorphizer):
    return (yield VirtualFunction2)(
        tuple(self(arg, finder, monomorphizer) for arg in a.args),
        self(a.output, finder, monomorphizer),
    )


@overload  # noqa: F811
def _fix_type(self, a: JTransformedFunction, finder, monomorphizer):
    def _jtag(x):
        if isinstance(x, AbstractFunction):
            v = x.get_sync()
            rval = AbstractFunction(
                *[
                    self(JTransformedFunction(poss), finder, monomorphizer)
                    for poss in v
                ]
            )
        elif isinstance(x, VirtualFunction2):
            return self(JTransformedFunction(x), finder, monomorphizer)
        else:
            rval = AbstractJTagged(self(x, finder, monomorphizer))
        return rval

    vfn = self(a.fn, finder, monomorphizer)
    jargs = tuple(_jtag(arg) for arg in vfn.args)
    jres = _jtag(vfn.output)
    bprop = compute_bprop_type(vfn, vfn.args, vfn.output, vfn2=True)
    out = AbstractTuple([jres, bprop])
    return VirtualFunction2(jargs, out)


@overload  # noqa: F811
def _fix_type(self, a: AbstractFunction, finder, monomorphizer):
    vfns = self(a.get_sync(), finder, monomorphizer)
    if len(vfns) == 1:
        (vfn,) = vfns
    else:
        vfns = [v for v in vfns if not isinstance(v, (DummyFunction, AbstractError))]
        assert vfns and all(isinstance(v, VirtualFunction2) for v in vfns)
        vfn = VirtualFunction2(
            reduce(amerge, [v.args for v in vfns]),
            reduce(amerge, [v.output for v in vfns]),
        )
    return vfn


@overload  # noqa: F811
def _fix_type(self, a: PrimitiveFunction, finder, monomorphizer):
    try:
        return self(
            finder.analyze_function(None, a, None)[0], finder, monomorphizer
        )
    except Unspecializable:
        # return DummyFunction()
        return AbstractError(DUMMY)


@overload  # noqa: F811
def _fix_type(self, a: MetaGraphFunction, finder, monomorphizer):
    inf = finder.engine.get_inferrer_for(a)
    argvals, outval = finder._find_unique_argvals(None, inf, None)
    return VirtualFunction2(tuple(argvals), outval)


@overload  # noqa: F811
def _fix_type(self, a: TypedPrimitive, finder, monomorphizer):
    return VirtualFunction2(
        tuple(self(ar, finder, monomorphizer) for ar in a.args),
        self(a.output, finder, monomorphizer),
    )


def type_fixer(finder, monomorphizer=None):
    """Return a function to canonicalize the type of the input."""
    return lambda x: _fix_type(x, finder, monomorphizer)


_count = count(1)


def _const(v, t):
    ct = Constant(v)
    ct.abstract = t
    if t is not None:
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
    return refmap(no_tracking_id, ctx)


class TypeFinder:
    """Find a unique type for inferrers, constants, etc."""

    def __init__(self, engine):
        """Initialize a TypeFinder."""
        self.engine = engine
        self.infcaches = {}

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
            tfn = TypedPrimitive(fn.prim, argvals, outval)
            a = AbstractFunction(tfn)
            return tfn, _const(fn.prim, a), None, None

        assert isinstance(inf, GraphInferrer)
        concretize_cache(inf.graph_cache)

        ctx = inf.make_context(self.engine, argvals)
        norm_ctx = _normalize_context(ctx)
        new_ct = _const(_Placeholder(norm_ctx), None)
        return None, new_ct, ctx, norm_ctx

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
            (choice,) = choices
            argrefs = [VirtualReference(v) for v in choice]
            try:
                res = self.engine.run_coroutine(
                    inf.run(self.engine, None, argrefs)
                )
            except InferenceError:  # pragma: no cover
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
                while hasattr(currinf, "subinf"):
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
                            warn(
                                f"{currinf} failed with a {type(e)}, but it"
                                f" should only fail with an InferenceError."
                            )
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

    def _find_unique_argvals_helper(self, a, inf, argvals, try_generalize=True):
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
            assert not isinstance(inf, VirtualInferrer)
            currinf = inf
            while hasattr(currinf, "subinf"):
                currinf = currinf.subinf
            if currinf is not inf:
                return self._find_unique_argvals_helper(
                    a, currinf, argvals, try_generalize
                )
            raise Unspecializable(DEAD)
        elif try_generalize:
            generalized, outval = self._find_generalized(inf)
            if generalized is not None:
                return generalized, outval
            raise Unspecializable(POLY, (a, *choices.keys()))
        else:
            raise Unspecializable(POLY, (a, *choices.keys()))


class Monomorphizer:
    """Monomorphize graphs using inferred type information.

    Monomorphization creates a separate function for each type signature the
    function may be called with.

    Arguments:
        engine: The InferenceEngine containing the type information.

    """

    def __init__(self, resources, engine):
        """Initialize the monomorphizer."""
        self.engine = engine
        self.infer_manager = resources.infer_manager
        self.specializations = {}
        self.replacements = defaultdict(dict)
        self.results = {}
        self.ctcache = {}
        self.invmap = {}

    def run(self, context):
        """Run monomorphization."""
        self.engine.concretize_cache()
        self.finder = TypeFinder(self.engine)
        self._fix_type = type_fixer(self.finder, self)
        self.collect(context)
        self.order_tasks()
        self.create_graphs()
        self.monomorphize()
        self.fill_placeholders()
        result = self.results[context]
        self.fix_types()
        return result

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
        todo = [
            _TodoEntry(self.engine.ref(root.return_, root_context), None, None)
        ]
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
                with About(ref.node.debug, "equiv"):
                    ct = _build(a)
                    if ct is not None:
                        ref = self.engine.ref(ct, ref.context)

            new_node = ref.node

            if ref.node.is_apply():
                # Grab the argvals
                irefs = [
                    self.engine.ref(inp, entry.ref.context)
                    for inp in ref.node.inputs
                ]
                absfn = concretize_abstract(irefs[0].get_resolved())
                argvals = [
                    concretize_abstract(iref.get_resolved())
                    for iref in irefs[1:]
                ]

                prim = absfn.get_prim()
                method = None
                if prim is not None:
                    method = getattr(self, f"_special_{prim}", None)

                if method is not None:
                    method(todo, ref, irefs, argvals)
                else:
                    # Keep traversing the graph. Element 0 is special.
                    todo.append(_TodoEntry(irefs[0], tuple(argvals), (ref, 0)))
                    # Iterate through the rest of the inputs
                    for i, iref in enumerate(irefs[1:]):
                        todo.append(_TodoEntry(iref, None, (ref, i + 1)))

            elif (
                ref.node.is_constant_graph()
                or ref.node.is_constant(MetaGraph)
                or ref.node.is_constant(Primitive)
            ):

                if ref.node.is_constant_graph():
                    ctabs = ref.node.value.abstract
                else:
                    ctabs = ref.node.abstract

                if ctabs is None or not (isinstance(ctabs, VirtualFunction2) or isinstance(
                    ctabs.get_unique(), VirtualFunction
                )):
                    fn = a.get_unique()
                    with About(ref.node.debug, "equiv"):
                        try:
                            (
                                _,
                                new_node,
                                ctx,
                                norm_ctx,
                            ) = self.finder.analyze_function(
                                a, fn, entry.argvals
                            )
                            if (
                                norm_ctx
                                and norm_ctx not in self.specializations
                            ):
                                self.specializations[norm_ctx] = ctx
                        except Unspecializable as e:
                            aerr = AbstractError(e.problem, e.data)
                            new_node = _const(e.problem, aerr)
                        else:
                            if isinstance(fn, GraphFunction):
                                self.ctcache[ref.node] = norm_ctx
                                if fn.tracking_id:
                                    self.ctcache[fn.tracking_id] = norm_ctx

                            if norm_ctx is not None:
                                retref = self.engine.ref(
                                    norm_ctx.graph.return_, norm_ctx
                                )
                                todo.append(_TodoEntry(retref, None, None))

            if new_node is not entry.ref.node:
                if entry.link is None:
                    raise AssertionError("Cannot replace a return node.")
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
        """
        seen = set()
        self.tasks = []

        def _process_ctx(ctx, orig_ctx):
            if ctx in seen or ctx in self.results:
                return
            self.infer_manager.add_graph(ctx.graph, root=True)
            seen.add(ctx)
            if ctx.parent != Context.empty():
                orig_parent_ctx = self.specializations[ctx.parent]
                _process_ctx(ctx.parent, orig_parent_ctx)
            self.tasks.append([ctx, orig_ctx])

        for ctx, orig_ctx in self.specializations.items():
            _process_ctx(ctx, orig_ctx)

    #################
    # Create graphs #
    #################

    def create_graphs(self):
        """Create the (empty) graphs associated to the contexts."""
        for entry in self.tasks:
            ctx, orig_ctx = entry
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
        m = self.infer_manager
        cloners = {}

        for ctx, orig_ctx, newgraph in self.tasks:

            def fv_function(fv, ctx=ctx):
                fv_ctx = ctx.filter(fv.graph)
                assert fv_ctx in cloners
                return cloners[fv_ctx][fv]

            # Rewire the graph to clone
            with m.transact() as tr:
                for (ref, key), repl in self.replacements[ctx].items():
                    tr.set_edge(ref.node, key, repl)

            # Clone the graph
            cl = GraphCloner(
                ctx.graph,
                total=False,
                clone_children=False,
                clone_constants=True,
                graph_repl={ctx.graph: newgraph},
                remapper_class=_MonoRemapper.partial(
                    engine=self.engine, fv_function=fv_function
                ),
            )
            assert cl[ctx.graph] is newgraph
            cloners[ctx] = cl

            # Populate the abstract field
            for old_node, new_node in cl.remapper.repl.items():
                if isinstance(old_node, tuple):
                    old_node = old_node[1]
                self.invmap[new_node] = self.engine.ref(old_node, orig_ctx)

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
        for ctx, orig_ctx, g in self.tasks:
            for node in dfs(g.return_, succ=succ_incoming):
                if node.is_constant(_Placeholder):
                    node.value = self.results[node.value.context]
                    node.abstract = AbstractFunction(
                        GraphFunction(node.value, Context.empty())
                    )

    #############
    # Fix types #
    #############

    def fix_types(self):
        """Fix all node types."""
        all_nodes = OrderedSet()
        for ctx, orig_ctx, g in self.tasks:
            all_nodes.update(g.parameters)
            all_nodes.update(dfs(g.return_, succ=succ_incoming))

        for node in all_nodes:
            old_ref = self.invmap.get(node, None)
            old_ref = _normalize_context(old_ref)
            if old_ref is None:
                assert node.abstract is not None
            elif getattr(old_ref.node, "force_abstract", False):
                assert old_ref.node.abstract is not None
                node.abstract = old_ref.node.abstract
            elif old_ref in self.engine.cache.cache:
                node.abstract = old_ref.get_resolved()
            else:
                node.abstract = AbstractError(DEAD)

        for node in all_nodes:
            if node.is_constant(Graph):
                node.abstract = node.value.abstract
            node.abstract = self._fix_type(node.abstract)
            assert node.abstract is not None


class _MonoRemapper(CloneRemapper):
    """Special remapper used by Monomorphizer to clone graphs.

    Arguments:
        graph_repl: Should map the graph to clone to the graph to use for
            the clone. The remapper will not create it.
        fv_function: A function that takes a free variable and returns a
            replacement free variable pointing to the right parent graph.

    """

    def __init__(
        self,
        graphs,
        inlines,
        manager,
        relation,
        graph_relation,
        clone_constants,
        engine,
        graph_repl,
        fv_function,
    ):
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
        assert ct.value.abstract is None
        with About(ct.debug, self.relation):
            new = _const(DEAD, AbstractError(DEAD))
            self.remap_node(ct, g, ct, ng, new)

    def gen_fv_direct(self, g, ng, fv):
        """Remap the free variables we want to remap."""
        new = self.fv_function(fv)
        if new is not None:
            self.remap_node((g, fv), g, fv, ng, new, link=False)


__all__ = [
    "Monomorphizer",
    "Unspecializable",
    "concretize_cache",
]
