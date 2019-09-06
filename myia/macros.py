"""Macros for Myia.

A macro transforms a subgraph into another. It is run during inference, which
means it has access to type information.
"""

from collections import defaultdict
from functools import reduce

from . import abstract, operations
from .abstract import (
    ALIASID,
    ANYTHING,
    AbstractArray,
    AbstractClassBase,
    Pending,
    build_value,
    find_coherent_result,
    force_pending,
    generate_getters,
    macro,
    setter_from_getter,
    type_to_abstract,
    union_simplify,
)
from .composite import gadd
from .info import About, DebugInfo
from .ir import (
    CloneRemapper,
    Constant,
    Graph,
    GraphCloner,
    MetaGraph,
    MultitypeGraph,
    Parameter,
    sexp_to_node,
)
from .prim import ops as P
from .prim.py_implementations import scalar_cast, scalar_to_array, typeof
from .utils import (
    Cons,
    Empty,
    InferenceError,
    MyiaAttributeError,
    MyiaNameError,
    MyiaTypeError,
    Named,
    Namespace,
    check_nargs,
    core,
)
from .xtype import Bool, Number


@macro
async def isinstance_(info):
    """Map isinstance to hastype."""
    r_data, r_type = check_nargs('isinstance', 2, info.argrefs)
    ts = build_value(await r_type.get(), default=ANYTHING)
    if not isinstance(ts, tuple):
        ts = (ts,)
    for t in ts:
        if not isinstance(t, type):
            if not (isinstance(t, AbstractClassBase)
                    and t.user_defined_version() is t):
                raise MyiaTypeError(
                    'isinstance expects a Python type'
                    ' or a tuple of Python types'
                )
    hastypes = [info.graph.apply(P.hastype, r_data.node,
                                 Constant(type_to_abstract(t)))
                for t in ts]
    return reduce(lambda x, y: info.graph.apply(P.bool_or, x, y),
                  hastypes)


@macro
async def make_list(info):
    """Create a list using Cons and Empty."""
    g = info.graph
    lst = g.apply(Empty)
    if info.abstracts == []:
        return lst
    restype = info.engine.abstract_merge(*info.abstracts)
    for arg in reversed(info.args):
        lst = g.apply(Cons, arg, lst)
    return g.apply(P.unsafe_static_cast, lst, abstract.listof(restype))


@macro(infer_args=False)
async def apply(info):
    """Expand a varargs and keyword args call."""
    fnref, *grouprefs = info.argrefs
    expanded = []
    g = info.graph
    for gref in grouprefs:
        t = await gref.get()
        if isinstance(t, abstract.AbstractDict):
            for k in t.entries:
                extract = g.apply(P.dict_getitem, gref.node, k)
                mkkw = g.apply(P.make_kwarg, k, extract)
                expanded.append(mkkw)
        elif isinstance(t, abstract.AbstractTuple):
            for i, _ in enumerate(t.elements):
                expanded.append(g.apply(P.tuple_getitem, gref.node, i))
        else:
            raise MyiaTypeError(
                'Can only expand tuple or dict in function application'
            )
    return g.apply(fnref.node, *expanded)


async def _resolve_case(resources, data, data_t, item_v):
    mmap = resources.method_map
    is_cls = isinstance(data, abstract.AbstractClassBase)

    # Try method map
    try:
        mmap_t = data_t and mmap[data_t]
    except KeyError:
        mmap_t = {} if is_cls else None

    if mmap_t is not None:
        # Method call
        if item_v in mmap_t:
            method = mmap_t[item_v]
            return ('method', method)
        elif is_cls:
            if item_v in data.attributes:
                return ('field', item_v)
            elif hasattr(data_t, item_v):
                return ('method', getattr(data_t, item_v))
            else:
                return ('no_method',)
        else:
            return ('no_method',)

    return ('static',)


@macro
async def getattr_(info):
    """Get an attribute from an object."""
    r_data, r_attr = check_nargs('getattr', 2, info.argrefs)
    data, attr = info.abstracts

    if isinstance(data, abstract.AbstractUnion):
        g = info.graph
        currg = g
        opts = await abstract.force_pending(data.options)
        for i, opt in enumerate(opts):
            last = (i == len(opts) - 1)
            if last:
                falseg = None
                cast = currg.apply(P.unsafe_static_cast, r_data.node, opt)
                out = currg.apply(operations.getattr, cast, r_attr.node)
            else:
                trueg = Graph()
                falseg = Graph()
                cond = currg.apply(P.hastype, r_data.node, opt)
                cast = trueg.apply(P.unsafe_static_cast, r_data.node, opt)
                trueg.output = trueg.apply(operations.getattr, cast,
                                           r_attr.node)
                info.engine.mng.add_graph(trueg)
                out = currg.apply(P.switch, cond, trueg, falseg)
                out = currg.apply(out)
            if currg is g:
                rval = out
            else:
                currg.output = out
                info.engine.mng.add_graph(currg)
            currg = falseg
        return rval

    data_t = data.xtype()
    attr_v = build_value(attr, default=ANYTHING)
    g = info.outref.node.graph

    if attr_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )
    elif not isinstance(attr_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'Argument to getattr must be a string, not {attr_v}.'
        )

    resources = info.engine.resources
    if isinstance(data_t, Pending):
        case, *args = await find_coherent_result(
            data_t,
            lambda t: _resolve_case(resources, data, t, attr_v)
        )
    else:
        case, *args = await _resolve_case(resources, data, data_t, attr_v)

    if case == 'field':
        # Get field from Class
        return g.apply(P.record_getitem, r_data.node, attr_v)

    elif case == 'method':
        method, = args
        if isinstance(method, property):
            return g.apply(method.fget, r_data.node)
        else:
            return g.apply(P.partial, method, r_data.node)

    elif case == 'no_method':
        msg = f"object of type {data} has no attribute '{attr_v}'"
        raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = build_value(data, default=ANYTHING)
        if data_v is ANYTHING:
            raise InferenceError(
                'Could not infer the type or the value of the object'
                f" on which to resolve the attribute '{attr_v}'"
            )
        try:
            raw = getattr(data_v, attr_v)
        except AttributeError as e:
            raise MyiaAttributeError(str(e))
        except Exception as e:  # pragma: no cover
            raise InferenceError(f'Unexpected error in getter: {e!r}')
        return Constant(raw)


@macro
async def resolve(info):
    """Perform static name resolution on a Namespace."""
    data, item = check_nargs('resolve', 2, info.abstracts)
    data_v = abstract.build_value(data)
    item_v = abstract.build_value(item)
    if not isinstance(data_v, Namespace):  # pragma: no cover
        raise MyiaTypeError(
            f'data argument to resolve must be Namespace,'
            f' not {data_v}',
        )
    if not isinstance(item_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'item argument to resolve must be a string,'
            f' not {item_v}.',
        )
    try:
        resolved = data_v[item_v]
    except NameError:
        raise MyiaNameError(f"Cannot resolve name '{item_v}'")
    return Constant(resolved)


@macro
async def dict_values(info):
    """Implement dict.values()."""
    dref, = check_nargs('dict_values', 1, info.argrefs)
    typ, = info.abstracts
    assert isinstance(typ, abstract.AbstractDict)
    getters = [info.graph.apply(P.dict_getitem, dref.node, k)
               for k in typ.entries]
    return info.graph.apply(P.make_tuple, *getters)


class _CastRemapper(CloneRemapper):

    def __init__(self,
                 graphs,
                 inlines,
                 manager,
                 relation,
                 graph_relation,
                 clone_constants,
                 graph_repl,
                 fv_replacements):
        """Initialize the GraphCloner."""
        super().__init__(
            graphs=graphs,
            inlines=inlines,
            manager=manager,
            relation=relation,
            graph_repl=graph_repl,
            graph_relation=graph_relation,
            clone_constants=clone_constants,
        )
        self.fv_replacements = fv_replacements

    def gen_fv(self, g, ng, fv):
        """Remap the free variables we want to remap."""
        if fv in self.fv_replacements:
            new = self.fv_replacements[fv]
            self.remap_node((g, fv), g, fv, ng, new, link=False)


@macro(infer_args=False)
async def user_switch(info):
    """Implement the switch functionality generated by the parser.

    If user_switch finds a Union in the condition, it will infer the value of
    the condition for each type in the union. If the condition is necessarily
    true or false for some types, the type of the variable for the
    corresponding conditional branch will be set to these types.
    """
    engine = info.engine
    g = info.graph
    condref, tbref, fbref = check_nargs(P.switch, 3, info.argrefs)

    async def type_trials(focus, opnode, argrefs):
        """Handle `user_switch(hastype(x, typ), tb, fb)`.

        We want to evaluate tb in a context where x has type typ and fb
        in a context where it doesn't.
        """
        def cond_trial(cg, opt):
            # For each possible type we make a "cond trial" which replaces the
            # focus input in the condition function by one that's cast to the
            # type. We can thus check if the value of the condition depends
            # directly on the type.
            return cg.apply(opnode,
                            *nodes[:focus],
                            cg.apply(P.unsafe_static_cast, nodes[focus], opt),
                            *nodes[focus + 1:])

        async def wrap(branch_ref, branch_type):
            # We transform branch_graph into a new graph which refers to a cast
            # version of x. We also transform all of the children of x's graph
            # so that closures called in the branch also refer to the cast
            # version of x.
            branch_graph = branch_ref.node.value
            if branch_graph not in xg.scope:
                return branch_graph
            rval = branch_graph.make_new(relation='copy')
            cast = rval.apply(P.unsafe_static_cast, xref.node, branch_type)
            cl = GraphCloner(
                *xg.children,
                total=False,
                graph_repl={branch_graph: rval},
                remapper_class=_CastRemapper.partial(
                    fv_replacements={xref.node: cast}
                )
            )
            assert rval is cl[branch_graph]
            engine.mng.add_graph(rval)
            return rval

        nodes = [ref.node for ref in argrefs]
        xref = argrefs[focus]
        fulltype = await xref.get()
        assert isinstance(fulltype, abstract.AbstractUnion)

        xg = xref.node.graph
        cg = cond.graph
        cond_trials = [cond_trial(cg, t) for t in
                       await force_pending(fulltype.options)]
        results = [await engine.ref(node, ctx).get()
                   for node in cond_trials]

        groups = {True: [], False: [], ANYTHING: []}

        for t, result in zip(fulltype.options, results):
            assert isinstance(result, abstract.AbstractScalar)
            assert result.xtype() is Bool
            value = result.xvalue()
            groups[value].append(t)

        if groups[ANYTHING]:
            return await default()

        tbtyp = union_simplify(groups[True])
        fbtyp = union_simplify(groups[False])

        if tbtyp is None:
            return fbref
        elif fbtyp is None:
            return tbref
        else:
            new_conds = [g.apply(P.hastype, xref.node, t)
                         for t in groups[True]]
            new_cond = reduce(lambda x, y: g.apply(P.bool_or, x, y),
                              new_conds)
            new_tb = await wrap(tbref, tbtyp)
            new_fb = await wrap(fbref, fbtyp)
            return g.apply(P.switch, new_cond, new_tb, new_fb)

    async def default():
        _, _, tb, fb = info.outref.node.inputs
        return g.apply(P.switch, cond, tb, fb)

    for branch_ref in [tbref, fbref]:
        if not branch_ref.node.is_constant_graph():
            raise MyiaTypeError(
                'Branches of switch must be functions when the condition'
                ' is hastype on a Union.'
            )

    cond = condref.node
    ctx = condref.context

    condt = await condref.get()
    if not engine.check_predicate(Bool, condt):
        to_bool = engine.resources.convert(bool)
        cond = cond.graph.apply(to_bool, cond)

    if cond.is_apply():
        opnode, *args = cond.inputs
        opref = engine.ref(opnode, ctx)
        ops = (await opref.get()).get_sync()
        if len(ops) == 1:
            op, = ops
            argrefs = [engine.ref(a, ctx) for a in args]
            argtypes = [await arg.get() for arg in argrefs]
            for i, arg in enumerate(argtypes):
                if isinstance(arg, abstract.AbstractUnion):
                    return await type_trials(i, opnode, argrefs)

    return await default()


@macro
async def grad(info):
    """Create a function for the gradient of another function.

    `grad` must be called on a function, and returns a function, so to
    get df/dx one must write `grad(f)(x, y, ...)`.

    Usage:
        grad(f)(x, y)              == df/dx
        grad(f, 'y')(x, y)         == df/dy
        grad(f, 'x', 'y')(x, y)    == (df/dx, df/dy)
        grad(f, return_value=True) == (f(x, y), df/dx)
        grad(f, dout=z)            == z * df/dx, if f(x, y) is a scalar
    """
    fn, *argtypes = info.abstracts
    wrt = []

    flags = {
        'return_value': False,
    }

    for arg in argtypes:
        if isinstance(arg, abstract.AbstractKeywordArgument):
            if arg.key in flags:
                flags[arg.key] = abstract.build_value(
                    arg.argument, default=abstract.ANYTHING
                )
            else:
                raise MyiaTypeError(f'grad takes no argument named {arg.key}')
        else:
            val = abstract.build_value(arg, default=abstract.ANYTHING)
            if isinstance(val, (int, str)):
                wrt.append(val)
            else:
                raise MyiaTypeError(f'Invalid argument to grad, {arg}')

    if not isinstance(fn, abstract.AbstractFunction):
        raise MyiaTypeError(
            f"'grad' takes a function as its argument, not {fn}."
        )

    fn = fn.get_unique()
    if isinstance(fn, abstract.GraphFunction):
        arg = fn.graph
        if arg.parent is not None:
            raise MyiaTypeError(
                f"'grad' does not work on closures ('grad' was given argument"
                f" '{arg}', which is a closure with parent '{arg.parent}'.)"
            )
    elif isinstance(fn, abstract.MetaGraphFunction):
        arg = fn.metagraph
    elif isinstance(fn, abstract.PrimitiveFunction):
        arg = fn.prim
    else:
        raise MyiaTypeError(f"'grad' cannot handle {fn}")

    return Constant(GradOperation(arg, wrt, **flags))


_cast_helper = MultitypeGraph('cast_helper')


@_cast_helper.register(Number, Number)
@core
def _scalar_cast_helper(x, model):
    t = typeof(model)
    return scalar_cast(x, t)


@_cast_helper.register(Number, AbstractArray)
@core
def _scalar_to_array_cast_helper(x, model):
    t = typeof(model)
    return scalar_to_array(scalar_cast(x, t.element), t)


ROOT = Named('ROOT')


class GradOperation(MetaGraph):
    """Implements the grad(f) operation.

    This MetaGraph is returned by a call to `grad`.
    """

    def __init__(self,
                 fn,
                 wrt,
                 *,
                 return_value=False,
                 always_return_tuple=False,
                 dout_parameter=False,
                 sum_aliases=True):
        """Initialize GradOperation."""
        super().__init__('grad')
        self.fn = fn
        self.wrt = wrt
        self.return_value = return_value
        self.always_return_tuple = always_return_tuple
        self.dout_parameter = dout_parameter
        self.sum_aliases = sum_aliases

    def make_signature(self, args):
        """Make the signature.

        The signature is a pair with the first element being the signature
        generated from self.fn and the second a boolean saying whether there is
        a dout argument or not.
        """
        aliases = defaultdict(list)
        if self.sum_aliases:
            for i, arg in enumerate(args):
                for elem, getter in generate_getters(arg, ROOT):
                    aid = elem.values.get(ALIASID, None)
                    if aid is not None:
                        assert aid is not ANYTHING
                        aliases[aid].append((i, getter))
        aliases = tuple(sorted((k, tuple(v)) for k, v in aliases.items()))

        if (len(args) > 0
                and isinstance(args[-1], abstract.AbstractKeywordArgument)
                and args[-1].key == 'dout'):
            dout = 'kw'
        else:
            dout = self.dout_parameter
        if dout:
            args = args[:-1]
        if any(isinstance(arg, abstract.AbstractKeywordArgument)
               for arg in args):
            raise MyiaTypeError(
                f"Only 'dout' is valid as a keyword argument in a"
                ' grad-transformed function.'
            )
        if isinstance(self.fn, (Graph, MetaGraph)):
            sig = self.fn.make_signature(args)
        else:
            sig = (len(args),)
        return sig, dout, aliases

    def generate_graph(self, sig):
        """Make the graph for the grad.

        If wrt is an integer, the wrt-th gradient will be returned directly.
        If it is a tuple of integers, then a tuple of the specified gradients
        will be returned in the same order (duplicates are allowed).

        If self.return_value is True, a tuple will always be returned and the
        first element will be the return value of the function. The other
        elements will be the gradients.
        """
        gsig, dout, aliases = sig
        if isinstance(self.fn, (Graph, MetaGraph)):
            g = self.fn.generate_graph(gsig)
            dbg = g.debug
            nargs = len(g.parameters)
            orig_parameters = g.parameters
            orig_parameter_names = g.parameter_names
        else:
            g = self.fn
            dbg = DebugInfo()
            nargs, = gsig
            orig_parameters = [Parameter(None) for _ in range(nargs)]
            orig_parameter_names = None

        def _getindex(wrt):
            if wrt == "*":
                raise MyiaTypeError(f"'*' in grad must be the only parameter")
            elif isinstance(wrt, str):
                try:
                    return orig_parameter_names.index(wrt)
                except ValueError:
                    raise MyiaTypeError(
                        f"{g} has no argument named '{wrt}'"
                    )
            elif 0 <= wrt < nargs:
                return wrt
            else:
                raise MyiaTypeError(
                    f"Cannot get gradient with respect to argument {wrt}"
                    f" for {g} because it is out of range."
                )

        if self.wrt == ['*']:
            wrt = list(range(nargs))
        else:
            wrt = list(map(_getindex, self.wrt))
            if len(wrt) == 1:
                wrt = wrt[0]
            elif wrt == []:
                wrt = 0

        with About(dbg, 'grad'):
            df = Graph()
            df.set_flags('core', 'reference')

        jf = df.apply(P.J, g)

        params = []
        for orig_p in orig_parameters:
            with About(orig_p.debug, 'grad'):
                params.append(df.add_parameter())

        jparams = [df.apply(P.J, p) for p in params]
        app = df.apply(jf, *jparams)
        out = df.apply(P.Jinv, df.apply(P.tuple_getitem, app, 0))
        bprop = df.apply(P.tuple_getitem, app, 1)

        if dout:
            bprop_arg = df.add_parameter()
            bprop_arg.debug.name = 'dout'
            if dout == 'kw':
                bprop_arg = df.apply(P.extract_kwarg, 'dout', bprop_arg)
        else:
            bprop_arg = df.apply(_cast_helper, 1, out)

        if isinstance(wrt, int):
            direct_return = not self.always_return_tuple
            wrt = [wrt]
        else:
            direct_return = False

        bapp = df.apply(bprop, bprop_arg)
        all_results = [df.apply(P.tuple_getitem, bapp, idx + 1)
                       for idx in range(nargs)]

        adjusted = {i: all_results[i] for i in range(nargs)}
        for aid, equivs in aliases:
            contribs = []
            for i, entry in equivs:
                node = sexp_to_node(entry, df, sub={ROOT: all_results[i]})
                contribs.append(node)
            combined = reduce(lambda x, y: df.apply(gadd, x, y), contribs)

            for i, entry in equivs:
                setter = setter_from_getter(entry, combined)
                node = sexp_to_node(setter, df, sub={ROOT: adjusted[i]})
                adjusted[i] = node

        elems = [out] if self.return_value else []
        elems += [adjusted[idx] for idx in wrt]

        if len(elems) == 1 and direct_return:
            df.output = elems[0]
        else:
            df.output = df.apply(P.make_tuple, *elems)

        return df


@core
def value_and_grad(*args, **kwargs):
    """Return the value of the function along with the gradient."""
    return grad(*args, **kwargs, return_value=True)
