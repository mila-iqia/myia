"""Macros for Myia.

A macro transforms a subgraph into another. It is run during inference, which
means it has access to type information.
"""


from . import abstract, operations
from .abstract import macro
from .dtype import Array, Number
from .info import About, DebugInfo
from .ir import Constant, Graph, MetaGraph, MultitypeGraph, Parameter
from .prim import ops as P
from .prim.py_implementations import scalar_cast, scalar_to_array, typeof
from .utils import (
    Cons,
    Empty,
    InferenceError,
    MyiaTypeError,
    check_nargs,
    core,
)


@macro
async def make_list(info):
    """Create a list using Cons and Empty."""
    g = info.graph
    lst = g.apply(Empty)
    argtypes = [await arg.get() for arg in info.argrefs]
    if argtypes == []:
        return lst
    restype = info.engine.abstract_merge(*argtypes)
    for arg in reversed(info.args):
        lst = g.apply(Cons, arg, lst)
    return g.apply(P.unsafe_static_cast, lst, abstract.listof(restype))


@macro
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


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


async def _resolve_case(resources, data_t, item_v):
    mmap = resources.method_map

    if (isinstance(data_t, type)
            and issubclass(data_t, abstract.AbstractClassBase)):
        return ('class', data_t)

    # Try method map
    try:
        mmap_t = mmap[data_t]
    except KeyError:
        mmap_t = None

    if mmap_t is not None:
        # Method call
        if item_v in mmap_t:
            method = mmap_t[item_v]
            return ('method', method)
        else:
            return ('no_method',)

    return ('static',)


@macro
async def getattr_(info):
    """Get an attribute from an object."""
    from .abstract import type_token, build_value, ANYTHING, \
        find_coherent_result, Pending

    r_data, r_attr = check_nargs('getattr', 2, info.argrefs)
    data = await r_data.get()

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

    attr = await r_attr.get()
    data_t = type_token(data)
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

    resources = info.engine.pipeline.resources
    if isinstance(data_t, Pending):
        case, *args = await find_coherent_result(
            data_t,
            lambda t: _resolve_case(resources, t, attr_v)
        )
    else:
        case, *args = await _resolve_case(resources, data_t, attr_v)

    def process_method(method):
        if isinstance(method, property):
            return g.apply(method.fget, r_data.node)
        else:
            return g.apply(P.partial, method, r_data.node)

    if case == 'class':
        # Get field from Class
        if attr_v in data.attributes:
            return g.apply(P.record_getitem, r_data.node, attr_v)
        elif attr_v in data.methods:
            return process_method(data.methods[attr_v])
        else:
            raise InferenceError(f'Unknown field in {data}: {attr_v}')

    elif case == 'method':
        method, = args
        return process_method(method)

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
async def dict_values(info):
    """Implement dict.values()."""
    dref, = check_nargs('dict_values', 1, info.argrefs)
    typ = await dref.get()
    assert isinstance(typ, abstract.AbstractDict)
    getters = [info.graph.apply(P.dict_getitem, dref.node, k)
               for k in typ.entries]
    return info.graph.apply(P.make_tuple, *getters)


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
    fn, *argtypes = [await ref.get() for ref in info.argrefs]
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

    fn = fn.get_unique()
    assert isinstance(fn, abstract.GraphFunction)
    return Constant(GradOperation(fn.graph, wrt, **flags))


_cast_helper = MultitypeGraph('cast_helper')


@_cast_helper.register(Number, Number)
@core
def _scalar_cast_helper(x, model):
    t = typeof(model)
    return scalar_cast(x, t)


@_cast_helper.register(Number, Array)
@core
def _scalar_to_array_cast_helper(x, model):
    t = typeof(model)
    return scalar_to_array(scalar_cast(x, t.element), typeof(model))


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
                 apply_j=True):
        """Initialize GradOperation."""
        super().__init__('grad')
        self.fn = fn
        self.wrt = wrt
        self.return_value = return_value
        self.always_return_tuple = always_return_tuple
        self.dout_parameter = dout_parameter
        self.apply_j = apply_j

    def make_signature(self, args):
        """Make the signature.

        The signature is a pair with the first element being the signature
        generated from self.fn and the second a boolean saying whether there is
        a dout argument or not.
        """
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
        return sig, dout

    def generate_graph(self, sig):
        """Make the graph for the grad.

        If wrt is an integer, the wrt-th gradient will be returned directly.
        If it is a tuple of integers, then a tuple of the specified gradients
        will be returned in the same order (duplicates are allowed).

        If self.return_value is True, a tuple will always be returned and the
        first element will be the return value of the function. The other
        elements will be the gradients.
        """
        gsig, dout = sig
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

        jf = g
        if self.apply_j:
            jf = df.apply(P.J, jf)

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
        elems = []
        if self.return_value:
            elems.append(out)
        for idx in wrt:
            elems.append(df.apply(P.tuple_getitem, bapp, idx + 1))

        if len(elems) == 1 and direct_return:
            df.output = elems[0]
        else:
            df.output = df.apply(P.make_tuple, *elems)

        return df


@core
def value_and_grad(*args, **kwargs):
    """Return the value of the function along with the gradient."""
    return grad(*args, **kwargs, return_value=True)
