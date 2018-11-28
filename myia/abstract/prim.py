
import numpy as np
import inspect
from functools import reduce
from collections import defaultdict
from operator import getitem

from .base import (
    ABSENT,
    AbstractBase,
    AbstractValue,
    AbstractScalar,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    PartialApplication,
    Possibilities,
)

from .inf import (
    AbstractTrack,
    XInferrer,
    GraphXInferrer,
)

from .. import dtype, dshape
from ..dtype import Number, Bool
from ..infer import ANYTHING, InferenceVar, Context, MyiaTypeError, \
    InferenceError, reify
from ..infer.core import Pending
from ..prim import ops as P
from ..utils import Namespace


abstract_inferrer_constructors = {}


def reg(*prims):
    def deco(cls):
        for prim in prims:
            abstract_inferrer_constructors[prim] = cls
        return cls
    return deco


class StandardXInferrer(XInferrer):
    def __init__(self, infer):
        super().__init__()
        self._infer = infer
        data = inspect.getfullargspec(infer)
        assert data.varkw is None
        assert data.defaults is None
        assert data.kwonlyargs == []
        assert data.kwonlydefaults is None
        self.nargs = None if data.varargs else len(data.args) - 1
        self.typemap = {}
        for name, ann in data.annotations.items():
            self.typemap[data.args.index(name) - 1] = ann

    async def infer(self, track, *args):
        if self.nargs is not None and len(args) != self.nargs:
            raise MyiaTypeError('Wrong number of arguments.')
        for i, arg in enumerate(args):
            typ = self.typemap.get(i)
            if typ is None:
                pass
            elif dtype.ismyiatype(typ):
                await track.check(typ, arg.values['type'])
            elif issubclass(typ, AbstractBase):
                if not isinstance(arg, typ):
                    raise MyiaTypeError('Wrong type')
        return await self._infer(track, *args)


def standard_prim(*prims):
    def deco(fn):
        xinf = StandardXInferrer.partial(infer=fn)
        for prim in prims:
            abstract_inferrer_constructors[prim] = xinf
    return deco


class WithImplXInferrer(XInferrer):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl
        data = inspect.getfullargspec(impl)
        assert data.varargs is None
        assert data.varkw is None
        assert data.defaults is None
        assert data.kwonlyargs == []
        assert data.kwonlydefaults is None
        self.nargs = len(data.args)
        self.impl_data = data

    def run_impl(self, args, outtype):
        min_count = min((arg.count for arg in args), default=0)
        if min_count <= 0:
            outval = ANYTHING
        else:
            values = [arg.values['value'] for arg in args]
            if any(v is ANYTHING for v in values):
                outval = ANYTHING
            else:
                outval = self.impl(*values)

        if outval is ANYTHING:
            min_count = 0
        rval = AbstractScalar({
            'value': outval,
            'type': outtype,
            'shape': dshape.NOSHAPE
        })
        rval.count = max(min_count - 1, 0)
        return rval


class UniformPrimitiveXInferrer(WithImplXInferrer):
    def __init__(self, impl):
        super().__init__(impl)
        data = self.impl_data
        self.typemap = defaultdict(list)
        for i, arg in enumerate(data.args):
            self.typemap[data.annotations[arg]].append(i)
        self.outtype = data.annotations['return']

    async def infer(self, track, *args):
        if len(args) != self.nargs:
            raise MyiaTypeError('Wrong number of arguments.')

        outtype = self.outtype
        ts = [arg.values['type'] for arg in args]
        for typ, indexes in self.typemap.items():
            selection = [ts[i] for i in indexes]
            # res = await track.will_check(typ, *selection)
            res = track.chk(typ, *selection)
            if typ == self.outtype:
                outtype = res

        return self.run_impl(args, outtype)


def uniform_prim(prim):
    def deco(fn):
        xinf = UniformPrimitiveXInferrer.partial(impl=fn)
        abstract_inferrer_constructors[prim] = xinf
    return deco


class MyiaNameError(InferenceError):
    """Raised when a name is not found in scope."""


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


async def static_getter(track, data, item, fetch, on_dcattr, chk=None):
    """Return an inferrer for resolve or getattr.

    Arguments:
        track: The track on which the inference operates.
        data: A ref to the data.
        item: A ref to the item/attribute.
        fetch: A function to resolve the item on the data.
        chk: A function to check the values inferred for the
            data and item.
    """
    resources = track.engine.pipeline.resources

    data_t = data.build('type')
    item_v = item.build('value')

    if item_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )

    if isinstance(data_t, InferenceVar):
        case, *args = await find_coherent_result(
            data_t,
            lambda t: _resolve_case(resources, t, item_v, chk)
        )
    else:
        case, *args = await _resolve_case(resources, data_t, item_v, chk)

    if case == 'class':
        # Get field from Class
        if item_v in data_t.attributes:
            return await on_dcattr(data, data_t, item_v)
        elif item_v in data_t.methods:
            method = data_t.methods[item_v]
            method = resources.convert(method)
            inferrer = track.from_value(method, Context.empty())
            fns = inferrer.values['value']
            assert isinstance(fns, Possibilities)
            return AbstractScalar({
                'value': Possibilities(
                    PartialApplication(fn, (data,)) for fn in fns
                ),
                'type': dtype.Function,
                'shape': dshape.NOSHAPE,
            })
        else:
            raise InferenceError(f'Unknown field in {data_t}: {item_v}')

    elif case == 'method':
        method, = args
        method = resources.convert(method)
        inferrer = track.from_value(method, Context.empty())
        fns = inferrer.values['value']
        assert isinstance(fns, Possibilities)
        return AbstractScalar({
            'value': Possibilities(
                PartialApplication(fn, (data,)) for fn in fns
            ),
            'type': dtype.Function,
            'shape': dshape.NOSHAPE,
        })

    elif case == 'no_method':
        msg = f"object of type {data_t} has no attribute '{item_v}'"
        raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = data.build('value')
        if data_v is ANYTHING:
            raise InferenceError(
                'Could not infer the type or the value of the object'
                f" on which to resolve the attribute '{item_v}"
            )
        if chk:
            chk(data_v, item_v)
        try:
            raw = fetch(data_v, item_v)
        except NameError:
            raise MyiaNameError(f"Cannot resolve name '{item_v}'")
        except AttributeError as e:
            raise MyiaAttributeError(str(e))
        except Exception as e:  # pragma: no cover
            raise InferenceError(f'Unexpected error in getter: {e!r}')
        value = resources.convert(raw)
        return track.from_value(value, Context.empty())


async def _resolve_case(resources, data_t, item_v, chk):
    mmap = resources.method_map

    if dtype.ismyiatype(data_t, dtype.Class):
        return ('class', data_t)

    # Try method map
    try:
        mmap_t = mmap[data_t]
    except KeyError:
        mmap_t = None

    if mmap_t is not None:
        # Method call
        if chk:
            chk(None, item_v)
        if item_v in mmap_t:
            method = mmap_t[item_v]
            return ('method', method)
        else:
            return ('no_method',)

    return ('static',)


##############
# Arithmetic #
##############


@uniform_prim(P.scalar_add)
def _inf_scalar_add(x: Number, y: Number) -> Number:
    return x + y


@uniform_prim(P.scalar_sub)
def _inf_scalar_sub(x: Number, y: Number) -> Number:
    return x - y


@uniform_prim(P.scalar_mul)
def _inf_scalar_mul(x: Number, y: Number) -> Number:
    return x * y


@uniform_prim(P.scalar_div)
def _inf_scalar_div(x: Number, y: Number) -> Number:
    if isinstance(x, (float, np.floating)):
        return x / y
    else:
        return int(x / y)


@uniform_prim(P.scalar_mod)
def _inf_scalar_mod(x: Number, y: Number) -> Number:
    return x % y


@uniform_prim(P.scalar_pow)
def _inf_scalar_pow(x: Number, y: Number) -> Number:
    return x ** y


@uniform_prim(P.scalar_trunc)
def _inf_scalar_trunc(x: Number) -> Number:
    return np.trunc(x)


@uniform_prim(P.scalar_floor)
def _inf_scalar_floor(x: Number) -> Number:
    return np.floor(x)


@uniform_prim(P.scalar_uadd)
def _inf_scalar_uadd(x: Number) -> Number:
    return x


@uniform_prim(P.scalar_usub)
def _inf_scalar_usub(x: Number) -> Number:
    return -x


@uniform_prim(P.scalar_exp)
def _inf_scalar_exp(x: Number) -> Number:
    return math.exp(x)


@uniform_prim(P.scalar_log)
def _inf_scalar_log(x: Number) -> Number:
    return math.log(x)


@uniform_prim(P.scalar_sin)
def _inf_scalar_sin(x: Number) -> Number:
    return math.sin(x)


@uniform_prim(P.scalar_cos)
def _inf_scalar_cos(x: Number) -> Number:
    return math.cos(x)


@uniform_prim(P.scalar_tan)
def _inf_scalar_tan(x: Number) -> Number:
    return math.tan(x)


###############
# Comparisons #
###############


@uniform_prim(P.scalar_eq)
def _inf_scalar_eq(x: Number, y: Number) -> Bool:
    return x == y


@uniform_prim(P.scalar_lt)
def _inf_scalar_lt(x: Number, y: Number) -> Bool:
    return x < y


@uniform_prim(P.scalar_gt)
def _inf_scalar_gt(x: Number, y: Number) -> Bool:
    return x > y


@uniform_prim(P.scalar_ne)
def _inf_scalar_ne(x: Number, y: Number) -> Bool:
    return x != y


@uniform_prim(P.scalar_le)
def _inf_scalar_le(x: Number, y: Number) -> Bool:
    return x <= y


@uniform_prim(P.scalar_ge)
def _inf_scalar_ge(x: Number, y: Number) -> Bool:
    return x >= y


@uniform_prim(P.bool_not)
def _inf_bool_not(x: Bool) -> Bool:
    return x >= y


@uniform_prim(P.bool_and)
def _inf_bool_and(x: Bool, y: Bool) -> Bool:
    return x and y


@uniform_prim(P.bool_or)
def _inf_bool_or(x: Bool, y: Bool) -> Bool:
    return x or y


@uniform_prim(P.bool_eq)
def _inf_bool_eq(x: Bool, y: Bool) -> Bool:
    return x == y


######################
# Type introspection #
######################


# typeof = Primitive('typeof')
# hastype = Primitive('hastype')


###################
# Data structures #
###################


@standard_prim(P.make_tuple)
async def _inf_make_tuple(track, *args):
    return AbstractTuple(args)


# make_list = Primitive('make_list')
# make_record = Primitive('make_record')
# tuple_getitem = Primitive('tuple_getitem')
# list_getitem = Primitive('list_getitem')
# array_getitem = Primitive('array_getitem')


@standard_prim(P.tuple_getitem)
async def _inf_tuple_getitem(track, arg: AbstractTuple, idx: dtype.Int[64]):
    i = idx.values['value']
    return arg.elements[i]


# list_setitem = Primitive('list_setitem')
# array_setitem = Primitive('array_setitem')
# list_append = Primitive('list_append')


@standard_prim(P.getattr)
async def _inf_getattr(track, data, item):
    def chk(data_v, item_v):
        if not isinstance(item_v, str):  # pragma: no cover
            raise MyiaTypeError(
                f'item argument to resolve must be a string, not {item_v}.'
            )

    async def on_dcattr(data, data_t, item_v):
        return data_t.attributes[item_v]

    return await static_getter(
        track, data, item,
        fetch=getattr,
        on_dcattr=on_dcattr,
        chk=chk
    )


# setattr = Primitive('setattr')


@standard_prim(P.tuple_len)
async def _inf_tuple_len(track, xs: AbstractTuple):
    return AbstractScalar({
        'value': len(xs.elements),
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


@standard_prim(P.list_len)
async def _inf_list_len(track, xs: AbstractList):
    return AbstractScalar({
        'value': ANYTHING,
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


@standard_prim(P.array_len)
async def _inf_array_len(track, xs: AbstractArray):
    return AbstractScalar({
        'value': ANYTHING,
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


# list_map = Primitive('list_map')
# list_reduce = Primitive('list_reduce')


##########
# Arrays #
##########


# scalar_to_array = Primitive('scalar_to_array')
# array_to_scalar = Primitive('array_to_scalar')
# broadcast_shape = Primitive('broadcast_shape')
# invert_permutation = Primitive('invert_permutation')
# shape = Primitive('shape')
# array_map = Primitive('array_map')
# array_scan = Primitive('array_scan')
# array_reduce = Primitive('array_reduce')
# distribute = Primitive('distribute')
# reshape = Primitive('reshape')
# transpose = Primitive('transpose')
# dot = Primitive('dot')


##############
# Statements #
##############


@standard_prim(P.switch)
async def switch_prim(track, cond: Bool, tb, fb):
    v = cond.values['value']
    if v is True:
        return tb
    elif v is False:
        return fb
    elif v is ANYTHING:
        return track.abstract_merge(tb, fb)
    else:
        raise AssertionError("Invalid condition value for switch")


#################
# Miscellaneous #
#################


# scalar_cast = Primitive('scalar_cast')


@standard_prim(P.identity, P.return_)
async def _inf_identity(track, x):
    return x


@standard_prim(P.resolve)
async def _inf_resolve(track, data, item):
    def chk(data_v, item_v):
        if not isinstance(data_v, Namespace):  # pragma: no cover
            raise MyiaTypeError(
                f'data argument to resolve must be Namespace, not {data_v}',
                refs=[data]
            )
        if not isinstance(item_v, str):  # pragma: no cover
            raise MyiaTypeError(
                f'item argument to resolve must be a string, not {item_v}.',
                refs=[item]
            )

    async def on_dcattr(data, data_t, item_v):  # pragma: no cover
        raise MyiaTypeError('Cannot resolve on Class.')

    return await static_getter(
        track, data, item,
        fetch=getitem,
        on_dcattr=on_dcattr,
        chk=chk
    )


@standard_prim(P.partial)
async def partial_prim(track, fn, *args):
    fns = fn.values['value']
    assert isinstance(fns, Possibilities)
    return AbstractScalar({
        'value': Possibilities([
            PartialApplication(fn, args) for fn in fns
        ]),
        'type': dtype.Function,
        'shape': dshape.NOSHAPE
    })


# J = Primitive('J')
# Jinv = Primitive('Jinv')
# embed = Primitive('embed')
# env_setitem = Primitive('env_setitem')
# env_getitem = Primitive('env_getitem')
# env_add = Primitive('env_add')
