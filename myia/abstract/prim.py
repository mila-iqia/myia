
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
)

from .inf import (
    AbstractTrack,
    XInferrer,
    GraphXInferrer,
)

from .. import dtype, dshape
from ..dtype import Number, Bool
from ..infer import ANYTHING, InferenceVar, Context, MyiaTypeError, \
    InferenceError
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
            print(typ, typ and issubclass(typ, AbstractBase))
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


@standard_prim(P.identity, P.return_)
async def identity_prim(track, x):
    return x


@standard_prim(P.make_tuple)
async def make_tuple_prim(track, *args):
    return AbstractTuple(args)


@standard_prim(P.tuple_getitem)
async def tuple_getitem_prim(track, arg: AbstractTuple, idx: dtype.Int[64]):
    i = idx.values['value']
    return arg.elements[i]


class UniformPrimitiveXInferrer(XInferrer):
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
            res = await track.will_check(typ, *selection)
            if typ == self.outtype:
                outtype = res

        values = [arg.values['value'] for arg in args]
        if any(v is ANYTHING for v in values):
            outval = ANYTHING
        else:
            outval = self.impl(*values)

        return AbstractScalar({
            'value': outval,
            'type': outtype,
            'shape': dshape.NOSHAPE
        })


def uniform_prim(prim):
    def deco(fn):
        xinf = UniformPrimitiveXInferrer.partial(impl=fn)
        abstract_inferrer_constructors[prim] = xinf
    return deco


@uniform_prim(P.scalar_usub)
def prim_usub(x: Number) -> Number:
    return -x


@uniform_prim(P.scalar_add)
def prim_add(x: Number, y: Number) -> Number:
    return x + y


@uniform_prim(P.scalar_mul)
def prim_mul(x: Number, y: Number) -> Number:
    return x * y


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
            inferrer = unwrap(inferrer)
            return PartialInferrer(
                track,
                inferrer,
                [data]
            )
        else:
            raise InferenceError(f'Unknown field in {data_t}: {item_v}')

    elif case == 'method':
        method, = args
        method = resources.convert(method)
        inferrer = track.from_value(method, Context.empty())
        inferrer = unwrap(inferrer)
        return PartialInferrer(
            track,
            inferrer,
            [data]
        )

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


@standard_prim(P.resolve)
async def resolve_prim(track, data, item):
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


@standard_prim(P.getattr)
async def getattr_prim(track, data, item):
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


@standard_prim(P.array_len)
async def array_len_prim(track, xs: AbstractArray):
    return AbstractScalar({
        'value': ANYTHING,
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


@standard_prim(P.list_len)
async def list_len_prim(track, xs: AbstractList):
    return AbstractScalar({
        'value': ANYTHING,
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


@standard_prim(P.tuple_len)
async def tuple_len_prim(track, xs: AbstractTuple):
    return AbstractScalar({
        'value': len(xs.elements),
        'type': dtype.Int[64],
        'shape': dshape.NOSHAPE
    })


@standard_prim(P.switch)
async def switch_prim(self, track, cond: Bool, tb, fb):
    v = cond.values['value']
    if v is True:
        return tb
    elif v is False:
        return fb
    elif v is ANYTHING:
        return abstract_merge(tb, fb)
    else:
        raise AssertionError("Invalid condition value for switch")


def abstract_merge(*values):
    resolved = set()
    pending = set()
    committed = None
    for v in values:
        if isinstance(v, Pending):
            if v.resolved():
                resolved.add(v.result())
            else:
                pending.add(v)
        else:
            resolved.add(v)

    if pending:
        def resolve(fut):
            pending.remove(fut)
            resolved.append(fut.result())
            if not pending:
                v = force_merge(values, model=committed)
                rval.resolve_to(v)

        for p in pending:
            p.add_done_callback(resolve)

        def premature_resolve():
            nonlocal committed
            committed = force_merge(resolved)
            resolved.clear()
            return committed

        rval = Pending(premature_resolve)
        return rval
    else:
        return force_merge(resolved)


def force_merge(values, model=None):
    if model is None:
        return reduce(lambda v1, v2: v1.merge(v2), values)
    else:
        return reduce(lambda v1, v2: v1.accept(v2), values, model)


# @type_inferrer(P.switch, nargs=3)
# async def infer_type_switch(track, cond, tb, fb):
#     """Infer the return type of switch."""
#     await track.check(Bool, cond)
#     v = await cond['value']
#     if v is True:
#         # We only visit the first branch if the condition is provably true
#         return await tb['type']
#     elif v is False:
#         # We only visit the second branch if the condition is provably false
#         return await fb['type']
#     elif v is ANYTHING:
#         # The first branch to finish will return immediately. When the other
#         # branch finishes, its result will be checked against the other.
#         res = await track.assert_same(tb, fb, refs=[tb, fb])
#         if isinstance(res, Inferrer):
#             tinf = await tb['type']
#             finf = await fb['type']
#             return MultiInferrer((tinf, finf), [tb, fb])
#         return res
#     else:
#         raise AssertionError("Invalid condition value for switch")
