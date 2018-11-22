
import inspect
from operator import getitem

from .base import (
    ABSENT,
    AbstractBase,
    AbstractValue,
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
from ..infer import ANYTHING, InferenceVar, Context, MyiaTypeError
from ..prim import ops as P
from ..utils import Namespace


abstract_inferrer_constructors = {}


def reg(*prims):
    def deco(cls):
        for prim in prims:
            abstract_inferrer_constructors[prim] = cls
        return cls
    return deco


@reg(P.identity, P.return_)
class IdentityXInferrer(XInferrer):
    async def infer(self, track, x):
        return x


class PrimitiveXInferrer(XInferrer):
    async def infer(self, track, *args):
        rval = await self.infer_structure(track, *args)
        for t in track.subtracks:
            method = getattr(self, f'infer_{t}')
            rval.values[t] = await method(track, *args)
        return rval

    async def infer_structure(self, track, *args):
        raise NotImplementedError()

    async def infer_value(self, track, *args):
        raise NotImplementedError()

    async def infer_type(self, track, *args):
        raise NotImplementedError()

    async def infer_shape(self, track, *args):
        raise NotImplementedError()


class StructuralXInferrer(PrimitiveXInferrer):

    async def infer_value(self, track, *args):
        return ANYTHING

    async def infer_type(self, track, *args):
        return ABSENT

    async def infer_shape(self, track, *args):
        return ABSENT


@reg(P.make_tuple)
class MakeTupleXInferrer(StructuralXInferrer):
    async def infer_structure(self, track, *args):
        return AbstractTuple(args)


@reg(P.tuple_getitem)
class TupleGetitemXInferrer(XInferrer):
    async def infer(self, track, arg, idx):
        i = idx.values['value']
        return arg.elements[i]


class ArithScalarXInferrer(PrimitiveXInferrer):

    async def infer_structure(self, track, *args):
        assert all(isinstance(arg, AbstractValue) for arg in args)
        return AbstractValue({})

    async def infer_value(self, track, *args):
        values = [arg.values['value'] for arg in args]
        if any(v is ANYTHING for v in values):
            return ANYTHING
        return self.impl(*values)

    async def infer_type(self, track, *args):
        nargs = len(inspect.getfullargspec(self.impl).args) - 1
        if len(args) != nargs:
            raise MyiaTypeError(f'Wrong number of arguments')
        ts = [arg.values['type'] for arg in args]
        return await track.will_check(dtype.Number, *ts)

    async def infer_shape(self, track, *args):
        return dshape.NOSHAPE


@reg(P.scalar_usub)
class USubXInferrer(ArithScalarXInferrer):
    def impl(self, x):
        return -x


@reg(P.scalar_add)
class AddXInferrer(ArithScalarXInferrer):
    def impl(self, x, y):
        return x + y


@reg(P.scalar_mul)
class MulXInferrer(ArithScalarXInferrer):
    def impl(self, x, y):
        return x * y


# @abstract_inferrer(P.switch, nargs=3)
# async def infer_abstract_switch(track, cond, tb, fb):
#     pass
#     # """Infer the return type of if."""
#     # await track.check(Bool, cond)
#     # tb_inf = await tb['type']
#     # fb_inf = await fb['type']
#     # v = await cond['value']
#     # if v is True:
#     #     # We only visit the first branch if the condition is provably true
#     #     return await tb_inf()
#     # elif v is False:
#     #     # We only visit the second branch if the condition is provably false
#     #     return await fb_inf()
#     # elif v is ANYTHING:
#     #     # The first branch to finish will return immediately. When the other
#     #     # branch finishes, its result will be checked against the other.
#     #     return await track.assert_same(tb_inf(), fb_inf(), refs=[tb, fb])
#     # else:
#     #     raise AssertionError("Invalid condition value for if")


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


@reg(P.resolve)
class ResolveXInferrer(XInferrer):
    async def infer(self, track, data, item):
        """Infer the return type of resolve."""
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
