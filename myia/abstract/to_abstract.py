"""Conversion to an abstract type."""

import typing
from dataclasses import is_dataclass
from functools import reduce

import numpy as np
from ovld import ovld

from .. import xtype
from ..classes import ADT, Cons, Empty
from ..ir import Graph, MetaGraph
from ..operations import Primitive
from ..utils import (
    EnvInstance,
    HandleInstance,
    MyiaTypeError,
    SymbolicKeyInstance,
    Tag,
    UniverseInstance,
    dataclass_fields,
    is_dataclass_type,
)
from ..utils.misc import RandomStateWrapper
from .amerge import amerge
from .data import (
    ALIASID,
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractADT,
    AbstractArray,
    AbstractClass,
    AbstractDict,
    AbstractExternal,
    AbstractFunction,
    AbstractHandle,
    AbstractRandomState,
    AbstractScalar,
    AbstractTuple,
    AbstractType,
    AbstractUnion,
    AbstractValue,
    DictDesc,
    Function,
    GraphFunction,
    MacroFunction,
    MetaGraphFunction,
    PrimitiveFunction,
    empty,
    listof,
)
from .macro import Macro
from .ref import Context
from .utils import broaden as _broaden, normalize_adt

_number_types = [
    xtype.Int[8],
    xtype.Int[16],
    xtype.Int[32],
    xtype.Int[64],
    xtype.UInt[8],
    xtype.UInt[16],
    xtype.UInt[32],
    xtype.UInt[64],
    xtype.Float[16],
    xtype.Float[32],
    xtype.Float[64],
    xtype.EnvType,
]


##############
# from_value #
##############


def from_value(v, broaden=False, **kwargs):
    """Convert a value to an abstract value.

    Arguments:
        v: The value to convert.
        broaden: If True, concrete values will be made more abstract, so e.g.
            the value 1234 would become ANYTHING.

    """
    a = to_abstract(v, **kwargs)
    if broaden:
        a = _broaden(a)
    return a


###############
# to_abstract #
###############


@ovld
def to_abstract(self, v: AbstractValue, **kwargs):
    """Translate the value to an abstract value.

    Arguments:
        v: The value to convert.
        context: The context in which the value was found, used if the value
            is a Graph.
        node: The node for the Constant we are converting, if there is one,
            so that we can generate a tracking_id.
        loop: The InferenceLoop, or None. If not None, scalars ints or floats
            will be given a Pending type so that it can adapt to the types of
            the variables they interact with.

    """
    return AbstractType(v)


@ovld  # noqa: F811
def to_abstract(self, v: Graph, context=None, node=None, **kwargs):
    ctx = context or Context.empty()
    if v.abstract is None:
        return AbstractFunction(GraphFunction(v, ctx, tracking_id=node))
    else:
        return v.abstract


@ovld  # noqa: F811
def to_abstract(self, v: MetaGraph, node=None, **kwargs):
    return AbstractFunction(
        MetaGraphFunction(v, Context.empty(), tracking_id=node)
    )


@ovld  # noqa: F811
def to_abstract(self, v: Macro, **kwargs):
    return AbstractFunction(MacroFunction(v))


@ovld  # noqa: F811
def to_abstract(self, v: Primitive, node=None, **kwargs):
    return AbstractFunction(PrimitiveFunction(v, tracking_id=node))


@ovld  # noqa: F811
def to_abstract(self, v: HandleInstance, **kwargs):
    return AbstractHandle(v.abstract or self(v.state, **kwargs))


@ovld  # noqa: F811
def to_abstract(
    self,
    v: (
        bool,
        type(None),
        str,
        type(NotImplemented),
        SymbolicKeyInstance,
        EnvInstance,
        UniverseInstance,
    ),
    **kwargs,
):
    typ = xtype.pytype_to_myiatype(type(v))
    return AbstractScalar({VALUE: v, TYPE: typ})


@ovld  # noqa: F811
def to_abstract(
    self, v: (int, float, np.integer, np.floating), loop=None, **kwargs
):
    typ = xtype.pytype_to_myiatype(type(v))
    if loop is not None:
        prio = 1 if issubclass(typ, xtype.Float) else 0
        typ = loop.create_pending_from_list(_number_types, typ, lambda: prio)
    return AbstractScalar({VALUE: v, TYPE: typ})


@ovld  # noqa: F811
def to_abstract(self, v: tuple, **kwargs):
    return AbstractTuple([self(elem, **kwargs) for elem in v])


@ovld  # noqa: F811
def to_abstract(self, v: RandomStateWrapper, **kwargs):
    return AbstractRandomState()


@ovld  # noqa: F811
def to_abstract(self, v: list, **kwargs):
    if v == []:
        return empty
    else:
        elem_types = [self(elem, **kwargs) for elem in v]
        elem_type = reduce(amerge, elem_types)
        return listof(_broaden(elem_type))


@ovld  # noqa: F811
def to_abstract(self, v: dict, **kwargs):
    entries = dict((k, self(val, **kwargs)) for k, val in v.items())
    return AbstractDict(entries)


@ovld  # noqa: F811
def to_abstract(self, v: np.ndarray, alias_map={}, **kwargs):
    tracks = {SHAPE: v.shape, TYPE: xtype.NDArray}
    if id(v) in alias_map:
        tracks[ALIASID] = alias_map[id(v)]
    return AbstractArray(
        AbstractScalar(
            {VALUE: ANYTHING, TYPE: xtype.np_dtype_to_type(str(v.dtype))}
        ),
        tracks,
    )


@ovld  # noqa: F811
def to_abstract(self, v: typing._GenericAlias, **kwargs):
    return AbstractType(type_to_abstract(v))


@ovld  # noqa: F811
def to_abstract(self, v: ADT, **kwargs):
    new_args = {}
    for name, value in dataclass_fields(v).items():
        new_args[name] = self(value, **kwargs)
    draft = AbstractADT(type(v), new_args)
    return normalize_adt(draft)


@ovld  # noqa: F811
def to_abstract(self, v: object, **kwargs):
    if is_dataclass(v):
        assert not isinstance(v, Function)
        new_args = {}
        for name, value in dataclass_fields(v).items():
            new_args[name] = self(value, **kwargs)
        return AbstractClass(type(v), new_args)
    else:
        return AbstractExternal({VALUE: v, TYPE: type(v)})


@ovld  # noqa: F811
def to_abstract(self, v: type, **kwargs):
    try:
        return AbstractType(type_to_abstract(v))
    except TypeError:
        return AbstractExternal({VALUE: v, TYPE: type(v)})


####################
# type_to_abstract #
####################


_default_type_params = {tuple: (), list: (object,), np.ndarray: (object,)}


@ovld
def type_to_abstract(self, t: xtype.TypeMeta):
    """Convert a type to an AbstractValue.

    If the value is already an AbstractValue, returns it directly.
    """
    return self[(t,)](t)


@ovld  # noqa: F811
def type_to_abstract(self, t: AbstractValue):
    return t


@ovld  # noqa: F811
def type_to_abstract(
    self,
    t: (
        xtype.Number,
        xtype.Bool,
        xtype.Nil,
        xtype.SymbolicKeyType,
        xtype.EnvType,
        xtype.UniverseType,
    ),
):
    return AbstractScalar({VALUE: ANYTHING, TYPE: t})


@ovld  # noqa: F811
def type_to_abstract(self, t: np.dtype):
    return self(xtype.np_dtype_to_type(t.name))


@ovld  # noqa: F811
def type_to_abstract(self, t: type):
    if is_dataclass_type(t):
        fields = t.__dataclass_fields__
        attributes = {
            name: ANYTHING
            if isinstance(field.type, (str, type(None)))
            else self(field.type)
            for name, field in fields.items()
        }
        if issubclass(t, ADT):
            return AbstractADT(t, attributes)
        else:
            return AbstractClass(t, attributes)

    elif t is object:
        return ANYTHING

    else:
        return pytype_to_abstract[t, object](
            t, _default_type_params.get(t, None)
        )


@ovld  # noqa: F811
def type_to_abstract(self, t: typing._GenericAlias):
    args = tuple(
        object if isinstance(arg, typing.TypeVar) else arg for arg in t.__args__
    )
    return pytype_to_abstract[t.__origin__, object](t, args)


@ovld  # noqa: F811
def type_to_abstract(self, t: Tag):
    return ANYTHING


@ovld  # noqa: F811
def type_to_abstract(self, t: object):
    raise MyiaTypeError(f"{t} is not a recognized type")


_numpy_types = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)


@ovld
def pytype_to_abstract(main: tuple, args):
    """Convert a Python type to an AbstractValue."""
    if args == () or args is None:
        targs = ANYTHING
    elif args == ((),):
        targs = []
    else:
        targs = [type_to_abstract(a) for a in args]
    return AbstractTuple(targs)


@ovld  # noqa: F811
def pytype_to_abstract(main: list, args):
    (arg,) = args
    argt = type_to_abstract(arg)
    if argt is ANYTHING:
        rval = AbstractUnion([type_to_abstract(Empty), type_to_abstract(Cons)])
    else:
        rval = listof(argt)
    return rval


@ovld  # noqa: F811
def pytype_to_abstract(main: np.ndarray, args):
    (arg,) = args
    arg = type_to_abstract(arg)
    shp = ANYTHING
    return AbstractArray(arg, {SHAPE: shp, TYPE: xtype.NDArray})


@ovld  # noqa: F811
def pytype_to_abstract(main: _numpy_types, args):
    return type_to_abstract(xtype.np_dtype_to_type(main.__name__))


@ovld  # noqa: F811
def pytype_to_abstract(main: int, args):
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Int[64]})


@ovld  # noqa: F811
def pytype_to_abstract(main: float, args):
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Float[64]})


@ovld  # noqa: F811
def pytype_to_abstract(main: bool, args):
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Bool})


@ovld  # noqa: F811
def pytype_to_abstract(main: AbstractArray, args):
    return AbstractArray(ANYTHING, values={SHAPE: ANYTHING, TYPE: ANYTHING})


@ovld  # noqa: F811
def pytype_to_abstract(main: AbstractRandomState, args):
    return AbstractRandomState()


@ovld  # noqa: F811
def pytype_to_abstract(main: RandomStateWrapper, args):
    return AbstractRandomState()


@ovld  # noqa: F811
def pytype_to_abstract(main: HandleInstance, args):
    return AbstractHandle(None)


@ovld  # noqa: F811
def pytype_to_abstract(main: dict, args):
    if args is None:
        # Just provide an empty dict as entries.
        entries = {}
    else:
        key_type, value_type = args
        if key_type is not str:
            raise MyiaTypeError(f"Expected dict type str, got {key_type}")
        # We cannot get keys from a type description, so we will instead use
        # a placeholder object keeping abstract value.
        entries = DictDesc(type_to_abstract(value_type))
    return AbstractDict(entries)


__all__ = [
    "from_value",
    "pytype_to_abstract",
    "to_abstract",
    "type_to_abstract",
]
