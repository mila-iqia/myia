"""Clean up Class types."""

import weakref
from itertools import count

from . import xtype
from .abstract import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractClassBase,
    AbstractDict,
    AbstractHandle,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractType,
    AbstractUnion,
    AbstractValue,
    abstract_clone,
    empty,
    from_value,
    split_type,
    type_to_abstract,
    typecheck,
)
from .classes import Cons, Empty
from .compile import BackendValue
from .ir import Constant
from .operations import primitives as P
from .utils import HandleInstance, MyiaInputTypeError, TaggedValue, overload
from .xtype import Int, NDArray, String

####################
# Tags and symbols #
####################

_idx = count()
_tagmap = weakref.WeakKeyDictionary()


def type_to_tag(t):
    """Return the numeric tag associated to the given type."""
    if t not in _tagmap:
        _tagmap[t] = next(_idx)
    return _tagmap[t]


_tagmap_str = {}
_strmap_tag = {}


def str_to_tag(t):
    """Return the numeric tag associated to the given type."""
    if t not in _tagmap_str:
        s = len(_tagmap_str)
        _tagmap_str[t] = s
        _strmap_tag[s] = t
    return _tagmap_str[t]


#########
# Reabs #
#########


@abstract_clone.variant
def _reabs(self, a: AbstractClassBase):
    return (yield AbstractTuple)(self(x) for x in a.attributes.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractScalar):
    new_values = self(a.values)
    if a.xtype() == String:
        v = a.xvalue()
        if v is not ANYTHING:
            v = str_to_tag(v)
        return AbstractScalar({**new_values, VALUE: v, TYPE: Int[64]})
    else:
        return AbstractScalar(new_values)


@overload  # noqa: F811
def _reabs(self, a: AbstractDict):
    return (yield AbstractTuple)(self(x) for x in a.entries.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractArray):
    return (yield AbstractArray)(self(a.element),
                                 {**self(a.values), TYPE: NDArray})


@overload  # noqa: F811
def _reabs(self, a: AbstractUnion):
    return (yield AbstractTaggedUnion)(
        [type_to_tag(opt), self(opt)] for opt in a.options
    )


@overload  # noqa: F811
def _reabs(self, a: AbstractKeywordArgument):
    return self(a.argument)


@overload  # noqa: F811
def _reabs(self, a: AbstractType):
    return (yield AbstractType)(self(a.values[VALUE]))


##################
# Simplify types #
##################


def simplify_types(root, manager):
    r"""Simplify the set of types that can be found in the graph.

    * Replace AbstractClass by AbstractTuple:

      * Class[x: t, ...] => Tuple[t, ...]
      * record_getitem(data, attr) => tuple_getitem(data, idx)
      * record_setitem(data, attr, value) => tuple_setitem(data, idx, value)
      * make_record(cls, \*args) => make_tuple(\*args)

    * Replace AbstractDict by AbstractTuple:

      * Dict[x: t, ...] => Tuple[t, ...]
      * dict_getitem(data, item) => tuple_getitem(data, idx)
      * dict_setitem(data, item, value) => tuple_setitem(data, idx, value)
      * make_dict(cls, \*args) => make_tuple(\*args)

    * Replace AbstractUnion by AbstractTaggedUnion:

      * Union[a, b, c, ...] => TaggedUnion[1 => a, 2 => b, 3 => c, ...]
      * hastype(x, type) => hastag(x, tag)
                         => bool_or(hastag(x, tag1), hastag(x, tag2), ...)
      * unsafe_static_cast(x, type) => casttag(x, tag)
    """
    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None
        keep_abstract = True

        def _mkct(idx):
            idx_c = Constant(idx)
            idx_c.abstract = AbstractScalar({
                VALUE: idx,
                TYPE: Int[64],
            })
            return idx_c

        def _record_makeindex(dt, attr):
            assert isinstance(dt, AbstractClassBase)
            idx = list(dt.attributes.keys()).index(attr)
            return _mkct(idx)

        def _dict_makeindex(dt, attr):
            assert isinstance(dt, AbstractDict)
            idx = list(dt.entries.keys()).index(attr)
            return _mkct(idx)

        if node.is_apply(P.record_getitem):
            _, data, item = node.inputs
            idx_c = _record_makeindex(data.abstract, item.value)
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

        elif node.is_apply(P.dict_getitem):
            _, data, item = node.inputs
            idx_c = _dict_makeindex(data.abstract, item.value)
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

        elif node.is_apply(P.record_setitem):
            _, data, item, value = node.inputs
            idx_c = _record_makeindex(data.abstract, item.value)
            new_node = node.graph.apply(P.tuple_setitem, data, idx_c, value)

        elif node.is_apply(P.dict_setitem):
            _, data, item, value = node.inputs
            idx_c = _dict_makeindex(data.abstract, item.value)
            new_node = node.graph.apply(P.tuple_setitem, data, idx_c, value)

        elif node.is_apply(P.make_record):
            mkr, typ, *args = node.inputs
            new_node = node.graph.apply(P.make_tuple, *args)

        elif node.is_apply(P.make_dict):
            mkr, typ, *args = node.inputs
            new_node = node.graph.apply(P.make_tuple, *args)

        elif node.is_apply(P.partial):
            orig_ptl, oper, *args = node.inputs
            if oper.is_constant() and oper.value is P.make_record:
                if len(args) == 1:
                    new_node = Constant(P.make_tuple)
                elif len(args) > 1:
                    new_node = node.graph.apply(
                        P.partial, P.make_tuple, *args[1:]
                    )

        elif node.is_apply(P.hastype):
            # hastype(x, type) -> hastag(x, tag)
            _, x, typ = node.inputs
            real_typ = type_to_abstract(typ.value)
            matches, _ = split_type(x.abstract, real_typ)
            assert not isinstance(matches, AbstractUnion)
            new_node = node.graph.apply(P.hastag, x, type_to_tag(matches))

        elif node.is_apply(P.unsafe_static_cast):
            # unsafe_static_cast(x, type) -> casttag(x, tag)
            # unsafe_static_cast(x, union_type) -> x, if x bigger union type
            _, x, typ = node.inputs
            assert isinstance(typ.value, AbstractValue)
            if isinstance(typ.value, AbstractUnion):
                new_node = x
                keep_abstract = False
            else:
                tag = type_to_tag(typ.value)
                new_node = node.graph.apply(P.casttag, x, tag)

        elif node.is_apply(P.tagged):
            # tagged(x) -> tagged(x, tag)
            # tagged(x, tag) -> unchanged
            if len(node.inputs) == 2:
                _, x = node.inputs
                tag = type_to_tag(x.abstract)
                new_node = node.graph.apply(P.tagged, x, tag)

        elif node.is_apply(P.string_eq):
            new_node = node.graph.apply(P.scalar_eq,
                                        node.inputs[1], node.inputs[2])

        elif node.is_apply(P.make_kwarg):
            new_node = node.inputs[2]

        elif node.is_apply(P.extract_kwarg):
            new_node = node.inputs[2]

        elif node.is_constant((str, AbstractValue)):
            new_node = Constant(to_canonical(node.value, node.abstract))
            keep_abstract = False

        if new_node is not None:
            if keep_abstract:
                new_node.abstract = node.abstract
            manager.replace(node, new_node)

    for graph in manager.graphs:
        graph._sig = None
        graph._user_graph = None

    for node in manager.all_nodes:
        node.abstract = _reabs(node.abstract)


########################
# Convert to canonical #
########################


@overload.wrapper(bootstrap=True)
def to_canonical(fn, self, arg, orig_t):
    """Check and convert an argument to the canonical representation.

    Arguments:
        arg: The argument to convert.
        orig_t: The type of the argument as returned by to_abstract.

    Returns:
        A version of the argument where classes/dicts become tuples
        and unions are properly tagged.

    """
    if isinstance(arg, BackendValue):
        if not typecheck(orig_t, arg.orig_t):
            raise MyiaInputTypeError("Bad type for backend value.")
        return arg
    if fn is None:
        raise AssertionError(f'to_canonical not defined for {orig_t}')
    return fn(self, arg, orig_t)


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractTuple):
    if not isinstance(arg, tuple):
        raise MyiaInputTypeError('Expected tuple')
    oe = orig_t.elements
    if len(arg) != len(oe):
        raise MyiaInputTypeError(f'Expected {len(oe)} elements')
    return tuple(self(x, o) for x, o in zip(arg, oe))


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractDict):
    if not isinstance(arg, dict):
        raise MyiaInputTypeError('Expected dict')
    types = orig_t.entries
    if len(arg) != len(types):
        raise MyiaInputTypeError(
            "Dictionary input doesn't have the expected size"
        )
    if set(arg.keys()) != set(types.keys()):
        raise MyiaInputTypeError("Mismatched keys for input dictionary.")
    return tuple(self(arg[k], o) for k, o in orig_t.entries.items())


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractClassBase):
    if orig_t.tag is Empty:
        if arg != []:
            raise MyiaInputTypeError(f'Expected empty list')
        return ()
    elif orig_t.tag is Cons:
        if arg == []:
            raise MyiaInputTypeError(f'Expected non-empty list')
        if not isinstance(arg, list):
            raise MyiaInputTypeError(f'Expected list')
        ot = orig_t.attributes['head']
        li = list(self(x, ot) for x in arg)
        rval = TaggedValue(type_to_tag(empty), ())
        for elem in reversed(li):
            rval = TaggedValue(type_to_tag(orig_t), (elem, rval))
        return rval.value
    else:
        if not isinstance(arg, orig_t.tag):
            raise MyiaInputTypeError(f'Expected {orig_t.tag.__qualname__}')
        arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
        oe = list(orig_t.attributes.values())
        res = tuple(self(x, o) for x, o in zip(arg, oe))
        return res


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractArray):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.xtype()
    assert issubclass(et, (xtype.Number, xtype.Bool))
    arg = orig_t.xtype().to_numpy(arg)
    arg_dtype = xtype.np_dtype_to_type(str(arg.dtype))
    if arg_dtype != et:
        raise MyiaInputTypeError(
            f"Expected array of type {et}, but got {arg_dtype}."
        )
    shp = orig_t.xshape()
    if (shp is not ANYTHING and arg.shape != shp):
        raise MyiaInputTypeError(
            f"Expected array with shape {shp}, but got {arg.shape}."
        )
    return arg


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractUnion):
    for opt in orig_t.options:
        try:
            value = self(arg, opt)
            tag = type_to_tag(opt)
        except TypeError:
            continue
        return TaggedValue(tag, value)
    else:
        opts = ", ".join(map(str, orig_t.options))
        raise MyiaInputTypeError(f'Expected one of {opts}, not {arg}')


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractHandle):
    if not isinstance(arg, HandleInstance):
        raise MyiaInputTypeError(f'Expected handle')
    arg.state = self(arg.state, orig_t.element)
    return arg


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractScalar):
    if not typecheck(orig_t, from_value(arg)):
        raise MyiaInputTypeError(
            f'Scalar has wrong type: expected {orig_t}, got {arg}'
        )
    if issubclass(orig_t.xtype(), xtype.String):
        arg = str_to_tag(arg)
    return arg


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractType):
    return _reabs(arg)


@overload  # noqa: F811
def to_canonical(self, arg, orig_t: AbstractKeywordArgument):
    return arg


#####################
# Convert to output #
#####################


@overload(bootstrap=True)
def from_canonical(self, res, orig_t: AbstractClassBase):
    if orig_t.tag in (Empty, Cons):
        rval = []
        while res:
            value = self(res[0], orig_t.attributes['head'])
            rval.append(value)
            res = res[1].value
        return rval
    tup = tuple(self(x, o) for x, o in zip(res, orig_t.attributes.values()))
    return orig_t.constructor(*tup)


@overload  # noqa: F811
def from_canonical(self, res, orig_t: AbstractDict):
    tup = tuple(self(x, o) for x, o in zip(res, orig_t.entries.values()))
    return dict(zip(orig_t.entries.keys(), tup))


@overload  # noqa: F811
def from_canonical(self, res, orig_t: AbstractTuple):
    return tuple(self(x, o) for x, o in zip(res, orig_t.elements))


@overload  # noqa: F811
def from_canonical(self, arg, orig_t: AbstractScalar):
    if orig_t.xtype() == xtype.String:
        arg = _strmap_tag[arg]
    return arg


@overload  # noqa: F811
def from_canonical(self, arg, orig_t: AbstractArray):
    return orig_t.xtype().from_numpy(arg)


@overload  # noqa: F811
def from_canonical(self, arg, orig_t: AbstractHandle):
    # The state is updated by the pipeline through universe.commit()
    return arg


@overload  # noqa: F811
def from_canonical(self, arg, orig_t: AbstractUnion):
    for typ in orig_t.options:
        tag = type_to_tag(typ)
        if tag == arg.tag:
            return self(arg.value, typ)
    else:
        raise AssertionError(f'Badly formed TaggedValue')


__consolidate__ = True
__all__ = [
    'from_canonical',
    'simplify_types',
    'str_to_tag',
    'to_canonical',
    'type_to_tag',
]
