"""Clean up Class types."""

import weakref
from itertools import count

from ..abstract import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractClassBase,
    AbstractDict,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    AbstractValue,
    abstract_clone,
    split_type,
    type_to_abstract,
)
from ..dtype import Int, String
from ..ir import Constant
from ..prim import ops as P
from ..utils import is_dataclass_type, overload

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


@abstract_clone.variant
def _reabs(self, a: AbstractClassBase):
    return (yield AbstractTuple)(self(x) for x in a.attributes.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractScalar):
    if a.values[TYPE] == String:
        v = a.values[VALUE]
        if v is not ANYTHING:
            v = str_to_tag(v)
        a = AbstractScalar({**a.values, VALUE: v, TYPE: Int[64]})
    return a


@overload  # noqa: F811
def _reabs(self, a: AbstractDict):
    return (yield AbstractTuple)(self(x) for x in a.entries.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractUnion):
    return (yield AbstractTaggedUnion)(
        [type_to_tag(opt), self(opt)] for opt in a.options
    )


@overload  # noqa: F811
def _reabs(self, a: AbstractKeywordArgument):
    return self(a.argument)


def simplify_types(root, manager):
    """Simplify the set of types that can be found in the graph.

    * Replace AbstractClass by AbstractTuple:
      * Class[x: t, ...] => Tuple[t, ...]
      * record_getitem(data, attr) => tuple_getitem(data, idx)
      * record_setitem(data, attr, value) => tuple_setitem(data, idx, value)
      * make_record(cls, *args) => make_tuple(*args)

    * Replace AbstractDict by AbstractTuple:
      * Dict[x: t, ...] => Tuple[t, ...]
      * dict_getitem(data, item) => tuple_getitem(data, idx)
      * dict_setitem(data, item, value) => tuple_setitem(data, idx, value)
      * make_dict(cls, *args) => make_tuple(*args)

    * Replace AbstractUnion by AbstractTaggedUnion:
      * Union[a, b, c, ...] => TaggedUnion[1 => a, 2 => b, 3 => c, ...]
      * hastype(x, type) => hastag(x, tag)
      *                  => bool_or(hastag(x, tag1), hastag(x, tag2), ...)
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

        elif (node.is_constant(AbstractArray) and
                type(node.value) is not AbstractArray):
            new_node = Constant(
                AbstractArray(node.value.element, node.value.values))
            keep_abstract = False

        elif node.is_apply(P.make_kwarg):
            new_node = node.inputs[2]

        elif node.is_apply(P.extract_kwarg):
            new_node = node.inputs[2]

        elif node.is_constant(AbstractClassBase):
            # This is a constant that contains a type, used e.g. with hastype.
            new_node = Constant(_reabs(node.value))
            keep_abstract = False

        elif node.is_constant(str):
            new_node = Constant(str_to_tag(node.value))

        elif node.is_constant(AbstractDict):
            # This is a constant that contains a type, used e.g. with hastype.
            new_node = Constant(_reabs(node.value))
            keep_abstract = False

        elif node.is_constant() and is_dataclass_type(node.value):
            raise NotImplementedError()
            # new_node = Constant(P.make_tuple)

        if new_node is not None:
            if keep_abstract:
                new_node.abstract = node.abstract
            manager.replace(node, new_node)

    for graph in manager.graphs:
        graph._sig = None
        graph._user_graph = None

    for node in manager.all_nodes:
        node.abstract = _reabs(node.abstract)
