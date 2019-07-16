"""Clean up Class types."""

import weakref
from itertools import count
from functools import reduce

from ..dtype import Int
from ..ir import Constant
from ..prim import ops as P
from ..abstract import abstract_clone, AbstractClassBase, AbstractTuple, \
    AbstractUnion, AbstractTaggedUnion, AbstractScalar, VALUE, TYPE, \
    AbstractValue, AbstractDict, type_to_abstract
from ..utils import is_dataclass_type, overload


_idx = count()
_tagmap = weakref.WeakKeyDictionary()


def type_to_tag(t):
    """Return the numeric tag associated to the given type."""
    if t not in _tagmap:
        _tagmap[t] = next(_idx)
    return _tagmap[t]


@abstract_clone.variant
def _reabs(self, a: AbstractClassBase):
    return (yield AbstractTuple)(self(x) for x in a.attributes.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractDict):
    return (yield AbstractTuple)(self(x) for x in a.entries.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractUnion):
    return (yield AbstractTaggedUnion)(
        [type_to_tag(opt), self(opt)] for opt in a.options
    )


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None
        keep_abstract = True

        if node.is_apply(P.getattr):
            _, data, item = node.inputs
            dt = data.abstract
            assert isinstance(dt, AbstractClassBase)
            idx = list(dt.attributes.keys()).index(item.value)
            idx_c = Constant(idx)
            idx_c.abstract = AbstractScalar({
                VALUE: idx,
                TYPE: Int[64],
            })
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

        elif node.is_apply(P.dict_getitem):
            _, data, item = node.inputs
            dt = data.abstract
            assert isinstance(dt, AbstractDict)
            idx = list(dt.entries.keys()).index(item.value)
            idx_c = Constant(idx)
            idx_c.abstract = AbstractScalar({
                VALUE: idx,
                TYPE: Int[64],
            })
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

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
            from ..abstract import split_type
            _, x, typ = node.inputs
            real_typ = type_to_abstract(typ.value)
            matches, _ = split_type(x.abstract, real_typ)
            if isinstance(matches, AbstractUnion):
                tags = [type_to_tag(poss) for poss in matches.options]
            else:
                tags = [type_to_tag(matches)]
            hastags = [node.graph.apply(P.hastag, x, tag) for tag in tags]
            new_node = reduce(
                (lambda x, y: node.graph.apply(P.bool_or, x, y)),
                hastags
            )

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
                tag = type_to_tag(node.abstract)
                new_node = node.graph.apply(P.tagged, x, tag)

        elif node.is_constant(AbstractClassBase):
            # This is a constant that contains a type, used e.g. with hastype.
            new_node = Constant(_reabs(node.value))
            keep_abstract = False

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

    for node in manager.all_nodes:
        node.abstract = _reabs(node.abstract)
