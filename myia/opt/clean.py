"""Clean up Class types."""

from ..dtype import Int, Tuple, List, Class, Function, Type, ismyiatype, \
    TypeMeta
from ..ir import Constant
from ..prim import ops as P
from ..prim.shape_inferrers import TupleShape, ListShape, ClassShape
from ..utils import overload, UNKNOWN


@overload
def _retype(t: Tuple):
    return Tuple[[_retype(t2) for t2 in t.elements]]


@overload  # noqa: F811
def _retype(t: List):
    return List[_retype(t.element_type)]


@overload  # noqa: F811
def _retype(t: Class):
    return Tuple[[_retype(t2) for t2 in t.attributes.values()]]


@overload  # noqa: F811
def _retype(t: Function):
    return Function[[_retype(t2) for t2 in t.arguments], _retype(t.retval)]


@overload  # noqa: F811
def _retype(t: Type):
    return t


@overload  # noqa: F811
def _retype(t: TypeMeta):
    return _retype[t](t)


@overload  # noqa: F811
def _retype(t: object):
    # This will be a validation error later on, and the validator
    # will report it better than we could here.
    return t  # pragma: no cover


@overload
def _reshape(s: TupleShape):
    return TupleShape(_reshape(s2) for s2 in s.shape)


@overload  # noqa: F811
def _reshape(s: ListShape):
    return ListShape(_reshape(s.shape))


@overload  # noqa: F811
def _reshape(s: ClassShape):
    return TupleShape(_reshape(s2) for s2 in s.shape.values())


@overload  # noqa: F811
def _reshape(s: object):
    return s


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None

        if node.is_apply(P.getattr):
            _, data, item = node.inputs
            dt = data.type
            assert ismyiatype(dt, Class)
            idx = list(dt.attributes.keys()).index(item.value)
            idx = Constant(idx)
            idx.type = Int[64]
            new_node = node.graph.apply(P.tuple_getitem, data, idx)

        elif node.is_apply(P.make_record):
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

        if new_node is not None:
            manager.replace(node, new_node)

    for node in manager.all_nodes:
        node.expect_inferred.clear()
        if node.type is not UNKNOWN:
            node.type = _retype(node.type)
        if node.shape is not UNKNOWN:
            node.shape = _reshape(node.shape)
