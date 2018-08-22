"""Clean up Class types."""

from ..dtype import Tuple, List, Class, Function, Type, ismyiatype
from ..ir import is_apply
from ..prim import ops as P
from ..utils import TypeMap


_retype_map = TypeMap()


@_retype_map.register(Tuple)
def _retype_Tuple(t):
    return Tuple[[_retype(t2) for t2 in t.elements]]


@_retype_map.register(List)
def _retype_List(t):
    return List[_retype(t.element_type)]


@_retype_map.register(Class)
def _retype_Class(t):
    return Tuple[[_retype(t2) for t2 in t.attributes.values()]]


@_retype_map.register(Function)
def _retype_Function(t):
    return Function[[_retype(t2) for t2 in t.arguments], _retype(t.retval)]


@_retype_map.register(Type)
def _retype_Type(t):
    return t


@_retype_map.register(object)
def _retype_object(t):
    return t


def _retype(t):
    if not ismyiatype(t):
        t = type(t)
    return _retype_map[t](t)


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None

        if is_apply(node, P.getattr):
            _, data, item = node.inputs
            dt = data.type
            assert ismyiatype(dt, Class)
            idx = list(dt.attributes.keys()).index(item.value)
            new_node = node.graph.apply(P.tuple_getitem, data, idx)

        elif is_apply(node, P.make_record):
            _, typ, *args = node.inputs
            new_node = node.graph.apply(P.make_tuple, *args)

        if new_node is not None:
            new_node.type = node.type
            manager.replace(node, new_node)

    for node in manager.all_nodes:
        node.type = _retype(node.type)


class EraseClass:
    """Remove the Class type from graphs."""

    def __init__(self, optimizer):
        """Initialize EraseClass."""
        self.optimizer = optimizer

    def __call__(self, root):
        """Remove the Class type from graphs."""
        erase_class(root, self.optimizer.resources.manager)
