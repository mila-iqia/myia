"""Clean up Class types."""

from ..dtype import Int, Tuple, List, Class, Function, Type, ismyiatype, \
    TypeMeta
from ..ir import Constant
from ..prim import ops as P
from ..utils import overload


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


def _mkr_to_mkt(mkr):
    argtypes = mkr.type.arguments[1:]
    mkt = Constant(P.make_tuple)
    mkt.type = Function[argtypes, Tuple[argtypes]]
    return mkt


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
            gi = Constant(P.tuple_getitem)
            gi.type = Function[[dt, Int[64]], node.type]
            new_node = node.graph.apply(gi, data, idx)

        elif node.is_apply(P.make_record):
            mkr, typ, *args = node.inputs
            mkt = _mkr_to_mkt(mkr)
            new_node = node.graph.apply(mkt, *args)

        elif node.is_apply(P.partial):
            orig_ptl, oper, *args = node.inputs
            if oper.is_constant() and oper.value is P.make_record:
                argtypes = [arg.type for arg in args[1:]]
                mkt = _mkr_to_mkt(oper)
                if len(args) == 1:
                    new_node = mkt
                elif len(args) > 1:
                    ptl = Constant(P.partial)
                    ret = orig_ptl.type.retval
                    ptl.type = Function[[mkt.type, *argtypes], ret]
                    new_node = node.graph.apply(ptl, mkt, *args[1:])

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
