"""Clean up Class types."""

from ..dtype import Int
from ..ir import Constant
from ..prim import ops as P
from ..abstract import abstract_clone, AbstractClass, AbstractTuple, \
    AbstractScalar, VALUE, TYPE, AbstractADT
from ..utils import overload


@abstract_clone.variant
def _reabs(self, a: AbstractClass):
    return (yield AbstractTuple)(self(x) for x in a.attributes.values())


@overload  # noqa: F811
def _reabs(self, a: AbstractADT):
    return (yield AbstractTuple)(self(x) for x in a.attributes.values())


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    from ..abstract import AbstractClass
    from ..utils import is_dataclass_type

    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None
        keep_abstract = True

        if node.is_apply(P.getattr):
            _, data, item = node.inputs
            dt = data.abstract
            assert isinstance(dt, (AbstractClass, AbstractADT))
            idx = list(dt.attributes.keys()).index(item.value)
            idx_c = Constant(idx)
            idx_c.abstract = AbstractScalar({
                VALUE: idx,
                TYPE: Int[64],
            })
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

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

        elif node.is_constant(AbstractADT):
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
