"""Tuple operations."""

import operator
from functools import reduce

from .. import operations
from ..lib import (
    AbstractTuple,
    Graph,
    MetaGraph,
    MyiaTypeError,
    Slice,
    build_value,
    check_nargs,
    core,
)
from ..operations import hastype, primitives as P, tuple_getitem
from .utils import to_opdef


@to_opdef
@core
def tuple_next(xs):
    """Next tuple."""
    return xs[0], xs[1:]


@to_opdef
@core
def tuple_hasnext(xs):
    """Whether the tuple is empty or not."""
    return len(xs) > 0


class TupleReorganizer(MetaGraph):
    """Parametrizable MetaGraph to combine or extract tuples."""

    def __init__(self, name, gen):
        """Initialize a TupleReorganizer."""
        super().__init__(name)
        self.__name__ = name
        self.gen = gen

    def map_tuples(self, g, params, tups):
        """Map each element of each tuple to a getitem on the parameter."""
        rval = []
        for tup, param in zip(tups, params):
            if not isinstance(tup, AbstractTuple):
                raise MyiaTypeError(f'Expected AbstractTuple, not {tup}')
            rval.append([
                g.apply(P.tuple_getitem, param, i)
                for i, elem in enumerate(tup.elements)
            ])
        return rval

    def generate_graph(self, args):
        """Generate the graph."""
        g = Graph()
        g.debug.name = self.gen.__name__
        for arg in args:
            g.add_parameter()
        g.output = self.gen(self, g, args)
        return g


def tuple_reorganizer(fn):
    """Shortcut to create a new TupleReorganizer from a function."""
    return TupleReorganizer(name=fn.__name__, gen=fn)


@to_opdef
@tuple_reorganizer
def tuple_concat(self, g, args):
    """Metagraph for tuple concatenation."""
    tups = self.map_tuples(g, g.parameters, args)
    return g.apply(P.make_tuple, *reduce(operator.add, tups))


@to_opdef
@tuple_reorganizer
def tuple_getslice(self, g, args):
    """Metagraph for getting a slice from a tuple."""
    tuparg, start, stop, step = check_nargs('tail', 4, args)
    try:
        start = build_value(start)
        stop = build_value(stop)
        step = build_value(step)
    except ValueError:
        raise MyiaTypeError('Slice start, stop and step must be static')
    tup, = self.map_tuples(g, g.parameters[:1], [tuparg])
    return g.apply(P.make_tuple, *tup[start:stop:step])


@to_opdef
@core
def tuple_get(t, item):
    """Implementation of `tuple.__getitem__`."""
    if hastype(item, Slice):
        return operations.tuple_getslice(t, item.start, item.stop, item.step)
    else:
        return tuple_getitem(t, item)
