"""Generate mapping graphs over classes, tuples, arrays, etc."""

import numpy as np
from dataclasses import is_dataclass

from . import operations, composite as C, abstract
from .abstract import MyiaTypeError, broaden
from .ir import MetaGraph, Graph
from .dtype import tag_to_dataclass, pytype_to_myiatype
from .utils import Overload
from .prim import ops as P
from .prim.py_implementations import array_map


nonleaf_defaults = (
    abstract.AbstractArray,
    abstract.AbstractList,
    abstract.AbstractTuple,
    abstract.AbstractClass,
)


class HyperMap(MetaGraph):
    """Map over tuples, classes, lists and arrays.

    Arguments:
        fn_leaf: The function to apply on leaves. If it is None,
            the function is passed as the first argument.
        fn_rec: The function to apply recursively. If it is None,
            the HyperMap will apply itself. Using another function
            will implement a "shallow" HyperMap.
        broadcast: Whether to automatically broadcast leaf arguments when
            there are nonleaf arguments (default: True).
        nonleaf: List of Types to generate a recursive map over.
            Any type not in this list will generate a call to
            fn_leaf.

    """

    def __init__(self, *,
                 fn_leaf=None,
                 fn_rec=None,
                 broadcast=True,
                 nonleaf=nonleaf_defaults,
                 name=None):
        """Initialize a HyperMap."""
        if name is None:
            if fn_leaf is None:
                name = 'hyper_map'
            else:
                name = f'hyper_map[{fn_leaf}]'  # pragma: no cover
        super().__init__(name)
        self.fn_leaf = fn_leaf
        self.fn_rec = fn_rec or self
        self.broadcast = broadcast
        self.nonleaf = nonleaf

    def normalize_args(self, args):
        """Return broadened arguments."""
        return tuple(broaden(a, None) for a in args)

    def _is_nonleaf(self, arg):
        return isinstance(arg, self.nonleaf)

    def _is_leaf(self, arg):
        return not self._is_nonleaf(arg)

    _make = Overload(name='hypermap._make')

    @_make.register
    def _make(self, t: abstract.AbstractArray, g, fnarg, argmap):
        if fnarg is None:
            fnarg = self.fn_leaf

        if len(argmap) > 1 and self.broadcast:
            args = [g.apply(operations.to_array, arg) if isleaf else arg
                    for arg, (a, isleaf) in argmap.items()]
            first, *rest = args
            shp = g.apply(P.shape, first)
            for other in rest:
                shp2 = g.apply(P.shape, other)
                shp = g.apply(P.broadcast_shape, shp2, shp)
            args = [g.apply(P.distribute, arg, shp) for arg in args]

        else:
            args = list(argmap.keys())

        return g.apply(P.array_map, fnarg, *args)

    @_make.register
    def _make(self, a: abstract.AbstractTuple, g, fnarg, argmap):
        for a2, isleaf in argmap.values():
            if not isleaf and len(a2.elements) != len(a.elements):
                raise MyiaTypeError(f'Tuple length mismatch: {a} != {a2}')

        elems = []
        for i in range(len(a.elements)):
            args = [arg if isleaf else g.apply(P.tuple_getitem, arg, i)
                    for arg, (_, isleaf) in argmap.items()]
            if fnarg is None:
                val = g.apply(self.fn_rec, *args)
            else:
                val = g.apply(self.fn_rec, fnarg, *args)
            elems.append(val)
        return g.apply(P.make_tuple, *elems)

    @_make.register
    def _make(self, a: abstract.AbstractList, g, fnarg, argmap):
        args = list(argmap.keys())
        mask = [not isleaf for _, isleaf in argmap.values()]
        if fnarg is None:
            lm = C.ListMap(self.fn_rec, loop_mask=mask)
            return g.apply(lm, *args)
        else:
            fn_rec = g.apply(P.partial, self.fn_rec, fnarg)
            lm = C.ListMap(loop_mask=mask)
            return g.apply(lm, fn_rec, *args)

    @_make.register
    def _make(self, a: abstract.AbstractClass, g, fnarg, argmap):
        for a2, isleaf in argmap.values():
            if not isleaf:
                if (a2.tag != a.tag
                        or a2.attributes.keys() != a.attributes.keys()):
                    raise MyiaTypeError(f'Class mismatch: {a} != {a2}')

        vals = []
        for k in a.attributes.keys():
            args = [arg if isleaf else g.apply(P.getattr, arg, k)
                    for arg, (_, isleaf) in argmap.items()]
            if fnarg is None:
                val = g.apply(self.fn_rec, *args)
            else:
                val = g.apply(self.fn_rec, fnarg, *args)
            vals.append(val)
        t = pytype_to_myiatype(tag_to_dataclass[a.tag])
        return g.apply(P.make_record, t, *vals)

    def _generate_helper(self, g, fnarg, argmap):
        nonleafs = [a for a, isleaf in argmap.values() if not isleaf]
        if not nonleafs:
            if fnarg is None:
                fnarg = self.fn_leaf
            return g.apply(fnarg, *argmap.keys())
        else:
            types = set(type(a) for a in nonleafs)
            if len(types) != 1:
                raise MyiaTypeError(
                    f'Incompatible types for hyper_map: {types}'
                )
            return self._make(nonleafs[0], g, fnarg, argmap)

    def generate_graph(self, all_args):
        """Create a graph for mapping over the given args."""
        g = Graph()
        g.debug.name = 'hyper_map'
        argmap = {}
        if self.fn_leaf is None:
            fn_t, *args = all_args
            fnarg = g.add_parameter()
        else:
            args = all_args
            fnarg = None
        for a in args:
            argmap[g.add_parameter()] = (a, self._is_leaf(a))
        g.output = self._generate_helper(g, fnarg, argmap)
        return g

    def __call__(self, *all_args):
        """Python implementation of HyperMap's functionality."""
        assert self.broadcast  # TODO: implement the non-broadcast version

        def _is_nonleaf(x):
            return (
                (isinstance(x, list)
                 and abstract.AbstractList in self.nonleaf)
                or (isinstance(x, tuple)
                    and abstract.AbstractTuple in self.nonleaf)
                or (isinstance(x, np.ndarray)
                    and abstract.AbstractArray in self.nonleaf)
                or (is_dataclass(x)
                    and abstract.AbstractClass in self.nonleaf)
            )

        def _reccall(args):
            if fnarg is None:
                return self.fn_rec(*args)
            else:
                return self.fn_rec(fnarg, *args)

        def _leafcall(args):
            if fnarg is None:
                return self.fn_leaf(*args)
            else:
                return fnarg(*args)

        if self.fn_leaf is None:
            fnarg, *args = all_args
        else:
            fnarg = None
            args = all_args

        argmap = [(x, _is_nonleaf(x)) for x in args]

        nonleafs = [x for x, nonleaf in argmap if nonleaf]
        assert len(set(map(type, nonleafs))) <= 1

        if not nonleafs:
            return _leafcall(args)

        main = nonleafs[0]

        if isinstance(main, (list, tuple)):
            assert all(len(x) == len(main) for x in nonleafs[1:])
            results = []
            for i in range(len(main)):
                args = [x[i] if nonleaf else x
                        for x, nonleaf in argmap]
                results.append(_reccall(args))
            return type(main)(results)

        elif is_dataclass(main):
            results = {}
            for name, field in main.__dataclass_fields__.items():
                args = [getattr(x, name) if nonleaf else x
                        for x, nonleaf in argmap]
                results[name] = _reccall(args)
            return type(main)(**results)

        elif isinstance(main, np.ndarray):
            if fnarg is None:
                fnarg = self.fn_leaf
            return array_map(fnarg, *args)

        else:
            raise AssertionError('Should be unreachable')


hyper_map = HyperMap()
