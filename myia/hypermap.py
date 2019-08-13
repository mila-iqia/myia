"""Generate mapping graphs over classes, tuples, arrays, etc."""

from dataclasses import is_dataclass
from functools import reduce

import numpy as np

from . import abstract, operations
from .abstract import broaden
from .ir import Graph, MetaGraph
from .prim import ops as P
from .prim.py_implementations import array_map
from .utils import MyiaTypeError, Overload

nonleaf_defaults = (
    abstract.AbstractArray,
    abstract.AbstractTuple,
    abstract.AbstractDict,
    abstract.AbstractClassBase,
    abstract.AbstractUnion,
    abstract.AbstractTaggedUnion,
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
                 trust_union_match=False,
                 name=None):
        """Initialize a HyperMap."""
        if name is None:
            if fn_leaf is None:
                name = 'hyper_map'
            else:
                name = f'hyper_map[{fn_leaf}]'
        super().__init__(name)
        self.fn_leaf = fn_leaf
        self._rec = fn_rec
        self.fn_rec = fn_rec or self
        self.broadcast = broadcast
        self.trust_union_match = trust_union_match
        self.nonleaf = nonleaf
        self._through = (*nonleaf,
                         abstract.Possibilities,
                         abstract.TaggedPossibilities)

    async def normalize_args(self, args):
        """Return broadened arguments."""
        res = [await abstract.force_through(arg, self._through)
               for arg in args]
        return self.normalize_args_sync(res)

    def normalize_args_sync(self, args):
        """Return broadened arguments."""
        return tuple(broaden(a) for a in args)

    def _name(self, graph, info):
        graph.debug.name = f'{self.name}[{info}]'

    def _is_nonleaf(self, arg):
        return isinstance(arg, self.nonleaf)

    def _is_leaf(self, arg):
        return not self._is_nonleaf(arg)

    _make = Overload(name='hypermap._make')

    def _make_union_helper(self, a, options, g, fnarg, argmap):
        # Options must be a list of (tag, type) pairs. If the tag is None,
        # we generate hastype, unsafe_static_cast and tagged(x)
        # Else we generate hastag, casttag and tagged(x, tag)
        trust = self.trust_union_match or len(argmap) == 1

        self._name(g, f'U')

        for a2, isleaf in argmap.values():
            if not isleaf and a != a2:
                raise MyiaTypeError(f'Union mismatch: {a} != {a2}')

        currg = g

        for i, (tag, t) in enumerate(options):
            is_last = i == len(options) - 1
            if tag is None:
                terms = [currg.apply(P.hastype, arg, t)
                         for arg, (_, nonleaf) in argmap.items()]
            else:
                terms = [currg.apply(P.hastag, arg, tag)
                         for arg, (_, nonleaf) in argmap.items()]
            if trust:
                terms = terms[:1]

            cond = reduce(lambda x, y: currg.apply(P.bool_and, x, y),
                          terms)

            if is_last and trust:
                trueg = currg
            else:
                trueg = Graph()
                self._name(trueg, f'U✓{tag}')

            if tag is None:
                args = [arg if isleaf
                        else trueg.apply(P.unsafe_static_cast, arg, t)
                        for arg, (_, isleaf) in argmap.items()]
            else:
                args = [arg if isleaf
                        else trueg.apply(P.casttag, arg, tag)
                        for arg, (_, isleaf) in argmap.items()]

            if fnarg is None:
                val = trueg.apply(self.fn_rec, *args)
            else:
                val = trueg.apply(self.fn_rec, fnarg, *args)

            if tag is None:
                trueg.output = trueg.apply(P.tagged, val)
            else:
                trueg.output = trueg.apply(P.tagged, val, tag)

            if not (is_last and trust):
                falseg = Graph()
                currg.output = currg.apply(
                    currg.apply(P.switch, cond, trueg, falseg)
                )
                currg = falseg

        if currg.return_ is None:
            currg.output = currg.apply(
                P.raise_,
                currg.apply(P.exception, "Type mismatch.")
            )
            currg.debug.name = 'type_error'

        g.set_flags(core=False)
        return g.output

    @_make.register
    def _make(self, a: abstract.AbstractUnion, g, fnarg, argmap):
        options = [[None, x] for x in a.options]
        return self._make_union_helper(a, options, g, fnarg, argmap)

    @_make.register
    def _make(self, a: abstract.AbstractTaggedUnion, g, fnarg, argmap):
        return self._make_union_helper(a, a.options, g, fnarg, argmap)

    @_make.register
    def _make(self, t: abstract.AbstractArray, g, fnarg, argmap):
        self._name(g, 'A')

        if fnarg is None:
            fnarg = self.fn_leaf

        if len(argmap) > 1 and self.broadcast:
            args = [g.apply(operations.to_array, arg, t)
                    if isleaf else arg
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
        self._name(g, 'T')
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
    def _make(self, a: abstract.AbstractDict, g, fnarg, argmap):
        for a2, isleaf in argmap.values():
            if not isleaf and a.entries.keys() != a2.entries.keys():
                raise MyiaTypeError(f'Dict keys mismatch: {a} != {a2}')

        elems = []
        for k, v in a.entries.items():
            args = [arg if isleaf else g.apply(P.dict_getitem, arg, k)
                    for arg, (_, isleaf) in argmap.items()]
            if fnarg is None:
                val = g.apply(self.fn_rec, *args)
            else:
                val = g.apply(self.fn_rec, fnarg, *args)
            elems.append(val)
        return g.apply(P.make_dict, a, *elems)

    @_make.register
    def _make(self, a: abstract.AbstractClassBase, g, fnarg, argmap):
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

        # We recover the original, potentially more generic type corresponding
        # to the tag. This allows the mapping function to return a different
        # type from its input.
        original = a.user_defined_version()
        return g.apply(P.make_record, original, *vals)

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
        g.debug.name = self.name
        g.set_flags('core', 'reference', metagraph=self)
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

    def __eq__(self, hm):
        return (isinstance(hm, HyperMap)
                and self.fn_leaf == hm.fn_leaf
                and self._rec == hm._rec
                and self.nonleaf == hm.nonleaf
                and self.broadcast == hm.broadcast)

    def __hash__(self):
        return hash((self.fn_leaf, self._rec, self.nonleaf, self.broadcast))

    def __call__(self, *all_args):
        """Python implementation of HyperMap's functionality."""
        assert self.broadcast  # TODO: implement the non-broadcast version

        def _is_nonleaf(x):
            return (
                (isinstance(x, list)
                 and abstract.AbstractClassBase in self.nonleaf)
                or (isinstance(x, tuple)
                    and abstract.AbstractTuple in self.nonleaf)
                or (isinstance(x, dict)
                    and abstract.AbstractDict in self.nonleaf)
                or (isinstance(x, np.ndarray)
                    and abstract.AbstractArray in self.nonleaf)
                or (is_dataclass(x)
                    and abstract.AbstractClassBase in self.nonleaf)
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
