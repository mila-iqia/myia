"""Generate mapping graphs over classes, tuples, arrays, etc."""


from . import operations, composite as C
from .abstract import InferenceError
from .ir import MetaGraph, Graph
from .dtype import Array, List, Tuple, Class, tag_to_dataclass, \
    pytype_to_myiatype
from .utils import TypeMap, Overload
from .prim import ops as P
from .prim.py_implementations import issubtype


class HyperMap(MetaGraph):
    """Map over tuples, classes, lists and arrays.

    Arguments:
        fn_leaf: The function to apply on leaves. If it is None,
            the function is passed as the first argument.
        fn_rec: The function to apply recursively. If it is None,
            the HyperMap will apply itself. Using another function
            will implement a "shallow" HyperMap.
        broadcast: Whether to automatically broadcast arguments
            when one of them is an Array (default: True).
        nonleaf: List of Types to generate a recursive map over.
            Any type not in this list will generate a call to
            fn_leaf.

    """

    def __init__(self, *,
                 fn_leaf=None,
                 fn_rec=None,
                 broadcast=True,
                 nonleaf=(Array, List, Tuple, Class)):
        """Initialize a HyperMap."""
        if fn_leaf is None:
            name = 'hyper_map'
        else:
            name = f'hyper_map[{fn_leaf}]'
        super().__init__(name)
        self.fn_leaf = fn_leaf
        self.fn_rec = fn_rec or self
        self.broadcast = broadcast
        # Pick out only the types we want to generate a mapping over.
        self.make_map = TypeMap()
        for t in (*nonleaf, object):
            self.make_map[t] = self._full_make.map[t]
        self.nonleaf = nonleaf

    _full_make = Overload()

    @_full_make.register
    def _full_make(self, t: List, g, fnarg, argmap):
        args = list(argmap.keys())
        if fnarg is None:
            fn_rec = self.fn_rec
        else:
            fn_rec = g.apply(P.partial, self.fn_rec, fnarg)
        return g.apply(C.list_map, fn_rec, *args)

    @_full_make.register
    def _full_make(self, t: Array, g, fnarg, argmap):
        if fnarg is None:
            fnarg = self.fn_leaf

        args = list(argmap.keys())
        first, *rest = args

        if rest and self.broadcast:
            shp = g.apply(P.shape, first)
            for other in rest:
                shp2 = g.apply(P.shape, other)
                shp = g.apply(P.broadcast_shape, shp2, shp)
            args = [g.apply(P.distribute, arg, shp) for arg in args]

        return g.apply(P.array_map, fnarg, *args)

    @_full_make.register
    def _full_make(self, t: Tuple, g, fnarg, argmap):
        assert all(len(t.elements) == len(t2.elements)
                   for t2 in argmap.values())
        elems = []
        for i in range(len(t.elements)):
            args = [g.apply(P.tuple_getitem, arg, i)
                    for arg in argmap.keys()]
            if fnarg is None:
                val = g.apply(self.fn_rec, *args)
            else:
                val = g.apply(self.fn_rec, fnarg, *args)
            elems.append(val)
        return g.apply(P.make_tuple, *elems)

    @_full_make.register
    def _full_make(self, t: Class, g, fnarg, argmap):
        assert all(t.tag == t2.tag
                   and t.attributes.keys() == t2.attributes.keys()
                   for t2 in argmap.values())
        vals = []
        for k in t.attributes.keys():
            args = [g.apply(P.getattr, arg, k)
                    for arg in argmap.keys()]
            if fnarg is None:
                val = g.apply(self.fn_rec, *args)
            else:
                val = g.apply(self.fn_rec, fnarg, *args)
            vals.append(val)
        t = pytype_to_myiatype(tag_to_dataclass[t.tag])
        return g.apply(P.make_record, t, *vals)

    @_full_make.register
    def _full_make(self, t: object, g, fnarg, argmap):
        if fnarg is None:
            fnarg = self.fn_leaf
        return g.apply(fnarg, *argmap.keys())

    def _make(self, g, fnarg, argmap):
        for t in argmap.values():
            # If any of the arguments is a nonleaf generic, pick it.
            if hasattr(t, 'generic') and t.generic in self.nonleaf:
                break
        if hasattr(t, 'generic') and t.generic in self.nonleaf:
            # In a nonleaf situation, all arguments must have the same
            # generic.
            for t2 in argmap.values():
                if t.generic is not t2.generic:
                    raise InferenceError(
                        f'HyperMap cannot match up types {t} and {t2}'
                    )
        return self.make_map[t](self, t, g, fnarg, argmap)

    def _harmonize(self, g, args):
        if self.broadcast \
                and any(issubtype(t, Array) for t in args.values()):
            rval = {}
            for arg, t in args.items():
                if not issubtype(t, Array):
                    arg = g.apply(operations.to_array, arg)
                    t = Array[t]
                rval[arg] = t
            return rval
        else:
            return args

    def generate_from_types(self, types):
        """Create a graph for mapping over the given types."""
        g = Graph()
        g.debug.name = 'hyper_map'
        argmap = {}
        if self.fn_leaf is None:
            fn_t, *arg_ts = types
            fnarg = g.add_parameter()
        else:
            arg_ts = types
            fnarg = None
        for t in arg_ts:
            argmap[g.add_parameter()] = t
        argmap = self._harmonize(g, argmap)
        g.output = self._make(g, fnarg, argmap)
        return g
