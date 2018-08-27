"""Generate mapping graphs over classes, tuples, arrays, etc."""


from . import composite as C
from .infer import InferenceError
from .ir import MetaGraph, Graph
from .dtype import Array, List, Tuple, Class, Type, tag_to_dataclass, \
    pytype_to_myiatype
from .utils import TypeMap
from .prim import ops as P
from .prim.py_implementations import hastype_helper


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
            self.name = 'hyper_map'
        else:
            self.name = f'hyper_map[{fn_leaf}]'
        self.fn_leaf = fn_leaf
        self.fn_rec = fn_rec or self
        self.broadcast = broadcast
        # Pick out only the types we want to generate a mapping over.
        self.make_map = TypeMap()
        for t in (*nonleaf, Type):
            self.make_map[t] = self.full_make_map[t]
        self.nonleaf = nonleaf
        self.cache = {}

    full_make_map = TypeMap()

    @full_make_map.register(List)
    def _make_List(self, resources, t, g, fnarg, argmap):
        args = list(argmap.keys())
        if fnarg is None:
            fn_rec = self.fn_rec
        else:
            fn_rec = g.apply(P.partial, self.fn_rec, fnarg)
        return g.apply(P.list_map, fn_rec, *args)

    @full_make_map.register(Array)
    def _make_Array(self, resources, t, g, fnarg, argmap):
        if fnarg is None:
            fnarg = resources.convert(self.fn_leaf)

        args = list(argmap.keys())
        first, *rest = args

        if rest and self.broadcast:
            shp = g.apply(P.shape, first)
            for other in rest:
                shp2 = g.apply(P.shape, other)
                shp = g.apply(P.broadcast_shape, shp2, shp)
            args = [g.apply(P.distribute, arg, shp) for arg in args]

        return g.apply(P.array_map, fnarg, *args)

    @full_make_map.register(Tuple)
    def _make_Tuple(self, resources, t, g, fnarg, argmap):
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

    @full_make_map.register(Class)
    def _make_Class(self, resources, t, g, fnarg, argmap):
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

    @full_make_map.register(Type)
    def _make_Type(self, resources, t, g, fnarg, argmap):
        if fnarg is None:
            fnarg = resources.convert(self.fn_leaf)
        return g.apply(fnarg, *argmap.keys())

    def _make(self, resources, g, fnarg, argmap):
        for t in argmap.values():
            # If any of the arguments is a nonleaf generic, pick it.
            if t.generic in self.nonleaf:
                break
        if t.generic in self.nonleaf:
            # In a nonleaf situation, all arguments must have the same
            # generic.
            for t2 in argmap.values():
                if t.generic is not t2.generic:
                    raise InferenceError(
                        f'HyperMap cannot match up types {t} and {t2}'
                    )
        return self.make_map[t](self, resources, t, g, fnarg, argmap)

    def _harmonize(self, resources, g, args):
        if self.broadcast \
                and any(hastype_helper(t, Array) for t in args.values()):
            rval = {}
            for arg, t in args.items():
                if not hastype_helper(t, Array):
                    arg = g.apply(resources.convert(C.to_array), arg)
                    t = Array[t]
                rval[arg] = t
            return rval
        else:
            return args

    def specialize(self, resources, types):
        """Create a graph for mapping over the given types."""
        types = tuple(types)
        if types in self.cache:
            return self.cache[types]
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
        argmap = self._harmonize(resources, g, argmap)
        g.output = self._make(resources, g, fnarg, argmap)
        resources.manager.add_graph(g)
        self.cache[types] = g
        return g
