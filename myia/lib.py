
import inspect
import numpy
from types import FunctionType
from copy import copy
from .util.buche import HReprBase
from .util.misc import Singleton


##############
# Singletons #
##############


class ZERO(Singleton):
    """
    ZERO serves as a generic zero: add(ZERO, y) == y, whether y is a scalar,
    a tuple, or whatever else. This is more efficient than creating a zero
    that has the same shape as y.
    """
    def __add__(self, other):
        return other

    def __map__(self, smap, *rest):
        return smap.fn(self, *rest)


class VALUE(Singleton):
    """
    Used as a key in myia.inference.AbstractValue to contain the actual value
    of the data at that point.
    """
    pass


class ERROR(Singleton):
    """
    Used as a key in myia.inference.AbstractValue to contain the error that
    occurred on the path to that value.
    """
    pass


class ANY(Singleton):
    """
    Represents any value.
    """
    pass


# Set singleton classes to their only instance

ZERO = ZERO()    # type: ignore
VALUE = VALUE()  # type: ignore
ERROR = ERROR()  # type: ignore
ANY = ANY()      # type: ignore


###################################
# Subclasses for mappable objects #
###################################

class IdempotentMappable:

    def __idem__(self, other):
        if self != other:
            raise ValueError('Expected two identical instances, but got:'
                             f' `{self}` and `{other}`')
        return self

    __add__ = __idem__
    __sub__ = __idem__
    __mul__ = __idem__
    __truediv__ = __idem__
    __floordiv__ = __idem__
    __pow__ = __idem__
    __mod__ = __idem__

    def __map__(self, smap, *others):
        return smap.fn(self, *others)


class StructuralMappable:

    def __add__(self, other):
        return structural_map(lambda x, y: x + y, self, other)

    def __sub__(self, other):
        return structural_map(lambda x, y: x - y, self, other)

    def __mul__(self, other):
        return structural_map(lambda x, y: x * y, self, other)

    def __truediv__(self, other):
        return structural_map(lambda x, y: x / y, self, other)

    def __floordiv__(self, other):
        return structural_map(lambda x, y: x // y, self, other)

    def __pow__(self, other):
        return structural_map(lambda x, y: x ** y, self, other)

    def __mod__(self, other):
        return structural_map(lambda x, y: x % y, self, other)

    def __map__(self, smap, *others):
        return smap.fn(self, *others)


########################
# Myia data structures #
########################


class Primitive(HReprBase, IdempotentMappable):
    """
    Wrapper around a pure Python implementation of a function.
    """
    def __init__(self, fn, name=None):
        argn = inspect.getargs(fn.__code__).args  # type: ignore
        self.argnames: List[str] = argn
        self.nargs = len(self.argnames)
        self.fn = fn
        if name:
            self.name = name
            self.__myia_symbol__ = name
        else:
            self.name = fn.__name__
        self.grad = None

    def __call__(self, *args):
        return self.fn(*args)

    def __hash__(self):
        return hash((self.fn, self.name))

    def __eq__(self, other):
        return isinstance(other, Primitive) \
            and self.fn == other.fn \
            and self.name == other.name

    def __str__(self):
        return f'Prim({self.name or self.fn})'

    def __repr__(self):
        return str(self)

    def __map__(self, smap, *others):
        return smap.fn(self, *others)

    def __hrepr__(self, H, hrepr):
        return hrepr.titled_box('Prim', [hrepr(self.name or self.fn)])


class Closure(HReprBase, StructuralMappable):
    """
    Associates a Primitive or a Function to a number
    of arguments in order to create a partial application.
    """
    def __init__(self, fn, args) -> None:
        self.fn = fn
        self.args = tuple(args)

    def __call__(self, *args):
        return self.fn(*self.args, *args)

    def __map__(self, smap, *clos):
        smap.require_same([type, lambda c: len(c.args)], [self, *clos])
        return Closure(smap(self.fn, *[c.fn for c in clos]),
                       smap(self.args, *[c.args for c in clos]))

    def __str__(self):
        return f'Clos({self.fn}, {self.args})'

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return hrepr.titled_box('Closure',
                                [hrepr(self.fn),
                                 hrepr(self.args)], 'v')


class Function(HReprBase, IdempotentMappable):
    pass


class Atom:
    def __init__(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return str(self.identifier)

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return hrepr(self.identifier)

    def __call__(self, **kw):
        return Record(self, kw)


class Record(HReprBase, StructuralMappable):
    def __init__(self, tag, kw):
        assert isinstance(tag, Atom)
        self.__dict__.update(kw, __tag__=tag)

    def __setattr__(self, attr, value):
        raise AttributeError(f"Cannot set attribute '{attr}' of {self}"
                             " -- Records are read-only.")

    def __getitem__(self, item):
        return getattr(self, item)

    def __variant__(self, field, value):
        obj = copy(self)
        obj.__dict__[field] = value
        return obj

    def __map__(self, smap, *recs):
        smap.require_same([type, lambda r: (r.__tag__, r.__dict__.keys())],
                          [self, *recs])
        acc = {}
        for k, v in self:
            acc[k] = smap(v, *[rec[k] for rec in recs])
        return Record(self.__tag__, acc)

    def __iter__(self):
        for k, v in self.__dict__.items():
            if k != '__tag__':
                yield k, v

    def __str__(self):
        entries = ", ".join(f'{k}={repr(v)}' for k, v in self)
        return f'{self.__tag__}({entries})'

    def __or__(self, other):
        if not isinstance(other, Record):
            return NotImplemented
        assert other.__tag__ is self.__tag__
        d = {**self.__dict__}
        d.update(other.__dict__)
        return Record(self.__tag__, d)

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            str(self.__tag__),
            [(k, v) for k, v in self]
        )


def same_record_type(r1, r2):
    return isinstance(r1, Record) and isinstance(r2, Record) and \
        r1.__dict__.keys() == r2.__dict__.keys()


def scalar_map(smap, *scalars):
    return smap.fn(*scalars)


def sequence_map(smap, *seqs):
    s0 = seqs[0]
    t = type(s0)
    n = len(s0)
    smap.require_same([type, len], seqs)
    return t(smap(*[s[i] for s in seqs]) for i in range(len(s0)))


def ndarray_map(smap, *arrs):
    return numpy.vectorize(smap.fn)(*arrs)


default_structural_map_dispatch = {
    int: scalar_map,
    float: scalar_map,
    bool: scalar_map,
    str: scalar_map,
    type(None): scalar_map,
    tuple: sequence_map,
    list: sequence_map,
    numpy.float64: scalar_map,
    numpy.float32: scalar_map,
    numpy.int64: scalar_map,
    numpy.int32: scalar_map,
    numpy.ndarray: ndarray_map
}


class StructuralMap:
    def __init__(self, fn, dispatch=default_structural_map_dispatch):
        self.fn = fn
        self.dispatch = dispatch

    @property
    def __code__(self):
        return self.fn.__code__

    def require_same(self, fns, objs):
        o, *rest = objs
        for fn in fns:
            for obj in rest:
                if fn(o) != fn(obj):
                    raise TypeError("Arguments to 'structural_map' do not"
                                    f" have the same properties:"
                                    f" `{o}` and `{obj}` are not conformant.")

    def __call__(self, *data):
        d0 = data[0]
        t = type(d0)
        if t in self.dispatch:
            return self.dispatch[t](self, *data)
        elif hasattr(d0, '__map__'):
            return d0.__map__(self, *data[1:])
        else:
            return self.fn(*data)
            # raise TypeError(f"'structural_map' is not defined for data"
            #                 f" of type {t}.")


def structural_map(fn, *args):
    return StructuralMap(fn)(*args)


#########
# Atoms #
#########


TrueAtom = Atom('True')
FalseAtom = Atom('False')
NoneAtom = Atom('None')


################
# Record types #
################


record = Atom('record')
TupleAtom = Atom('tuple')


def tuple_record(*args):
    return Record(TupleAtom, {i: v for i, v in enumerate(args)})


############
# Universe #
############


def is_struct(x):
    # set, frozenset, dict
    return isinstance(x, (list, tuple, Record, Closure))


class Universe:
    __cachable__ = (FunctionType,)

    def __init__(self):
        self.cache = {}

    def acquire(self, item):
        raise NotImplementedError()

    def __copy__(self):
        raise Exception('COP')

    def __deepcopy__(self):
        raise Exception('DCOP')

    def __getitem__(self, item):
        if isinstance(item, Universe.__cachable__):
            try:
                return self.cache[item]
            except KeyError:
                v = self.acquire(item)
                self.cache[item] = v
                return v
        else:
            return self.acquire(item)


class BackedUniverse(Universe):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
