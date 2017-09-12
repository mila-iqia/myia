
import inspect
from myia.util import HReprBase, Singleton


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
        self.name = name or fn.__name__
        self.grad = None

    def __call__(self, *args):
        return self.fn(*args)

    def __str__(self):
        return f'Prim({self.name or self.fn})'

    def __repr__(self):
        return str(self)

    def __map__(self, smap, *others):
        return smap.fn(self, *others)

    def __hrepr__(self, H, hrepr):
        return H.div['Primitive'](
            H.div['class_title']('Primitive'),
            H.div['class_contents'](self.name or hrepr(self.fn))
        )


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
        return H.div['ClosureImpl'](
            H.div['class_title']('Closure'),
            H.div['class_contents'](
                hrepr(self.fn),
                hrepr(self.args)
            )
        )


class Record(HReprBase, StructuralMappable):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __setattr__(self, attr, value):
        raise AttributeError(f"Cannot set attribute '{attr}' of {self}"
                             " -- Records are read-only.")

    def __getitem__(self, item):
        return getattr(self, item)

    def __map__(self, smap, *recs):
        smap.require_same([type, lambda r: r.__dict__.keys()], [self, *recs])
        acc = {}
        for k in self.__dict__.keys():
            acc[k] = smap(self[k], *[rec[k] for rec in recs])
        return Record(**acc)

    def __str__(self):
        entries = ", ".join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'Record({entries})'

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return H.div['Record'](
            H.div['class_title']('Record'),
            H.div['class_contents'](
                hrepr(self.__dict__)
            )
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


# def ndarray_map(smap, *arrs):
#     return numpy.vectorize(smap.fn)(*arrs)


default_structural_map_dispatch = {
    int: scalar_map,
    float: scalar_map,
    bool: scalar_map,
    type(None): scalar_map,
    tuple: sequence_map,
    list: sequence_map,
    # numpy.ndarray: ndarray_map
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
            raise TypeError(f"'structural_map' is not defined for data"
                            f" of type {t}.")


def structural_map(fn, *args):
    return StructuralMap(fn)(*args)
