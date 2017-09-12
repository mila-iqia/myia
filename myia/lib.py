

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


class Closure(StructuralMappable):
    def __init__(self, fn, args) -> None:
        self.fn = fn
        self.args = tuple(args)

    def __call__(self, *args):
        return self.fn(*self.args, *args)

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


class Record(StructuralMappable):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __setattr__(self, attr, value):
        raise AttributeError(f"Cannot set attribute '{attr}' of {self}"
                             " -- Records are read-only.")

    def __getitem__(self, item):
        return getattr(self, item)

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


def sequence_map(smap, *seqs):
    s0 = seqs[0]
    t = type(s0)
    n = len(s0)
    if not all(n == len(s) for s in seqs[1:]):
        raise TypeError(f"All arguments in 'structural_map' must"
                        f" have the same type and the same length.")

    return t(smap(*[s[i] for s in seqs]) for i in range(len(s0)))


def record_map(smap, *recs):
    if not all(same_record_type(recs[0], r2) for r2 in recs[1:]):
        raise TypeError(f"'record_map' on multiple records requires all "
                        f"arguments to be records, and to have the same keys.")
    acc = {}
    for k in recs[0].__dict__.keys():
        acc[k] = smap(*[rec[k] for rec in recs])
    return Record(**acc)


def closure_map(smap, *clos):
    if not all(clos[0].fn is c2.fn for c2 in clos[1:]):
        raise TypeError(f"'closure_map' on multiple closures requires all"
                        f" closures to have the same underlying function.")
    return Closure(clos[0].fn, sequence_map(smap, *[c.args for c in clos]))


def ndarray_map(smap, *arrs):
    return numpy.vectorize(smap.fn)(*arrs)


default_structural_map_dispatch = {
    tuple: sequence_map,
    list: sequence_map,
    Closure: closure_map,
    Record: record_map
    # numpy.ndarray: ndarray_map
}


class StructuralMap:
    def __init__(self, fn, dispatch=default_structural_map_dispatch):
        self.fn = fn
        self.dispatch = dispatch

    @property
    def __code__(self):
        return self.fn.__code__

    def __call__(self, *data):
        d0 = data[0]
        t = type(d0)

        if t is int or t is float:
            return self.fn(*data)

        if not all(type(d) is t for d in data[1:]):
            raise TypeError(f"All arguments in 'structural_map' must"
                            f" have exactly the same type.")

        if t in self.dispatch:
            return self.dispatch[t](self, *data)
        else:
            raise TypeError(f"'structural_map' is not defined for data"
                            f" of type {t}.")


def structural_map(fn, *args):
    return StructuralMap(fn)(*args)
