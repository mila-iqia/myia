
from unification import Var, unifiable, unify as _unify, reify
from unification import isvar  # type: ignore
from unification.dispatch import dispatch


# TODO: use the typing module instead. Int8 etc. would have to be defined
# as types in order to appear in typing.Tuple and so on.


class RestrictedVar:
    # unification.Var does some magic with __new__ and its __eq__
    # is not compatible with what we are trying to do here, so it's
    # best not to subclass it.
    def __init__(self, token, legal_values):
        self.token = token
        self.legal_values = legal_values

    def __str__(self):
        return "~" + str(self.token)

    __repr__ = __str__


class FilterVar:
    def __init__(self, token, filter):
        self.token = token
        self.filter = filter

    def __str__(self):
        return "~" + str(self.token)

    __repr__ = __str__


@dispatch(RestrictedVar)  # type: ignore
def isvar(v):
    # Extend unification.isvar to recognize RestrictedVar.
    return True


@dispatch(FilterVar)  # type: ignore
def isvar(v):
    # Extend unification.isvar to recognize FilterVar.
    return True


def var(token, filter=None):
    """
    Create a variable for unification purposes.

    Arguments:
        token: The name of the variable.
        filter: A predicate, or a set of values the variable is
            allowed to take.
    """
    if callable(filter):
        return FilterVar(token, filter)
    elif filter:
        return RestrictedVar(token, filter)
    else:
        return Var(token)


def unify(a, b, U=None):
    """
    Unify a and b and return a dictionary associating variables to
    values that can unify a and b. This takes into account the legal
    values a RestrictedVar can take.
    """
    if U is None:
        d = _unify(a, b)
    else:
        d = _unify(a, b, U)
    if not d:
        return d
    for v, value in d.items():
        if isinstance(v, RestrictedVar):
            if value not in v.legal_values:
                return False
        elif isinstance(v, FilterVar):
            if not v.filter(value):
                return False
    return d


@unifiable
class Type:
    def __init__(self, name, elem_types=None):
        self.name = name
        self.elem_types = elem_types

    def __getitem__(self, elem_types):
        assert self.elem_types == ()
        if not isinstance(elem_types, tuple):
            elem_types = (elem_types,)
        return Type(self.name, elem_types)

    def __hash__(self):
        return hash(self.name) ^ hash(self.elem_types)

    def __eq__(self, other):
        return isinstance(other, Type) \
            and other.name == self.name \
            and other.elem_types == self.elem_types

    def __str__(self):
        if self.elem_types:
            etypes = ", ".join(map(str, self.elem_types))
            return f'{self.name}[{etypes}]'
        else:
            return self.name

    def __repr__(self):
        return str(self)


Bool = Type('Bool')
Float32 = Type('Float32')
Float64 = Type('Float64')
Int8 = Type('Int8')
Int16 = Type('Int16')
Int32 = Type('Int32')
Int64 = Type('Int64')
UInt8 = Type('UInt8')
UInt16 = Type('UInt16')
UInt32 = Type('UInt32')
UInt64 = Type('UInt64')
Array = Type('Array', ())
List = Type('List', ())
Tuple = Type('Tuple', ())
Record = Type('Record', ())


Number = {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64
}


type_map = {
    bool: Bool,
    float: Float64,
    int: Int64,
    'float32': Float32,
    'float64': Float64,
    'int8': Int8,
    'int16': Int16,
    'int32': Int32,
    'int64': Int64,
    'uint8': Int8,
    'uint16': Int16,
    'uint32': Int32,
    'uint64': Int64
}


def typeof(x):
    t = type(x)
    res = type_map.get(type(x), None)
    if res:
        return res
    if isinstance(x, tuple):
        return Tuple[tuple(map(typeof, x))]
    if t.__name__ == 'ndarray' and hasattr(x, 'dtype'):
        return Array[type_map.get(x.dtype.name)]
    raise TypeError(f'Unknown data type: {type(x)}')
