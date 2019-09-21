"""Array operations."""

from dataclasses import dataclass

from .. import operations
from ..hypermap import Elemwise
from ..lib import core, myia_static
from ..operations import array_reduce, primitives as P, scalar_add, shape
from .utils import to_opdef


def arrayify(name, op):
    """Create an array_map over the op."""
    @core(name=name)
    def arrayified(xs):
        return operations.array_map(op, xs)

    return {
        'name': name,
        'registered_name': name,
        'mapping': arrayified,
        'python_implementation': None,
    }


def elemwise(name, field, op, pyop=None, **kwargs):
    """Create an operation from an application of Elemwise."""
    ew = Elemwise(field, op, name=name, **kwargs)
    return {
        'name': name,
        'registered_name': name,
        'mapping': ew,
        'python_implementation': pyop,
    }


array_add = elemwise('array_add', '__add__', operations.scalar_add)
array_sub = elemwise('array_sub', '__sub__', operations.scalar_sub)
array_mul = elemwise('array_mul', '__mul__', operations.scalar_mul)
array_mod = elemwise('array_mod', '__mod__', operations.scalar_mod)
array_pow = elemwise('array_pow', '__pow__', operations.scalar_pow)
array_exp = elemwise('array_exp', None, operations.scalar_exp)
array_log = elemwise('array_log', None, operations.scalar_log)
array_sin = elemwise('array_sin', None, operations.scalar_sin)
array_cos = elemwise('array_cos', None, operations.scalar_cos)
array_tan = elemwise('array_tan', None, operations.scalar_tan)
array_tanh = elemwise('array_tanh', None, operations.scalar_tanh)
array_floor = arrayify('array_floor', operations.floor)
array_trunc = arrayify('array_trunc', operations.trunc)
array_uadd = arrayify('array_uadd', operations.pos)
array_usub = arrayify('array_usub', operations.neg)
array_truediv = elemwise('array_truediv', '__truediv__', None)
array_floordiv = elemwise('array_floordiv', '__floordiv__', None)

array_eq = elemwise('array_eq', '__eq__', operations.scalar_eq,
                    infer_value=True)
array_lt = elemwise('array_lt', '__lt__', operations.scalar_lt,
                    infer_value=True)
array_gt = elemwise('array_gt', '__gt__', operations.scalar_gt,
                    infer_value=True)
array_ne = elemwise('array_ne', '__ne__', operations.scalar_ne,
                    infer_value=True)
array_le = elemwise('array_le', '__le__', operations.scalar_le,
                    infer_value=True)
array_ge = elemwise('array_ge', '__ge__', operations.scalar_ge,
                    infer_value=True)


@to_opdef
@core
def sum(x):
    """Implementation of `sum`."""
    return array_reduce(scalar_add, x, ())


@to_opdef
@core
def ndim(arr):
    """Return the number of dimensions of an array."""
    return len(shape(arr))


@myia_static
def _revperm(n):
    return tuple(reversed(range(n)))


@to_opdef
@core
def t(arr, permutation=None):
    """Transpose an array."""
    if permutation is None:
        permutation = _revperm(operations.ndim(arr))
    return P.transpose(arr, permutation)


@dataclass(frozen=True)
class SequenceIterator:
    """Iterator to use for sequences like Array."""

    idx: int
    seq: object

    @core(ignore_values=True)
    def __myia_hasnext__(self):
        """Whether the index is past the length of the sequence."""
        return self.idx < len(self.seq)

    @core(ignore_values=True)
    def __myia_next__(self):
        """Return the next element and a new iterator."""
        return self.seq[self.idx], SequenceIterator(self.idx + 1, self.seq)


@to_opdef
@core
def array_iter(xs):
    """Iterator for Array."""
    return SequenceIterator(0, xs)
