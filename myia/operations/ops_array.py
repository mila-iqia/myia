"""Array operations."""

from dataclasses import dataclass

from .. import lib, operations
from ..hypermap import HyperMap
from ..lib import core, myia_static
from ..operations import (
    array_reduce,
    primitives as P,
    scalar_add,
    scalar_mul,
    shape,
)
from .utils import OperationDefinition, to_opdef


def elemwise(name, op, infer_value=False):
    """Define an elemwise operation on one or more arrays."""
    hm = HyperMap(
        name=name,
        fn_leaf=op,
        nonleaf=(lib.AbstractArray,),
        infer_value=infer_value,
    )
    return OperationDefinition(
        name=name, registered_name=name, mapping=hm, python_implementation=None
    )


array_add = elemwise("array_add", operations.add)
array_sub = elemwise("array_sub", operations.sub)
array_mul = elemwise("array_mul", operations.mul)
array_mod = elemwise("array_mod", operations.mod)
array_pow = elemwise("array_pow", operations.pow)
array_exp = elemwise("array_exp", operations.scalar_exp)
array_log = elemwise("array_log", operations.scalar_log)
array_sin = elemwise("array_sin", operations.scalar_sin)
array_cos = elemwise("array_cos", operations.scalar_cos)
array_tan = elemwise("array_tan", operations.scalar_tan)
array_tanh = elemwise("array_tanh", operations.scalar_tanh)
array_abs = elemwise("array_abs", operations.scalar_abs)
array_sign = elemwise("array_sign", operations.scalar_sign)
array_floor = elemwise("array_floor", operations.floor)
array_trunc = elemwise("array_trunc", operations.trunc)
array_uadd = elemwise("array_uadd", operations.pos)
array_usub = elemwise("array_usub", operations.neg)
array_truediv = elemwise("array_truediv", operations.truediv)
array_floordiv = elemwise("array_floordiv", operations.floordiv)

array_eq = elemwise("array_eq", operations.eq, infer_value=True)
array_lt = elemwise("array_lt", operations.lt, infer_value=True)
array_gt = elemwise("array_gt", operations.gt, infer_value=True)
array_ne = elemwise("array_ne", operations.ne, infer_value=True)
array_le = elemwise("array_le", operations.le, infer_value=True)
array_ge = elemwise("array_ge", operations.ge, infer_value=True)


@to_opdef
@core
def sum(x):
    """Implementation of `sum`."""
    return array_reduce(scalar_add, x, ())


@to_opdef
@core
def prod(x):
    """Implementation of `np.prod`. Parameter `axis` is not yet supported."""
    return array_reduce(scalar_mul, x, ())


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
