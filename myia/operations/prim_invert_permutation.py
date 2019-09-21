"""Definitions for the primitive `invert_permutation`."""

from .. import xtype
from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    AbstractTuple,
    standard_prim,
    u64tup_typecheck,
)
from . import primitives as P


def pyimpl_invert_permutation(perm):
    """Implement `invert_permutation`."""
    return tuple(perm.index(i) for i in range(len(perm)))


@standard_prim(P.invert_permutation)
async def infer_invert_permutation(self, engine, perm: u64tup_typecheck):
    """Infer the return type of primitive `invert_permutation`."""
    v = [x.xvalue() for x in perm.elements]
    return AbstractTuple([
        AbstractScalar({
            VALUE: v.index(i),
            TYPE: xtype.UInt[64],
        })
        if i in v
        else AbstractScalar({
            VALUE: ANYTHING,
            TYPE: xtype.UInt[64],
        }) for i in range(len(v))
    ])


__operation_defaults__ = {
    'name': 'invert_permutation',
    'registered_name': 'invert_permutation',
    'mapping': P.invert_permutation,
    'python_implementation': pyimpl_invert_permutation,
}


__primitive_defaults__ = {
    'name': 'invert_permutation',
    'registered_name': 'invert_permutation',
    'type': 'backend',
    'python_implementation': pyimpl_invert_permutation,
    'inferrer_constructor': infer_invert_permutation,
    'grad_transform': None,
}
