"""Primitives signatures for the type inference."""
from typing import Dict as DictT

from myia.dtype import Function, Number, Bool, Tuple
from myia.unify import var, svar, Var, SVar

from . import ops as P
from .ops import Primitive


SIGNATURES: DictT[Primitive, Function] = dict()


def register_signature(prim: Primitive, s: Function):
    """Register the signature for a primitive.

    Does not accept duplicates.
    """
    assert prim not in SIGNATURES
    SIGNATURES[prim] = s


def isnum(v) -> bool:
    """Check if type is a number."""
    return isinstance(v, Number)


a = var(filter=isnum)
num_binop = Function((a, a), a)

register_signature(P.add, num_binop)
register_signature(P.sub, num_binop)
register_signature(P.mul, num_binop)
register_signature(P.div, num_binop)
register_signature(P.mod, num_binop)
register_signature(P.pow, num_binop)

num_unop = Function((a,), a)
register_signature(P.uadd, num_unop)
register_signature(P.usub, num_unop)

b: Var = var()
cmp_op = Function((b, b), Bool())
register_signature(P.eq, cmp_op)
register_signature(P.lt, cmp_op)
register_signature(P.gt, cmp_op)
register_signature(P.ne, cmp_op)
register_signature(P.le, cmp_op)
register_signature(P.ge, cmp_op)

register_signature(P.not_, Function((b,), Bool()))

sv: SVar = svar()
register_signature(P.cons_tuple, Function((b, Tuple(sv)), Tuple(b, sv)))
register_signature(P.head, Function((Tuple(b, sv),), b))
register_signature(P.tail, Function((Tuple(b, sv),), Tuple(sv)))

register_signature(P.if_, Function((Bool(), Function((), b),
                                    Function((), b)), b))
register_signature(P.return_, Function((b,), b))
