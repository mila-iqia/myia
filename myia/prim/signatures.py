"""Primitives signatures for the type inference."""
from myia.dtype import Function, Number, Bool, Tuple, Int, UInt, List
from myia.unify import var, uvar, svar, Var, SVar, Unification, \
    UnificationError
from myia.utils import Registry

from . import ops as P
from .ops import Primitive


SIGNATURES: Registry[Primitive, Function] = Registry()


class MetaVar:
    def infer(self, args, frame, ga):
        raise NotImplementedError("infer")


def register_signature(prim: Primitive, s: Function):
    """Register the signature for a primitive.

    Does not accept duplicates.
    """
    SIGNATURES.register(prim)(s)


U = Unification()


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

intv = var(filter=(Int(8), Int(16), Int(32), Int(64),
                   UInt(8), UInt(16), UInt(32), UInt(64)))


class GetitemVar(MetaVar):
    def infer(self, args, frame, equiv):
        from myia.analyze.value import NO_VALUE

        assert len(args) == 2
        x_t = frame.types[args[0]]
        i_t = frame.types[args[1]]
        r_t = var()

        try:
            U.unify_raw(i_t, U.clone(intv), equiv)
        except UnificationError:
            raise TypeError("index is not an integer")

        i_v = frame.values[args[1]]
        if i_v is not NO_VALUE:
            if i_v < 0:
                raise ValueError("getitem with negative value")
            tts = tuple(var() for _ in range(i_v)) + (r_t, svar())
            uv = uvar((List(r_t), Tuple(tts)))
            equiv = U.unify(uv, x_t, equiv)
        else:
            equiv = U.unify(List(r_t), x_t, equiv)

        if equiv is None:
            raise TypeError("Mismatched getitem")

        return U.reify(r_t, equiv)


register_signature(P.getitem, GetitemVar())

register_signature(P.if_, Function((Bool(), Function((), b),
                                    Function((), b)), b))
register_signature(P.return_, Function((b,), b))
