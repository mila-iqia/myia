"""Implementations of the primitives' gradients.

Each primitive is associated to an augmented function, which returns a pair of
the (augmented) original primitive's output and a backpropagator function.
"""


from types import FunctionType
from myia.utils import Registry
from myia.api import parse
from myia.info import NamedDebugInfo, About
from myia.anf_ir import Graph
from myia import primops
from myia.primops import Primitive
from myia.anf_ir_utils import replace
from myia.py_implementations import \
    Jinv, J, zeros_like, cons_tuple, head, tail, setitem


def bprop_to_augm(prim: Primitive, fn: FunctionType) -> Graph:
    """Given a function for the bprop, make the augmented function."""
    info = NamedDebugInfo(prim=prim, name=prim.name)

    bprop = parse(fn)
    bprop.debug.name = None
    bprop.debug.about = About(info, 'grad_bprop')
    bprop.output = bprop.apply(primops.cons_tuple, (), bprop.output)

    *args, dout = bprop.parameters

    with About(info, 'grad_fw'):
        outer = Graph()
        outer.transforms['primal'] = prim

    transf_args = []
    for p in args:
        with About(p.debug, 'grad_fw'):
            outer_p = outer.add_parameter()
        replace(p, outer_p)
        transf_args.append(outer.apply(primops.Jinv, outer_p))

    with About(dout.debug, 'grad_bw'):
        new_dout = bprop.add_parameter()
        replace(dout, new_dout)
        # We remove all parameters except new_dout
        bprop.parameters = [new_dout]

    result = outer.apply(primops.J, outer.apply(prim, *transf_args))
    outer.output = outer.apply(
        primops.cons_tuple,
        result,
        outer.apply(primops.cons_tuple, bprop, ())
    )
    return outer


augmented_graphs: Registry[primops.Primitive, Graph] = Registry()
register = augmented_graphs.register


def register_bprop(prim):
    """Register an augmented function for prim, given a backpropagator."""
    def deco(fn):
        fn2 = bprop_to_augm(prim, fn)
        return register(prim)(fn2)
    return deco


def register_augm(prim):
    """Register an augmented function for prim."""
    def deco(fn):
        fn2 = parse(fn)
        return register(prim)(fn2)
    return deco


@register_bprop(primops.add)
def bprop_add(x, y, dz):
    """Backpropagator for primitive `add`."""
    return (dz, dz)


@register_bprop(primops.sub)
def bprop_sub(x, y, dz):
    """Backpropagator for primitive `sub`."""
    return (dz, -dz)


@register_bprop(primops.mul)
def bprop_mul(x, y, dz):
    """Backpropagator for primitive `mul`."""
    return (dz * y, dz * x)


@register_bprop(primops.div)
def bprop_div(x, y, dz):
    """Backpropagator for primitive `div`."""
    return (dz / y, -dz * x / (y * y))


@register_bprop(primops.uadd)
def bprop_uadd(x, dz):
    """Backpropagator for primitive `uadd`."""
    return (dz,)


@register_bprop(primops.usub)
def bprop_usub(x, dz):
    """Backpropagator for primitive `usub`."""
    return (-dz,)


@register_bprop(primops.gt)
def bprop_gt(x, y, dz):
    """Backpropagator for primitive `gt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.lt)
def bprop_lt(x, y, dz):
    """Backpropagator for primitive `lt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.cons_tuple)
def bprop_cons_tuple(_head, _tail, dz):
    """Backpropagator for primitive `cons_tuple`."""
    return (head(dz), tail(dz))


@register_bprop(primops.head)
def bprop_head(tup, dz):
    """Backpropagator for primitive `head`."""
    return (cons_tuple(dz, zeros_like(tail(tup))),)


@register_bprop(primops.tail)
def bprop_tail(tup, dz):
    """Backpropagator for primitive `tail`."""
    return (cons_tuple(zeros_like(head(tup)), dz),)


@register_bprop(primops.getitem)
def bprop_getitem(data, idx, dz):
    """Backpropagator for primitive `getitem`."""
    return (setitem(zeros_like(data), idx, dz), zeros_like(idx))


@register_bprop(primops.J)
def bprop_J(x, dz):
    """Backpropagator for primitive `J`."""
    return (Jinv(dz),)


@register_bprop(primops.Jinv)
def bprop_Jinv(x, dz):
    """Backpropagator for primitive `Jinv`."""
    return (J(dz),)


@register_bprop(primops.zeros_like)
def bprop_zeros_like(x, dz):
    """Backpropagator for primitive `zeros_like`."""
    return (zeros_like(x),)


@register_augm(primops.if_)
def bprop_if_(c, tb, fb):
    """Backpropagator for primitive `if`."""
    zeros_like  # Currently required for parser to see it in bprop()

    if Jinv(c):
        rval, branch_bprop = tb()
    else:
        rval, branch_bprop = fb()

    def bprop(dout):
        if Jinv(c):
            return (), zeros_like(c), branch_bprop(dout)[0], zeros_like(fb)
        else:
            return (), zeros_like(c), zeros_like(tb), branch_bprop(dout)[0]

    return rval, bprop
