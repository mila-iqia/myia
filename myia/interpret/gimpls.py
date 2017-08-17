
from typing import List, Any
from .runtime import bprop_impl


###########################################
# Gradients of primitives needed for Grad #
###########################################


@bprop_impl
def bprop_zeros_like(x, d):
    return GRAD(zeros_like(x))


@bprop_impl
def bprop_mapadd(x, y, d):
    # TODO: correct when x is ZERO (its shape can be different from y)?
    # Probably unneeded?
    return GRAD(d, d)


# @bprop_impl
# def bprop_J(x, d):
#     return GRAD(Jinv(d))


# @bprop_impl
# def bprop_Jinv(x, d):
#     return GRAD(J(d))


######################################
# Gradients of arithmetic primitives #
######################################


@bprop_impl
def bprop_add(x, y, dz):
    return GRAD(dz, dz)


@bprop_impl
def bprop_subtract(x, y, dz):
    return GRAD(dz, -dz)


@bprop_impl
def bprop_multiply(x, y, dz):
    return GRAD(dz * y, dz * x)


@bprop_impl
def bprop_divide(x, y, dz):
    return GRAD(dz / y, -dz * x / (y * y))


@bprop_impl
def bprop_unary_subtract(x, dz):
    return GRAD(-dz)


###################################################
# Gradients of boolean and conditional primitives #
###################################################


@bprop_impl
def bprop_equal(x, y, dz):
    return GRAD(False, False)


@bprop_impl
def bprop_greater(x, y, dz):
    return GRAD(False, False)


@bprop_impl
def bprop_less(x, y, dz):
    return GRAD(False, False)


@bprop_impl
def bprop_switch(c, t, f, dz):
    # There's a subtlety here, which is that we must return
    # appropriately-sized gradients for each argument. This
    # requires the use of zeros_like to match input shapes.
    # TODO: zeros_like shouldn't be needed for the condition
    # if it is always boolean (as it should be).
    if c:
        return GRAD(
            zeros_like(Jinv(c)),  # False
            dz,
            zeros_like(Jinv(f))
        )
    else:
        return GRAD(
            zeros_like(Jinv(c)),  # False
            zeros_like(Jinv(t)),
            dz
        )


@bprop_impl
def bprop_identity(v, dz):
    return GRAD(dz)


#################################
# Gradients of other primitives #
#################################


@bprop_impl
def bprop_index(tup, idx, dz):
    def f(pair):
        return switch(pair[0] == idx, dz,
                      zeros_like(Jinv(pair[1])))
    rval = map(f, enumerate(tup))
    return GRAD(rval, 0)


@bprop_impl
def bprop_len(xs, dz):
    return GRAD(zeros_like(Jinv(xs)))


@bprop_impl
def bprop_range(n, dz):
    return GRAD(0)


@bprop_impl
def bprop_map(f, xs, dz):
    # I... think that's right?
    # TODO: test it
    results = map(f, xs)
    bprops = map(second, results)
    # TODO: THIS IS WRONG, SHOULD BE SOMETHING LIKE THIS:
    # d = map(lambda xy: xy[0](xy[1]), zip(bprops, dz))
    # but we don't have zip yet.
    d = map(bprops[0], dz)
    df = reduce(mapadd, map(first, d))
    dxs = map(second, d)
    return GRAD(df, dxs)


@bprop_impl
def bprop_enumerate(xs, dz):
    return GRAD(map(second, dz))


__all__: List[Any] = []
