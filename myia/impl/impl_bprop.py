
from typing import List, Any, Union, Callable, Dict
from .main import symbol_associator, impl_bank
from ..stx import Symbol, ApplyNode as Apply, ClosureNode, \
    LambdaNode, TupleNode, GenSym, BPROP, JTAG, create_lambda, \
    globals_pool
from ..symbols import inst_builtin
from ..parse import parse_function
from .impl_interp import zeros_like, J, Jinv, switch, first, second, \
    Closure, closure_fn, closure_args, reduce, add, exp, log, transpose, \
    broadcast, shape, fit, setslice
from ..transform.grad import ggen, grad_computers
from ..lib import Primitive


_ = True


def macro_grad_for(sym, nargs_closure):
    """
    This generates a ``GRAD`` function for use in primitive
    gradients. ``GRAD`` is parameterized by the number of
    arguments that are closed on.

    See the *Partial Application* section above for a detailed
    explanation of what this is for.
    """
    def macro_grad(*args):
        # *args = (a, b, c, ...) =>
        #   ((), a, b, c, ...)  # If nargs_closure == 0
        #   ((a,), b, c, ...)  # If nargs_closure == 1
        #   ((a, b), c, ...)  # If nargs_closure == 2
        #   etc.
        return TupleNode([ClosureNode(sym, args[:nargs_closure]),
                          *args[nargs_closure:]])
    return macro_grad


@symbol_associator('bprop')
def impl_bprop(sym, name, orig_fn: Callable) -> Callable:
    """
    Decorator to declare a backpropagator function. For instance,
    to define the backpropagator for operator builtins.whatever,
    write:

        @impl_bprop
        def bprop_whatever(x, y, ..., gwhatever):
            return GRAD(gx, gy, ...)

    It is important to return ``GRAD(...)``. The ``GRAD`` macro
    will group the gradients correctly given how many arguments
    are part of a partial application.

    Refer to the previous section on Partial Application.
    """

    # This is the implementation for the forward pass
    root_globals = impl_bank['interp']
    prim = root_globals[sym]
    assert isinstance(prim, Primitive)

    def mkgrad(nargs_closure: int) -> LambdaNode:
        # Copy symbol to grad namespace
        bprop_sym = Symbol(sym,
                           version=nargs_closure + 1,
                           namespace='global::builtin_bprop',
                           relation=BPROP)

        # We compile the backpropagator using Myia. We provide
        # the GRAD macro which will account for nargs_closure
        # stored arguments.
        lbda = parse_function(
            orig_fn,
            macros={'GRAD': macro_grad_for(sym, nargs_closure)}
        )

        # Now we generate a combined function that returns the
        # result of the forward pass along with a backpropagator
        # function.
        G = GenSym()
        args = [G.sym(a) for a in prim.argnames]
        forward = Apply(inst_builtin.J,
                        Apply(sym, *[Apply(inst_builtin.Jinv, a)
                                     for a in args]))
        backward = ClosureNode(bprop_sym, args)

        # Final function:
        augm_sym = ggen(sym, JTAG)
        ast = create_lambda(augm_sym, args, TupleNode([forward, backward]), G)
        ast.primal = sym
        globals_pool.associate(bprop_sym, lbda)
        return ast

    grad_computers[sym] = mkgrad

    return orig_fn


###########################################
# Gradients of primitives needed for Grad #
###########################################


@impl_bprop
def bprop_zeros_like(x, d):
    return GRAD(zeros_like(x))


@impl_bprop
def bprop_J(x, d):
    return GRAD(Jinv(d))


@impl_bprop
def bprop_Jinv(x, d):
    return GRAD(J(d))


######################################
# Gradients of arithmetic primitives #
######################################


@impl_bprop
def bprop_add(x, y, dz):
    # TODO: correct when x is ZERO (its shape can be different from y)?
    # Probably unneeded?
    return GRAD(dz, dz)


@impl_bprop
def bprop_subtract(x, y, dz):
    return GRAD(dz, -dz)


@impl_bprop
def bprop_multiply(x, y, dz):
    return GRAD(dz * y, dz * x)


@impl_bprop
def bprop_divide(x, y, dz):
    return GRAD(dz / y, -dz * x / (y * y))


@impl_bprop
def bprop_power(x, y, dz):
    # Note: this will often give a warning because the second element
    # in the pair is ill-defined when x < 0 and it is calculated even
    # if we do not care about it (we usually want the derivative wrt x,
    # not wrt y). An optimization pass will need to prune it out.
    return GRAD(dz * y * x ** (y - 1),
                dz * log(x) * x ** y)


@impl_bprop
def bprop_unary_subtract(x, dz):
    return GRAD(-dz)


@impl_bprop
def bprop_dot(x, y, dz):
    return GRAD(dz @ transpose(y), transpose(x) @ dz)


@impl_bprop
def bprop_transpose(x, dz):
    return GRAD(transpose(dz))


@impl_bprop
def bprop_exp(x, dz):
    return GRAD(dz * exp(x))


@impl_bprop
def bprop_log(x, dz):
    return GRAD(dz / x)


@impl_bprop
def bprop_sum(xs, dz):
    _, bdz = broadcast((xs, dz))
    return GRAD(bdz)
    # return GRAD(zeros_like(xs) + dz)


@impl_bprop
def bprop_shape(arr, dz):
    # TODO: We should mark this kind of gradient somehow to indicate that
    # it's nonsensical.
    return GRAD(zeros_like(arr))


@impl_bprop
def bprop_fit(arr, shp, dz):
    return GRAD(fit(dz, shape(arr)),
                zeros_like(shp))


###################################################
# Gradients of boolean and conditional primitives #
###################################################


@impl_bprop
def bprop_equal(x, y, dz):
    return GRAD(False, False)


@impl_bprop
def bprop_greater(x, y, dz):
    return GRAD(False, False)


@impl_bprop
def bprop_less(x, y, dz):
    return GRAD(False, False)


@impl_bprop
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


@impl_bprop
def bprop_identity(v, dz):
    return GRAD(dz)


#################################
# Gradients of other primitives #
#################################


@impl_bprop
def bprop_Closure(fn, args, dz):
    return GRAD(closure_fn(dz), closure_args(dz))


@impl_bprop
def bprop_closure_fn(clos, dz):
    return GRAD(Closure(dz, Jinv(zeros_like(closure_args(clos)))))


@impl_bprop
def bprop_closure_args(clos, dz):
    return GRAD(Closure(Jinv(closure_fn(clos)), dz))


@impl_bprop
def bprop_index(tup, idx, dz):
    rval = zeros_like(tup)
    rval[idx] = dz
    return GRAD(rval, 0)


@impl_bprop
def bprop_setslice(tup, idx, value, dz):
    dtup = setslice(dz, idx, zeros_like(tup[idx]))
    return GRAD(dtup, 0, dz[idx])


@impl_bprop
def bprop_len(xs, dz):
    return GRAD(zeros_like(Jinv(xs)))


@impl_bprop
def bprop_range(n, dz):
    return GRAD(0)


@impl_bprop
def bprop_map(f, xs, dz):
    # I... think that's right?
    # TODO: test it
    results = map(f, xs)
    bprops = map(second, results)
    # TODO: THIS IS WRONG, SHOULD BE SOMETHING LIKE THIS:
    # d = map(lambda xy: xy[0](xy[1]), zip(bprops, dz))
    # but we don't have zip yet.
    d = map(bprops[0], dz)
    df = reduce(add, map(first, d))
    dxs = map(second, d)
    return GRAD(df, dxs)


@impl_bprop
def bprop_enumerate(xs, dz):
    return GRAD(map(second, dz))


@impl_bprop
def bprop_getattr(rec, field, dz):
    rval = zeros_like(Jinv(rec))
    rval = setattr(rval, field, dz)
    return GRAD(rval, 0)


@impl_bprop
def bprop_setattr(rec, field, value, dz):
    drec = setattr(dz, field, zeros_like(getattr(rec, field)))
    return GRAD(drec, 0, getattr(dz, field))
