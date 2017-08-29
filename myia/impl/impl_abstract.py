

from .main import symbol_associator, impl_bank
from ..inference.infer import \
    PrimitiveAImpl, VALUE, UNIFY, unify, AbstractValue, Not, Union
from ..inference.types import typeof


_ = True


##########################
# Implementation helpers #
##########################


def args_product(args):
    opts = [arg.opts if isinstance(arg, Union) else [arg]
            for arg in args]
    return product(*opts)


@symbol_associator("abstract")
def impl_abstract(sym, name, fn):
    prim = PrimitiveAImpl(fn, sym)
    impl_bank['abstract'][sym] = prim
    return prim


def aimpl_factory(unwrap_union=True,
                  any_to_any=True,
                  var_to_any=True):
    def deco(fn):
        def newfn(*_args):
            opts = []
            for args in args_product(_args):
                unwrapped = [unwrap_abstract(arg) for arg in args]
                u = merge(*[arg[UNIFY] for arg in args])
                if u is False:
                    raise Unsatisfiable()

                if any_to_any and any(a is ANY for a in unwrapped):
                    result = ANY
                elif var_to_any and any(isvar(a) for a in unwrapped):
                    result = ANY
                else:
                    result = fn(*unwrapped)
                opts.append(AbstractValue(result))
            return Union(opts)

        newfn.__name__ = fn.__name__
        return impl_abstract(newfn)

    return deco


def std_aimpl(fn):
    return aimpl_factory()(fn)


############################
# Abstract implementations #
############################


@std_aimpl
def abstract_add(x, y):
    return x + y


@std_aimpl
def abstract_subtract(x, y):
    return x - y


@std_aimpl
def abstract_dot(x, y):
    return x @ y


@std_aimpl
def abstract_index(xs, idx):
    return xs[idx]


@std_aimpl
def abstract_shape(xs):
    return xs.shape


@std_aimpl
def abstract_greater(x, y):
    return x > y


@std_aimpl
def abstract_type(x):
    return typeof(x)


# @std_aimpl
# def abstract_assert_true(v, message):
#     if message:
#         assert v, message
#     else:
#         assert v


# @std_aimpl
# def abstract_equal(x, y):
#     return x == y


@impl_abstract
def abstract_assert_true(v, message):
    # print('One', v)
    vv = v[VALUE]
    # assert vv, message
    # print('Two', v)
    if vv:
        return AbstractValue({VALUE: True,
                              UNIFY: v[UNIFY]})
    else:
        return AbstractValue({ERROR: AssertionError(message),
                              UNIFY: v[UNIFY]})


@impl_abstract
def abstract_equal(x, y):
    vx = x[VALUE]
    vy = y[VALUE]
    d = unify(vx, vy)
    # print('Unify', vx, vy, d)
    if not d:
        return AbstractValue(False)
    elif d == {}:
        return AbstractValue(True)
    else:
        t = AbstractValue({VALUE: True, UNIFY: d})
        f = AbstractValue({VALUE: False, UNIFY: Not(d)})
        return Union([t, f])


@impl_abstract
def abstract_identity(x):
    # return x[VALUE]
    return x


@impl_abstract
def abstract_switch(cond, t, f):
    cond = cond[VALUE]
    if isinstance(cond, bool):
        # return t[VALUE] if cond else f[VALUE]
        return t if cond else f
    elif isinstance(cond, Var):
        return Union([t, f])
    else:
        raise TypeError(f'Cannot switch on {cond}')
