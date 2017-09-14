

from .main import symbol_associator, impl_bank, GlobalEnv
from ..inference.avm import Fork, WrappedException, \
    AbstractValue, unwrap_abstract
from .flow_all import ANY, VALUE, ERROR
from ..inference.types import typeof, type_map, Tuple
from ..interpret import PrimitiveImpl, ClosureImpl
from itertools import product


_ = True


abstract_globals = GlobalEnv(impl_bank['abstract'])


##########################
# Implementation helpers #
##########################


def complete_switches(args, sw):
    if sw is True or sw is False:
        return [sw for _ in args]
    else:
        return sw + [True for _ in args[len(sw):]]


@symbol_associator("abstract")
def impl_abstract(sym, name, fn):
    prim = PrimitiveImpl(fn, sym)
    impl_bank['abstract'][sym] = prim
    return prim


def aimpl_factory(unwrap_args=True,
                  any_to_any=None,
                  var_to_any=None,
                  wrap_result=True):
    if any_to_any is None:
        any_to_any = unwrap_args
    if var_to_any is None:
        var_to_any = unwrap_args

    def deco(fn):
        def newfn(*args):
            # TODO: do the complete_switches when deco() is called.
            c_any_to_any = complete_switches(args, any_to_any)
            c_var_to_any = complete_switches(args, var_to_any)
            opts = []

            max_depth = max(arg.depth if isinstance(arg, AbstractValue)
                            else 0 for arg in args)

            errors = tuple(arg[ERROR] for arg in args
                           if isinstance(arg, AbstractValue) and
                           ERROR in arg.values)

            if errors:
                if len(errors) == 1:
                    errors, = errors
                return AbstractValue({ERROR: errors})

            unwrapped = [unwrap_abstract(arg) for arg in args]

            if unwrapped is None:
                pass
            elif any((a2a and a is ANY)
                     for (a, a2a) in zip(unwrapped, c_any_to_any)):
                result = ANY
            # elif any((v2a and isvar(v))
            #          for (v, v2a) in zip(unwrapped, c_var_to_any)):
            #     result = ANY
            else:
                result = fn(*unwrapped)
                # try:
                #     result = fn(*unwrapped)
                # except WrappedException as exc:
                #     result = AbstractValue({ERROR: exc.error})

            if wrap_result:
                result = AbstractValue(result, max_depth + 1)

            return result

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
    try:
        return xs.shape
    except AttributeError as e:
        raise WrappedException(e)


@std_aimpl
def abstract_greater(x, y):
    return x > y


@std_aimpl
def abstract_less(x, y):
    return x < y


@std_aimpl
def abstract_type(x):
    t = type(x)
    res = type_map.get(type(x), None)
    if res:
        return res
    if isinstance(x, tuple):
        return Tuple[tuple(map(abstract_type, x))]
    if t.__name__ == 'ndarray' and hasattr(x, 'dtype'):
        return Array[type_map.get(x.dtype.name)]
    raise WrappedException(TypeError(f'Unknown data type: {type(x)}'))


@std_aimpl
def abstract_raise_exception(x):
    raise WrappedException(x)


@std_aimpl
def abstract_Exception(x):
    return Exception(x)


@std_aimpl
def abstract_print(x):
    print(x)


@std_aimpl
def abstract_equal(x, y):
    return x == y


@impl_abstract
def abstract_identity(x):
    return x


@impl_abstract
def abstract_mktuple(*args):
    return tuple(args)


@aimpl_factory(unwrap_args=[True],
               var_to_any=False,
               any_to_any=False,
               wrap_result=False)
def abstract_switch(cond, t, f):
    if isinstance(cond, bool):
        return t if cond else f
    elif cond is ANY:  # or isvar(cond):
        return Fork([t, f])
    else:
        raise TypeError(f'Cannot switch on {cond}')
