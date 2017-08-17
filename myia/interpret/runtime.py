
from typing import Callable, List, Dict, Any, Union
import inspect
from ..util import HReprBase
from ..stx import Lambda, Symbol, Tuple, Apply, Closure, \
    GenSym, BPROP, JTAG
from ..symbols import builtins
from ..front import parse_function, get_global_parse_env


EnvT = Dict[Symbol, Any]


###########
# Globals #
###########


class BuiltinCollection:
    """
    Implements a "module" of sorts. It has no methods,
    only fields that are populated by ``impl``.
    """
    pass


# Myia's global variables. Used for evaluation.
root_globals: Dict[Any, Any] = {
    builtins.myia_builtins: BuiltinCollection()
}


##########################
# Implementation helpers #
##########################


def impl(fn):
    """
    Define the implementation for the given symbol.
    The implementation will be set in ``root_globals``
    and in the ``myia_builtins`` global.
    """
    assert fn.__name__.startswith('impl_')
    fname = fn.__name__[5:]
    assert hasattr(builtins, fname)
    sym = getattr(builtins, fname)
    prim = PrimitiveImpl(fn)
    root_globals[sym] = prim
    setattr(root_globals[builtins.myia_builtins],
            fname,
            prim)
    return prim


# def myia_impl(sym):
#     # Implement a symbol by parsing it through Myia.
#     # Unused at the moment.
#     def decorator(orig_fn):
#         r, genv = parse_function(orig_fn)
#         fn = evaluate(r, genv)
#         root_globals[sym] = fn
#         setattr(root_globals[builtins.myia_builtins],
#                 fn.__name__.lstrip('_'),
#                 fn)
#         return fn
#     return decorator


###################
# Special objects #
###################


class PrimitiveImpl(HReprBase):
    """
    Wrapper around a pure Python implementation of a function.
    """
    def __init__(self, fn: Callable, name: str = None) -> None:
        argn = inspect.getargs(fn.__code__).args  # type: ignore
        self.argnames: List[str] = argn
        self.nargs = len(self.argnames)
        self.fn = fn
        self.name = name or fn.__name__
        self.grad: Callable[[int], FunctionImpl] = None

    def __call__(self, *args):
        return self.fn(*args)

    def __str__(self):
        return f'Prim({self.name or self.fn})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['PrimitiveImpl'](
            H.div['class_title']('Primitive'),
            H.div['class_contents'](self.name or hrepr(self.fn))
        )


class FunctionImpl(HReprBase):
    """
    Represents a Myia-transformed function.
    """
    def __init__(self, ast: Lambda, envs: List[EnvT]) -> None:
        assert isinstance(ast, Lambda)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.ast = ast
        self.code = VMCode(ast.body)
        self.envs = envs
        self.primal_sym = ast.primal
        self.grad: Callable[[int], FunctionImpl] = None

    def debug(self, args, debugger):
        ast = self.ast
        assert len(args) == len(ast.args)
        return run_vm(self.code,
                      {s: arg for s, arg in zip(ast.args, args)},
                      *self.envs,
                      debugger=debugger)

    def __call__(self, *args):
        return self.debug(args, None)

    def __str__(self):
        return f'Func({self.ast.ref or self.ast})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['FunctionImpl'](
            H.div['class_title']('Function'),
            H.div['class_contents'](hrepr(self.ast.ref or self.ast))
        )


class ClosureImpl(HReprBase):
    """
    Associates a PrimitiveImpl or a FunctionImpl to a number
    of arguments in order to create a partial application.
    """
    def __init__(self,
                 fn: Union[PrimitiveImpl, FunctionImpl],
                 args: List[Any]) -> None:
        self.argnames = [a for a in fn.argnames[len(args):]]
        self.nargs = fn.nargs - len(args)
        self.fn = fn
        self.args = args

    def __call__(self, *args):
        return self.fn(*self.args, *args)

    def __str__(self):
        return f'Clos({self.fn}, {self.args})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['ClosureImpl'](
            H.div['class_title']('Closure'),
            H.div['class_contents'](
                hrepr(self.fn),
                hrepr(self.args)
            )
        )


############################
# Gradient-related methods #
############################


def macro_grad_for(nargs_closure):
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
        return Tuple([Tuple(args[:nargs_closure]),
                     *args[nargs_closure:]])
    return macro_grad


def bprop_impl(orig_fn: Callable) -> Callable:
    """
    Decorator to declare a backpropagator function. For instance,
    to define the backpropagator for operator builtins.whatever,
    write:

        @bprop_impl
        def bprop_whatever(x, y, ..., gwhatever):
            return GRAD(gx, gy, ...)

    It is important to return ``GRAD(...)``. The ``GRAD`` macro
    will group the gradients correctly given how many arguments
    are part of a partial application.

    Refer to the previous section on Partial Application.
    """
    assert orig_fn.__name__.startswith('bprop_')
    fname = orig_fn.__name__[6:]

    sym = getattr(builtins, fname)

    # This is the implementation for the forward pass
    prim = root_globals[sym]
    assert isinstance(prim, PrimitiveImpl)

    # We will cache the result for each nargs_closure value, to
    # avoid needless recomputation.
    _cache: Dict[int, FunctionImpl] = {}

    # Wrap the primitive and a closure-converted backpropagator
    # in a combined method that follows the protocol. Essentially:
    # (forward_fn, bprop_fn) ==>
    #   lambda *args: (forward_fn(*args),
    #                  lambda grad_out: bprop_fn(*args, grad_out))
    def mkgrad(nargs_closure: int) -> FunctionImpl:
        # Check if we have compiled this before
        cached = _cache.get(nargs_closure, None)
        if cached:
            return cached

        # Copy symbol to grad namespace
        rsym = Symbol(sym,
                      version=nargs_closure + 1,
                      namespace='builtin',
                      relation=BPROP)

        # We compile the backpropagator using Myia. We provide
        # the GRAD macro which will account for nargs_closure
        # stored arguments.
        r, genv = parse_function(
            orig_fn,
            macros={'GRAD': macro_grad_for(nargs_closure)}
        )

        # Create a FunctionImpl.
        fn = evaluate(r, genv)

        # Now we generate a combined function that returns the
        # result of the forward pass along with a backpropagator
        # function.
        G = GenSym()
        args = [G.sym(a) for a in prim.argnames]
        forward = Apply(builtins.J,
                        Apply(sym, *[Apply(builtins.Jinv, a)
                                     for a in args]))
        backward = Closure(rsym, args)
        # Final function:
        ast = Lambda(args, Tuple([forward, backward]), G)

        # Boilerplate stuff that should be properly abstracted
        # somewhere else.
        ast.global_env = get_global_parse_env('__root__')
        impl_sym = ast.global_env.gen(sym, JTAG)
        ast.global_env[impl_sym] = ast
        ast.ref = impl_sym
        ast.primal = sym
        impl = FunctionImpl(ast, [root_globals])
        root_globals[impl_sym] = impl
        root_globals[rsym] = fn

        # Let's not forget to save our work in the cache!
        _cache[nargs_closure] = impl
        return impl

    # prim's gradient is to be compiled lazily using
    # prim.grad(nclos_args)
    prim.grad = mkgrad

    return orig_fn


from .vm import VMCode, run_vm, evaluate
