
from typing import Callable, List, Dict, Any, Union
import inspect
from ..util import HReprBase
from ..stx import Lambda, Symbol
from ..symbols import builtins


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

from .vm import VMCode, run_vm
