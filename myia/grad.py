"""
Defines a program transformation that can map any function to a
new function that can compute its own back-propagated gradient
while reusing intermediate results. This is the transformation
originally formulated in [1].

## Gradient functions

For each function transformed by Grad, several auxiliary functions
will be created. These are the ones you need to know about:

* ``↑f`` is the "tagged" version of ``f``. In [1] this is ``f``
  with a top-left harpoon, i.e. f⃐. ``↑f`` returns two values:
  first, it returns the "tagged" version of ``f``'s normal
  output. Second, it returns the backpropagator closure ``♢f``
  (see below).

* ``♢f`` is the backpropagator for ``f``. In [1] this is ``f``
  with a bar on top, i.e. f̄. Its input is the
  gradient with respect to its output (let's call it ``∇f``).
  Its output is the tuple ``(closure_grads, *argument_grads)``.
  The latter part is what typically interests us, but the former
  part is required to make everything run smoothly (see Partial
  Application section below).

  Note that ``♢f`` is not a top-level function, but a closure over
  the real top-level function ``♢*f`` (see below).

* ``♢*f`` is the closure-converted backpropagator function for ``f``.
  Being closure-converted means it has no free variables, hence the
  ``*``. This is an implementation detail.

## Gradient variables

In the generated code you will see the following variables:

* ``↑x`` is the "tagged" version of ``x``. If ``x`` is a scalar,
  that's the identity, if ``x`` is a data structure this applies
  the tag on every member, and if ``x`` is a function, see above.

* ``♢x`` is the "backpropagator" for ``x``.

* ``∇x`` is the "sensitivity" with respect to ``x``. In other
  words, this is where we accumulate the gradient for ``x``.

In a nutshell, when there is an assignment like ``x = f(y, z)``
in the code:

* In the forward pass, we generate:
      ↑x, ♢x = ↑f(↑y, ↑z)   # forward computation

* In the backward pass, we generate:
      ∇x = zeros_like(x)    # initialization
      ...
      ∇f, ∇y, ∇z += ♢x(∇x)  # propagation

Note that the statements for the propagation are generated
in reverse order, so if ``x`` is an input to other function
calls later in the code, the gradient will accumulate into
``∇x`` *before* it is used to accumulate into ``∇f, ∇y, ∇z``,
starting from ``∇out``, the input to the backpropagator
function. That's why it's called *back*prop.

## Example

In a nutshell, supposing we have the following function:

    z = 10  # free variable
    def f(x, y):
        a = g(x, y, z)
        b = h(a, x)
        return b

Then we will get something like this:

    ↑z = 10  # free variable

    def ♢*f(♢a, ♢b, ∇b):
        # The zeros should be the same "shape" as
        # g, x, y, ...
        zero_init(∇g, ∇x, ∇y, ∇z, ∇h, ∇a, ∇h)
        # Backpropagation, operates in reverse order:
        # propagate through h, then through g. Notice:
        # * We have gradient terms for g and h, because they
        #   could be closures or partial applications, and
        #   we must track the contributions.
        # * The left-hand side looks just like the function
        #   application. h(a, x) becomes ∇h, ∇a, ∇x +=
        # * The right-hand side is also very easy to remember,
        #   it's bprop(grad) for each variable you set in the
        #   original function, in reverse order
        ∇h, ∇a, ∇x += ♢b(∇b)
        ∇g, ∇x, ∇y, ∇z += ♢a(∇a)
        # Note that ∇z is stashed in the first return value.
        # Gradients for all of f's free variables must go there.
        return ((∇z,), ∇x, ∇y)

    def ↑f(↑x, ↑y):
        # The "tagged" functions ↑g and ↑h give us both
        # tagged forward results and backpropagators.
        ↑a, ♢a = ↑g(↑x, ↑y, ↑z)
        ↑b, ♢b = ↑h(↑a, ↑x)
        def ♢f(∇f):
            # Closure on ♢*f
            return ♢*f(♢a, ♢b, ∇f)
        # We return the tagged original return value
        # and a backpropagator.
        return ↑b, ♢f

The reality is a bit more complicated, but not by much. Take
note that we transform functions to a-normal form before we
run Grad. In a-normal form, all statements look like
``variable1 = fn(variable2, ...)``. No nested expressions.

We accumulate gradients for functions as well, because they
may be closures. If there is a closure ``f`` in the code,
and ``x`` and ``y`` are its free variables, then we will
simply generate something like this:

    ∇x, ∇y = ∇f

This allows us to recuperate contributions made by calls
to ``f``.

## Partial application

Closures in Myia are compiled to partial applications, and we
allow partial applications to primitives (``while`` generates
a partial application to ``identity``). This creates a subtlety
in the interaction with backpropagators.

The backpropagator for ``f`` should return
``(closure_grads, *argument_grads)``. Now, if ``f`` has no
free variables then ``closure_grads`` is, quite naturally,
the empty tuple ``()``. However, note that this is
the case of all the functions Myia compiles, because the free
variables are prepended to the list of arguments.

When we make a partial application of ``f`` on its first n
arguments, we basically state that these n arguments are "free
variables". Concretely, that means we need the first n elements
of ``argument_grads`` to *move* to the end of ``closure_grads``
in the backpropagator for the partial.

We could do this by taking the return value of a backpropagator
and fudging it appropriately (we did at first), but that creates
a lot of crud in the graph and it's cleaner to do it directly.
What this means is:

* The Grad class takes a ``nargs_closure`` argument stating
  how many arguments at the beginning are free variables.
* Gradients of primitives *also* require an ``nargs_closure``
  parameter, because we can--and do--take partials of them,
  and the same logic must apply. This is implemented using
  the ``GRAD`` "macro", which generates the right return
  value depending on an internal parameter.
* Thus the ``grad`` field of both ``PrimitiveImpl`` and
  ``FunctionImpl`` is actually a function of one argument,
  ``nargs_closure`` (integer). Its output is cached to
  avoid needless recomputation.

[1] B. Pearlmutter, J. Siskind,
    Reverse-Mode AD in a Functional Framework:
    Lambda the Ultimate Backpropagator (2008)
    http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf
"""

from typing import Dict, List, Tuple as TupleT, Any, \
    Union, cast, Optional, Sequence, Iterable, Callable

from .ast import \
    Transformer, GenSym, MyiaASTNode, \
    Symbol, Value, Lambda, Let, Apply, Tuple, Closure
from .interpret import \
    root_globals, impl, evaluate, \
    PrimitiveImpl, FunctionImpl, ClosureImpl
from .front import \
    ParseEnv, parse_function, get_global_parse_env
from .symbols import builtins, bsym, gsym
from copy import copy
from .compile import a_normal
from .util import Props
from .buche import buche
from collections import OrderedDict


LeafType = Union[Symbol, Value]


######################################
# Decorator for gradient definitions #
######################################


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


# def prim_rgrad(sym):
#     # Copy symbol to grad namespace
#     rsym = Symbol(sym, namespace='builtin', relation='♢*')
#     #Symbol(sym.label, namespace='grad:builtin')

#     prim = root_globals[sym]
#     assert isinstance(prim, PrimitiveImpl)

#     def decorator(fn):

#         # Wrap the primitive and a closure-converted backpropagator
#         # in a combined method that follows the protocol
#         G = GenSym()
#         args = [G.sym(a) for a in prim.argnames]
#         forward = Apply(builtins.J,
#                         Apply(sym, *[Apply(builtins.Jinv, a)
#                                      for a in args]))
#         backward = Closure(rsym, args)
#         ast = Lambda(args, Tuple([forward, backward]), G)
#         impl = FunctionImpl(ast, (root_globals,))
#         prim.grad = impl
#         impl.primal = prim

#         root_globals[rsym] = PrimitiveImpl(fn)
#         return impl

#     return decorator


def rgrad(sym: Symbol) -> Callable[..., Any]:
    """
    Decorator to declare a function as the backpropagator for
    the given symbol.

    Usage:

        @rgrad(builtins.whatever)
        def bprop_whatever(x, y, ..., gwhatever):
            return GRAD(gx, gy, ...)

    It is important to return ``GRAD(...)``. The ``GRAD`` macro
    will group the gradients correctly given how many arguments
    are part of a partial application.

    Refer to the previous section on Partial Application.
    """

    assert isinstance(sym, Symbol)

    def decorator(orig_fn: Callable) -> Callable:
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
                          relation='♢*')

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
            impl_sym = ast.global_env.gen(sym, '↑')
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

        return impl

    return decorator


################################################
# Implementation of primitives needed for Grad #
################################################


@impl(builtins.fill)
def fill(x: Any, value: Union[int, float]) -> Any:
    """
    Creates a structure just like ``x`` but where each scalar element
    is set to ``value``.

    If ``x`` is a PrimitiveImpl or a FunctionImpl, this returns
    (). If ``x`` is a ClosureImpl, this returns a filled value
    for each value in the closure.
    """
    if isinstance(x, (int, float)):
        return value
    elif isinstance(x, tuple):
        return tuple(fill(a, value) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return ()
    elif isinstance(x, ClosureImpl):
        return tuple(fill(a, value) for a in x.args)
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot create a {value} conformant with {x}')


@impl(builtins.zeros_like)
def zeros_like(x):
    """
    Creates a structure just like ``x`` but "zeroed out."

    If ``x`` is a PrimitiveImpl or a FunctionImpl, this returns
    (). If ``x`` is a ClosureImpl, this returns a zero
    for each value in the closure.

    >>> zeros_like(17)
    0
    >>> zeros_like((1, 2, (3, 4)))
    (0, 0, (0, 0))
    >>> zeros_like(lambda x, y: x + y)  # (metaphorically)
    ()
    >>> x = 10; zeros_like(lambda y: x + y)  # (metaphorically)
    (0,)

    Implements the "0" operator in Pearlmutter & Siskind.
    """
    # TODO: rename to zeros_like
    return fill(x, 0)


@impl(builtins.ones_like)
def ones_like(x):
    # TODO: rename to ones_like
    return fill(x, 1)


@impl(builtins.mapadd)
def mapadd(x: Any, y: Any) -> Any:
    """
    Element-wise addition.

    >>> mapadd(10, 9)
    19
    >>> mapadd((1, 2, (3, 4)), (4, 3, (2, 1)))
    (5, 5, (5, 5))

    Implements the "⊕" (circled plus) operator in Pearlmutter & Siskind.
    """
    # TODO: this should be add, but add concatenates tuples, whereas
    # this adds their values element-wise.
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x + y
    elif type(x) is not type(y):
        raise TypeError(f'Cannot mapadd {x} and {y} (not same type).')
    elif isinstance(x, tuple):
        assert len(x) == len(y)
        return tuple(mapadd(a, b) for a, b in zip(x, y))
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot mapadd values of type {type(x)}')


def JGrad(x: FunctionImpl) -> Callable[[int], FunctionImpl]:
    """
    Helper function that creates a gradient factory for
    a FunctionImpl.

    See previous section on Partial Application for the
    purpose of the ``nargs_closure`` argument.
    """

    # We cache the compilation results.
    _cache: Dict[int, FunctionImpl] = {}

    def make_grad(nargs_closure: int) -> FunctionImpl:
        gfn = _cache.get(nargs_closure, None)
        if gfn:
            return gfn

        normalized = a_normal(x.ast)
        assert isinstance(normalized, Lambda)
        assert x.ast.ref

        # Generate the gradient expression
        G = Grad(
            name = x.ast.ref,
            primal = normalized,
            nargs_closure = nargs_closure
        )
        g = G.transform()

        # Create a FunctionImpl
        gfn = evaluate(g, G.global_env)

        # Don't forget to cache.
        _cache[nargs_closure] = gfn
        return gfn
    return make_grad


def JX(x: Union[PrimitiveImpl, FunctionImpl],
       nargs_closure: int) -> FunctionImpl:
    """
    Helper function for the gradient of PrimitiveImpl or
    FunctionImpl, given nargs_closure closure arguments.

    See previous section on Partial Application for the
    purpose of the ``nargs_closure`` argument.
    """
    if isinstance(x, PrimitiveImpl):
        # x.grad is set by the rgrad decorator. If it is
        # None, it means no one defined a gradient for that
        # operation.
        assert x.grad is not None
        return x.grad(nargs_closure)
    elif isinstance(x, FunctionImpl):
        if not x.grad:
            x.grad = JGrad(x)
        return x.grad(nargs_closure)
    else:
        raise TypeError(f'JX applied on wrong type: {x}')


@impl(builtins.J)
def J(x: Any) -> Any:
    """
    Return a Grad-transformed version of this data.

    * On scalars, this is the identity function.
    * On a data structure, this applies ``J`` on each element and
      returns a data structure with the same shape.
    * On a function of type ``T -> U``, this returns the
      Grad-transformed function, with signature (more or less)
      ``J(T) -> (J(U), S(U) -> S(T))``. That is to say, it returns
      J-transformed outputs and a backpropagator function that
      takes an output sentisitivity and returns an input
      sensitivity (don't look for an S type operator, I made that
      up (J(T) isn't exactly correct either, since it's not a type
      operator), but for what it's worth, ``zeros_like(x)`` would
      have the signature ``T -> S(T)``).

    Implements the J operator in Pearlmutter & Siskind.
    """
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(J(a) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return JX(x, 0)
    elif isinstance(x, ClosureImpl):
        c = ClosureImpl(JX(x.fn, len(x.args)),
                        J(tuple(x.args)))
        return c
    elif x is None:
        return None
    else:
        raise TypeError(f'Invalid argument for J: {x}')


@impl(builtins.Jinv)
def Jinv(x: Any) -> Any:
    """
    Undo the effect of ``J``.

    * On scalars, this is the identity function.
    * On a data structure, this applies ``Jinv`` on each element and
      returns a data structure with the same shape.
    * On a function, this undoes the effect of ``J``. This should
      *never* be applied on a function that was not the result of
      transforming through ``J``.

    Implements the J^{-1} operator in Pearlmutter & Siskind.
    """
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(Jinv(a) for a in x)
    elif isinstance(x, PrimitiveImpl):
        raise Exception('Primitives have no primals.')
    elif isinstance(x, FunctionImpl):
        assert x.primal_sym is not None
        if isinstance(x.primal_sym, Symbol):
            primal = evaluate(x.primal_sym, x.ast.global_env)
        else:
            primal = x.primal_sym
        if not isinstance(primal, (FunctionImpl, PrimitiveImpl)):
            raise Exception('Should be FunctionImpl, but found:'
                            f' {primal}, type {type(primal)},'
                            f' for {x.primal_sym}')
        return primal
    elif isinstance(x, ClosureImpl):
        c = ClosureImpl(Jinv(x.fn), Jinv(tuple(x.args)))
        return c
    elif x is None:
        return x
    else:
        raise TypeError(f'Invalid argument for Jinv: {x}')


###########################################
# Gradients of primitives needed for Grad #
###########################################


myia_builtins = Props(globals())


@rgrad(builtins.zeros_like)
def gzeros_like(x, d):
    return GRAD(zeros_like(x))


@rgrad(builtins.mapadd)
def gmapadd(x, y, d):
    return GRAD(d, d)


@rgrad(builtins.J)
def gJ(x, d):
    return GRAD(Jinv(d))


@rgrad(builtins.Jinv)
def gJinv(x, d):
    return GRAD(J(d))


######################################
# Gradients of arithmetic primitives #
######################################


@rgrad(builtins.add)
def gadd(x, y, dz):
    return GRAD(dz, dz)


@rgrad(builtins.subtract)
def gsubtract(x, y, dz):
    return GRAD(dz, -dz)


@rgrad(builtins.multiply)
def gmultiply(x, y, dz):
    return GRAD(dz * y, dz * x)


@rgrad(builtins.divide)
def gdivide(x, y, dz):
    return GRAD(dz / y, -dz * x / (y * y))


@rgrad(builtins.unary_subtract)
def gunary_subtract(x, dz):
    return GRAD(-dz)


###################################################
# Gradients of boolean and conditional primitives #
###################################################


@rgrad(builtins.equal)
def gequal(x, y, dz):
    return GRAD(False, False)


@rgrad(builtins.greater)
def ggreater(x, y, dz):
    return GRAD(False, False)


@rgrad(builtins.less)
def gless(x, y, dz):
    return GRAD(False, False)


@rgrad(builtins.switch)
def gswitch(c, t, f, dz):
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


@rgrad(builtins.identity)
def gidentity(v, dz):
    return GRAD(dz)


#################################
# Gradients of other primitives #
#################################


@rgrad(builtins.index)
def gindex(tup, idx, dz):
    def f(pair):
        return switch(pair[0] == idx, dz,
                      zeros_like(Jinv(pair[1])))
    rval = map(f, enumerate(tup))
    return GRAD(rval, 0)


@rgrad(builtins.len)
def glen(xs, dz):
    return GRAD(zeros_like(Jinv(xs)))


@rgrad(builtins.range)
def grange(n, dz):
    return GRAD(0)


@rgrad(builtins.map)
def gmap(f, xs, dz):
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


@rgrad(builtins.enumerate)
def genumerate(xs, dz):
    return GRAD(map(second, dz))


class Grad:
    """
    Transform a Lambda into a Lambda that returns a backpropagator
    in addition to its normal return value.
    """

    # Notation:
    # x_up is the reverse (backprop-ready) version of x
    # x_bprop is a function that takes the sensitivity of x and
    #     returns the sensitivity of the inputs of the function
    #     that returns x
    # x_sen is the sensitivity of the gradient to changes in x,
    #     i.e. the quantity we are ultimately interested in

    def __init__(self,
                 name: Symbol,
                 primal: Lambda,
                 nargs_closure = 0) -> None:
        self.name = name
        assert isinstance(primal, Lambda)
        self.primal = primal
        self.gensym = primal.gen
        assert primal.global_env
        self.global_env = primal.global_env
        self.tagged_map: Dict[Symbol, Symbol] = {}
        self.sensitivity_map: Dict[Symbol, Symbol] = {}
        self.backpropagator_map: Dict[Symbol, Symbol] = {}
        self.zeros: List[TupleT[Symbol, MyiaASTNode]] = []
        self.bprop_variables: Dict[Symbol, bool] = OrderedDict()
        self.nargs_closure = nargs_closure

    def phi(self, var: Symbol, value: MyiaASTNode) \
            -> Sequence[TupleT[Symbol, MyiaASTNode]]:
        """
        Given a variable and the expression it is bound to,
        return a list of (variable, value) bindings to append to
        the forward phase. See p. 26 in the P&S paper.
        """

        # Keep in mind:
        # self.tagged_var(x)          ==>  ↑x, x must be Symbol
        # self.tagged_expr(x)         ==>  ↑x, or x if x is a Value
        # self.backpropagator_var(x)  ==>  ♢x

        if isinstance(value, Symbol):
            # Original:     x = y
            # Transformed:  ↑x = ↑y
            return [(self.tagged_var(var), self.tagged_expr(value)),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Value):
            # Original:     x = 5
            # Transformed:  ↑x = 5
            return [(self.tagged_var(var), value),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Tuple):
            # Original:     x = (y, z)
            # Transformed:  ↑x = (↑y, ↑z)
            return [(self.tagged_var(var),
                     Tuple(self.tagged_expr(a) for a in value.values)),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Apply):
            # Original:     x = f(y)
            # Transformed:  tmp = ↑f(↑y)
            #               ↑x = tmp[0]
            #               ♢x = tmp[1]
            tmp = self.gensym('tmp')
            return [(tmp,
                     Apply(self.tagged_expr(value.fn),
                           *[self.tagged_expr(a) for a in value.args])),
                    (self.tagged_var(var),
                     Apply(builtins.index, tmp, Value(0))),
                    (self.backpropagator_var(var),
                     Apply(builtins.index, tmp, Value(1)))]

        elif isinstance(value, Closure):
            # Original:     x = Closure(f, y, z)
            # Transformed:  ↑f = JX(f, 2)  # evaluated immediately
            #               ↑x = Closure(↑f, ↑y, ↑z)
            # where the last argument to JX is the number of free
            # variables the function is referring to (y and z in
            # the example).

            # We should always statically know ``f`` in order to
            # apply this rule, but this will always be the case
            # if we have a closure in the code.
            assert isinstance(value.fn, Symbol)
            if value.fn.namespace not in {'global', 'builtin'}:
                raise Exception(
                    'First argument to Closure'
                    ' should always be a global variable.'
                )

            args = [self.tagged_expr(a) for a in value.args]
            fn = evaluate(value.fn, self.global_env)
            jfn = JX(fn, len(value.args))
            expr = Closure(jfn.ast.ref, args)

            return [(self.tagged_var(var), expr),
                    (self.backpropagator_var(var), Value(None))]

        else:
            raise Exception(f'phi is not defined on node type: {value}')

    def rho(self, var: Symbol, value: MyiaASTNode) \
            -> List[TupleT[Symbol, MyiaASTNode]]:
        """
        Given a variable and the expression it is bound to,
        return a list of (variable, value) bindings to prepend
        to the backward phase. See p. 26 in the P&S paper.
        """

        # Keep in mind:
        # self.sensitivity_var(x)     ==>  ∇x
        # self.backpropagator_var(x)  ==>  ♢x
        # x += y means x_2 = mapadd(x, y), where x_2 is a fresh
        # variable (to keep single assignment property)

        def args_cast(args: List[MyiaASTNode]) -> List[LeafType]:
            # Just a helper function to satisfy mypy
            assert all(isinstance(a, (Symbol, Value)) for a in args)
            return cast(List[LeafType], args)

        if isinstance(value, Symbol):
            # Original:     x = y
            # Transformed:  ∇x += ∇y
            return self.accum([value], Tuple([self.sensitivity_var(var)]))

        elif isinstance(value, Value):
            # Original:     x = 5
            # Transformed:  <nothing to do>
            return []

        elif isinstance(value, Tuple):
            # Original:     x = (y, z)
            # Transformed:  ∇y += ∇x[0]
            #               ∇z += ∇x[1]
            args = args_cast(value.values)
            return self.accum(args, self.sensitivity_var(var))

        elif isinstance(value, Apply):
            # Original:     x = f(y)
            # Transformed:  tmp = ♢x(∇x)
            #               ∇f += tmp[0]
            #               ∇y += tmp[1]
            args = args_cast([value.fn, *value.args])
            increment = Apply(self.backpropagator_var(var),
                              self.sensitivity_var(var))
            return self.accum(args, increment)

        elif isinstance(value, Closure):
            # Original:     x = Closure(f, y, z)
            # Transformed:  ∇y += ∇x[0]
            #               ∇z += ∇x[1]
            # Why yes, this works the same as Tuple.
            args = args_cast(value.args)
            return self.accum(args, self.sensitivity_var(var))

        else:
            raise Exception(f'rho is not defined on node type: {value}')

    def zero_init(self, var: Symbol) -> Symbol:
        """
        Handle zero initialization code for a variable's gradient.
        That code is:

            ∇x = zeros_like(Jinv(↑x))

        ``Jinv(↑x)`` is the same as ``x``, but we don't have access to
        ``x`` since a transformed function ``↑f`` receives ``↑x``
        directly as an argument. Thankfully, the transformation is
        invertible, and that is why we use ``Jinv``.

        The initialization code is stored in ``self.zeros``, and ``∇x``
        is returned.
        """
        new_var = self.new_sensitivity_var(var)
        init = (new_var,
                Apply(builtins.zeros_like,
                      Apply(builtins.Jinv, self.tagged_expr(var))))
        self.zeros.append(init)
        return new_var

    def accum(self, vars: List[LeafType], value: MyiaASTNode) \
            -> List[TupleT[Symbol, MyiaASTNode]]:
        """
        Return code to accumulate the gradients returned as ``value``
        into a tuple of ``vars``. In other words:

            tmp = value
            vars[0] = tmp[0]
            vars[1] = tmp[1]
            ...

        Some of the variables may be Values (aka not variables),
        in which case we simply ignore them.
        """
        tmp = self.gensym('tmp')
        rval = [(tmp, value)]
        for i, v in enumerate(vars):
            if isinstance(v, Value):
                # No accumulation in non-variables.
                continue
            sen = self.sensitivity_var(v)
            new_sen = self.new_sensitivity_var(v)
            rval.append((new_sen,
                         Apply(builtins.mapadd, sen,
                               Apply(builtins.index, tmp, Value(i)))))
        return rval

    def tagged_var(self, v: MyiaASTNode) -> Symbol:
        """
        Return ``↑v``. Creates it if it does not exist.
        """
        # Maps v to the v_up variable i.e. the tagged variable for v
        assert isinstance(v, Symbol)
        assert v.namespace not in {'global', 'builtin'}
        return copy(self.tagged_map.setdefault(v, self.gensym(v, '↑')))

    def tagged_expr(self, v: MyiaASTNode) -> MyiaASTNode:
        """
        * If ``v`` is a Value, return ``v``.
        * If ``v`` is a global Symbol, return ``J(v)``.
        * Otherwise return ``↑v``.
        """
        assert isinstance(v, (Symbol, Value))
        if isinstance(v, Value):
            return v
        if v.namespace in {'global', 'builtin'}:
            return Apply(builtins.J, v)
        else:
            return self.tagged_var(v)

    def sensitivity_var(self, v: MyiaASTNode) -> Optional[Symbol]:
        """
        Return ``∇v``. If it does not exist, create it and perform
        zero_init.
        """
        assert isinstance(v, (Symbol, Value))
        if isinstance(v, Value):
            return None
        try:
            return copy(self.sensitivity_map[v])
        except KeyError:
            return self.zero_init(v)

    def new_sensitivity_var(self, v: Symbol) -> Symbol:
        """
        Create a new sensitivity variable for v. This is used to preserve
        the single-assignment property: instead of ∇v = ∇v + x,
        we do ∇v_2 = ∇v + x. self.sensitivity_var maps to the latest
        return value for this function.
        """
        assert isinstance(v, Symbol)
        new_v = self.gensym(v, '∇')
        self.sensitivity_map[v] = new_v
        return new_v

    def backpropagator_var(self, v: Symbol) -> Symbol:
        """
        Return ``♢v``. Create it if it does not exist.
        """
        return copy(self.backpropagator_map.setdefault(v, self.gensym(v, '♢')))

    def transform(self) -> Symbol:
        args = self.primal.args
        let = self.primal.body

        if isinstance(let, Symbol):
            tmp = self.gensym('tmp')
            let = Let([(tmp, let)], tmp)
        assert isinstance(let, Let)  # TODO: could be symbol too

        # Create this sensitivity variable first (it's an argument).
        assert isinstance(let.body, Symbol)
        out_sen = self.new_sensitivity_var(let.body)

        forward: List[TupleT[Symbol, MyiaASTNode]] = []
        backward: List[TupleT[Symbol, MyiaASTNode]] = []
        for s, v in let.bindings:
            forward += self.phi(s, v)

        for s, v in reversed(let.bindings):
            backward += self.rho(s, v)

        backp_bargs: List[Symbol] = \
            [self.backpropagator_var(s) for s, _ in let.bindings]
        backp_cargs: List[Symbol] = \
            [self.tagged_var(s) for s, _ in let.bindings]
        backp_rargs: List[Symbol] = \
            [self.tagged_var(arg) for arg in args]
        backp_args = backp_bargs + backp_cargs + backp_rargs
        backp_all_ret = [self.sensitivity_var(arg) for arg in args]
        backp_ret = Tuple([
            Tuple(backp_all_ret[:self.nargs_closure]),
            *backp_all_ret[self.nargs_closure:]
            # Tuple(backp_all_ret[self.nargs_closure:])
        ])

        backp_args_copy: Iterable[Symbol] = map(copy, backp_args)
        backp_fn = Lambda([*backp_args_copy, out_sen],
                          Let(self.zeros + backward, backp_ret),
                          self.gensym)
        backp_sym = self.global_env.gen(self.name, '♢*')
        backp_fn.global_env = self.global_env
        backp_fn.ref = backp_sym
        self.global_env[backp_sym] = backp_fn
        # if self.global_env.url == '__root__':
        root_globals[backp_sym] = backp_fn

        backp_cl = Closure(backp_sym, backp_args)
        backp_clsym = self.gensym(self.name, '♢')
        forward.append((backp_clsym, backp_cl))
        new_body = Let(forward,
                       Tuple([self.tagged_expr(let.body), backp_clsym]))

        new_args = list(map(self.tagged_var, args))
        ret_fn = Lambda(new_args, new_body, self.gensym)
        ret_sym = self.global_env.gen(self.name, '↑')
        ret_fn.global_env = self.global_env
        ret_fn.ref = ret_sym
        self.global_env[ret_sym] = ret_fn
        # if self.global_env.url == '__root__':
        root_globals[ret_sym] = ret_fn
        ret_fn.primal = self.name
        return ret_sym
