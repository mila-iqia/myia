from typing import Dict, List

from .ast import \
    Transformer, GenSym, MyiaASTNode, \
    Symbol, Value, Lambda, Let, Apply, Tuple, Closure
from .interpret import \
    global_env, impl, evaluate, \
    PrimitiveImpl, FunctionImpl, ClosureImpl
from .front import Env, parse_function0
from .symbols import builtins, bsym
from copy import copy
from .compile import a_normal
from .util import Props


builtins.zero = bsym('zero')
builtins.merge = bsym('merge')
builtins.J = bsym('J')
builtins.Jinv = bsym('Jinv')
builtins.shift_grad = bsym('shift_grad')


######################################
# Decorator for gradient definitions #
######################################


# def rgrad(sym):
#     # Copy symbol to grad namespace
#     rsym = Symbol(sym, namespace='builtin', relation='♢*')
#     #Symbol(sym.label, namespace='grad:builtin')

#     def decorator(fn):
#         prim = global_env[sym]
#         assert isinstance(prim, PrimitiveImpl)

#         # Wrap the primitive and a closure-converted backpropagator
#         # in a combined method that follows the protocol
#         G = GenSym()
#         args = [G.sym(a) for a in prim.argnames]
#         forward = Apply(sym, *[Apply(builtins.Jinv, a)
#                                for a in args])
#         backward = Closure(rsym, args)
#         ast = Lambda(args, Tuple([forward, backward]), G)
#         impl = FunctionImpl(ast, global_env)
#         prim.grad = impl
#         impl.primal = prim

#         global_env[rsym] = PrimitiveImpl(fn)
#         return impl

#     return decorator


def rgrad(sym):
    # Copy symbol to grad namespace
    rsym = Symbol(sym, namespace='builtin', relation='♢*')
    #Symbol(sym.label, namespace='grad:builtin')

    def decorator(orig_fn):
        r, bindings = parse_function0(orig_fn)
        fn = evaluate(r, bindings)

        prim = global_env[sym]
        assert isinstance(prim, PrimitiveImpl)

        # Wrap the primitive and a closure-converted backpropagator
        # in a combined method that follows the protocol
        G = GenSym()
        args = [G.sym(a) for a in prim.argnames]
        forward = Apply(sym, *[Apply(builtins.Jinv, a)
                               for a in args])
        backward = Closure(rsym, args)
        ast = Lambda(args, Tuple([forward, backward]), G)
        impl = FunctionImpl(ast, global_env)
        prim.grad = impl
        impl.primal = prim

        global_env[rsym] = fn #PrimitiveImpl(fn)
        return impl

    return decorator


################################################
# Implementation of primitives needed for Grad #
################################################


@impl(builtins.shift_grad)
def shift_grad(closure, n):
    """Given a transformed closure, transforms its bprop
    so that it groups the first n arguments together (these
    arguments are assumed to be the variables closed over)."""
    # TODO: this functionality should be implemented elsewhere,
    # as it is it will play awkwardly with grad(grad), I think.

    # assert isinstance(closure, ClosureImpl)
    def f(*args):
        result, bprop = closure(*args)

        def bprop2(*args):
            results = bprop(*args)
            return (results[0] + results[1:1 + n], *results[1 + n:])
        return (result, PrimitiveImpl(bprop2))
    prim = PrimitiveImpl(f)
    prim.primal = closure.primal
    return prim


@impl(builtins.zero)
def zero(x):
    if isinstance(x, (int, float)):
        return 0
    elif isinstance(x, tuple):
        return tuple(zero(a) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return ()
    elif isinstance(x, ClosureImpl):
        return tuple(zero(a) for a in x.args)
    else:
        raise TypeError(f'Cannot create a zero conformant with {x}')


@impl(builtins.merge)
def merge(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x + y
    elif type(x) is not type(y):
        raise TypeError(f'Cannot merge {x} and {y} (not same type).')
    elif isinstance(x, tuple):
        assert len(x) == len(y)
        return tuple(merge(a, b) for a, b in zip(x, y))
    else:
        raise TypeError(f'Cannot merge values of type {type(x)}')


@impl(builtins.J)
def J(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(J(a) for a in x)
    elif isinstance(x, PrimitiveImpl):
        return x.grad
    elif isinstance(x, FunctionImpl):
        G = Grad(
            name = x.ast.ref or x.ast.gen('???'),
            primal = a_normal(x.ast)
        )
        g = G.transform()
        bindings = {**x.bindings, **G.global_env.bindings}
        gfn = evaluate(g, bindings)
        gfn.primal = x
        x.grad = gfn
        return gfn
        # return make_grad(x)
    elif isinstance(x, ClosureImpl):
        # ??
        c = ClosureImpl(shift_grad(J(x.fn), len(x.args)),
                        J(tuple(x.args)))
        c.primal = x
        return c
    else:
        raise TypeError(f'Invalid argument for J: {x}')


@impl(builtins.Jinv)
def Jinv(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(Jinv(a) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        assert x.primal is not None
        return x.primal
    elif isinstance(x, ClosureImpl):
        c = ClosureImpl(Jinv(x.fn), Jinv(tuple(x.args)))
        c.grad = x
        return c
    else:
        raise TypeError(f'Invalid argument for Jinv: {x}')


###########################################
# Gradients of primitives needed for Grad #
###########################################

myia_builtins = Props(globals())

@rgrad(builtins.zero)
def gzero(x, d):
    return ((), myia_builtins.zero(x))


@rgrad(builtins.merge)
def gmerge(x, y, d):
    return ((), d, d)


@rgrad(builtins.J)
def gJ(x, d):
    return ((), myia_builtins.Jinv(d),)


@rgrad(builtins.Jinv)
def gJinv(x, d):
    return ((), myia_builtins.J(d))


######################################
# Gradients of arithmetic primitives #
######################################


@rgrad(builtins.add)
def gadd(x, y, dz):
    return ((), dz, dz)


@rgrad(builtins.subtract)
def gsubtract(x, y, dz):
    return ((), dz, -dz)


@rgrad(builtins.multiply)
def gmultiply(x, y, dz):
    return ((), dz * y, dz * x)


@rgrad(builtins.divide)
def gdivide(x, y, dz):
    return ((), dz / y, -dz * x / (y * y))


@rgrad(builtins.unary_subtract)
def gunary_subtract(x, dz):
    return ((), -dz)


###################################################
# Gradients of boolean and conditional primitives #
###################################################


@rgrad(builtins.greater)
def ggreater(x, y, dz):
    return ((), False, False)


@rgrad(builtins.less)
def gless(x, y, dz):
    return ((), False, False)


@rgrad(builtins.lazy_if)
def glazy_if(c, t, f, dz):
    if c:
        return ((),
                False,
                t()[1](dz)[0],
                myia_builtins.zero(myia_builtins.Jinv(f)))
    else:
        return ((),
                False,
                myia_builtins.zero(myia_builtins.Jinv(t)),
                f()[1](dz)[0])


@rgrad(builtins.half_lazy_if)
def ghalf_lazy_if(c, t, f, dz):
    if c:
        return ((),
                False,
                t()[1](dz)[0],
                myia_builtins.zero(myia_builtins.Jinv(f)))
    else:
        return ((),
                False,
                myia_builtins.zero(myia_builtins.Jinv(t)),
                dz)


@rgrad(builtins.switch)
def gswitch(c, t, f, dz):
    if c:
        return ((),
                False,
                dz,
                myia_builtins.zero(myia_builtins.Jinv(f)))
    else:
        return ((),
                False,
                myia_builtins.zero(myia_builtins.Jinv(t)),
                dz)


#################################
# Gradients of other primitives #
#################################

@rgrad(builtins.index)
def gindex(tup, idx, dz):
    def f(pair):
        return myia_builtins.switch(pair[0] == idx, dz, 0)
    rval = myia_builtins.map(f, myia_builtins.enumerate(tup))
    return ((), rval, 0)


# Following the methodology in the following paper:
#   http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf

class Grad:
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
                 global_env: Env = None) -> None:
        self.name = name
        assert(isinstance(primal, Lambda))
        self.primal = primal
        self.gensym = primal.gen
        self.global_env = global_env or Env(namespace='global')
        self.tagged_map: Dict[Symbol, Symbol] = {}
        self.sensitivity_map: Dict[Symbol, Symbol] = {}
        self.backpropagator_map: Dict[Symbol, Symbol] = {}
        self.zeroes: List[MyiaASTNode] = []

    def phi(self, var, value):
        # phi (p. 26) transformation on let bindings, transforms
        # the forward phase.

        if isinstance(value, Symbol):
            # x = y ==> x_up = y_up
            return [(self.tagged_var(var), self.tagged_var(value)),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Value):
            # x = 5 ==> x_up = 5
            return [(self.tagged_var(var), value),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Apply):
            # x = f(y) ==> (x_up, x_bprop) = f_up(y_up)
            tmp = self.gensym('tmp')
            return [(tmp,
                     Apply(self.tagged_var(value.fn),
                           *[self.tagged_var(a) for a in value.args])),
                    (self.tagged_var(var),
                     Apply(builtins.index, tmp, Value(0))),
                    (self.backpropagator_var(var),
                     Apply(builtins.index, tmp, Value(1)))]

        elif isinstance(value, Closure):
            # x = lambda y: ... ==> x_up = (lambda y: ...)_up
            # But in our system, we feed free variables explicitly
            # through Closure, and lambda has no freevars, so we do:
            # x = Closure(f, w, z) ==> x_up = Closure(f_up, w_up, z_up) (???)

            args = [self.tagged_var(a) for a in value.args]
            clos = Closure(value.fn, args)
            expr = Apply(builtins.J, clos)
            return [(self.tagged_var(var), expr),
                    (self.backpropagator_var(var), Value(None))]

        elif isinstance(value, Tuple):
            return [(self.tagged_var(var),
                     Tuple(self.tagged_var(a) for a in value.values)),
                    (self.backpropagator_var(var), Value(None))]

        else:
            raise Exception(f'phi is not defined on node type: {value}')

    def rho(self, var, value):
        # rho (p. 26) transformation on let bindings, represents the
        # corresponding operations to do in the backward phase

        if isinstance(value, Symbol):
            # x = y ==> y_sen += x_sen
            return self.accum([value], Tuple([self.sensitivity_var(var)]))

        elif isinstance(value, Value):
            # x = 5 ==> <nothing>
            return []

        elif isinstance(value, Apply):
            # x = f(y) ==> (f_sen, y_sen) += x_bprop(x_sen)
            args = [value.fn, *value.args]
            increment = Apply(self.backpropagator_var(var),
                              self.sensitivity_var(var))
            return self.accum(args, increment)

        elif isinstance(value, Closure):
            # x = Closure(f, w, z) ==> (w_sen, z_sen) += x_sen
            return self.accum(value.args, self.sensitivity_var(var))

        elif isinstance(value, Tuple):
            return self.accum(value.values, self.sensitivity_var(var))

        else:
            raise Exception(f'rho is not defined on node type: {value}')

    def zero_init(self, var):
        new_var = self.new_sensitivity_var(var)
        init = (new_var,
                Apply(builtins.zero,
                      Apply(builtins.Jinv, self.tagged_var(var))))
        self.zeroes.append(init)
        return new_var

    def accum(self, vars, value):
        if isinstance(vars, list):
            vvars = [(i, v) for i, v in enumerate(vars)
                     if not isinstance(v, Value)]
            sens = [self.sensitivity_var(v) or
                    Apply(builtins.zero, Apply(builtins.Jinv, v))
                    for v in vars]
            new_sens = [self.new_sensitivity_var(v) for _, v in vvars]
            tmp = self.gensym('tmp')
            group = Tuple(sens)
            app = Apply(builtins.merge, group, value)
            rval = [(tmp, app)]
            for new_sen, (i, _) in zip(new_sens, vvars):
                rval.append((new_sen, Apply(builtins.index, tmp, Value(i))))
            return rval
        else:
            sen = self.sensitivity_var(var)
            new_sen = self.new_sensitivity_var(var)
            app = Apply(builtins.merge, sen, value)
            return [(new_sen, app)]

    def tagged_var(self, v):
        # Maps v to the v_up variable i.e. the tagged variable for v
        assert isinstance(v, (Symbol, Value))
        if isinstance(v, Value):
            return v
        if v.namespace in {'global', 'builtin'}:
            return Apply(builtins.J, v)
        else:
            return copy(self.tagged_map.setdefault(v, self.gensym(v, '↑')))

    def sensitivity_var(self, v):
        # Maps v to the v_sen variable i.e. the gradient of v
        if isinstance(v, Value):
            return None
        assert isinstance(v, Symbol)
        try:
            return copy(self.sensitivity_map[v])
        except KeyError:
            # self.zeroes.append(self.zero_init(v))
            # return self.new_sensitivity_var(v)
            return self.zero_init(v)

    def new_sensitivity_var(self, v):
        # Create a new sensitivity variable for v. This is used to preserve
        # the single-assignment property: instead of v_sen = v_sen + x,
        # we do v_sen2 = v_sen + x. self.sensitivity_var maps to the latest
        # return value for this function.
        assert isinstance(v, Symbol)
        new_v = self.gensym(v, '∇')
        self.sensitivity_map[v] = new_v
        return new_v

    def backpropagator_var(self, v):
        # Maps v to the v_bprop variable i.e. the backpropagator for v
        return copy(self.backpropagator_map.setdefault(v, self.gensym(v, '♢')))

    def transform(self):
        args = self.primal.args
        let = self.primal.body

        if isinstance(let, Symbol):
            tmp = self.gensym('tmp')
            let = Let([(tmp, let)], tmp)
        assert isinstance(let, Let)  # TODO: could be symbol too

        # Create this sensitivity variable first (it's an argument).
        out_sen = self.new_sensitivity_var(let.body)

        forward = []
        backward = []
        for s, v in let.bindings:
            forward += self.phi(s, v)

        for s, v in reversed(let.bindings):
            backward += self.rho(s, v)

        backp_bargs = [self.backpropagator_var(s) for s, _ in let.bindings]
        backp_cargs = [self.tagged_var(s) for s, _ in let.bindings]
        backp_rargs = [self.tagged_var(arg) for arg in args]
        backp_args = backp_bargs + backp_cargs + backp_rargs
        backp_ret = Tuple([
            # Tuple([self.sensitivity_var(arg.label) for arg in backp_cargs]),
            Tuple([]),
            *[self.sensitivity_var(arg) for arg in args]
        ])
        backp_fn = Lambda([*map(copy, backp_args), out_sen],
                          Let(self.zeroes + backward, backp_ret),
                          self.gensym)
        backp_sym = self.global_env.gen(self.name, '♢*')
        self.global_env[backp_sym] = backp_fn

        backp_cl = Closure(backp_sym, backp_args)
        backp_clsym = self.gensym(self.name, '♢')
        forward.append((backp_clsym, backp_cl))
        new_body = Let(forward,
                       Tuple([self.tagged_var(let.body), backp_clsym]))

        new_args = list(map(self.tagged_var, args))
        ret_fn = Lambda(new_args, new_body, self.gensym)
        ret_sym = self.global_env.gen(self.name, '↑')
        self.global_env[ret_sym] = ret_fn
        return ret_sym

    # def transform_Value(self, node):
    #     return node

    # def transform_Apply(self, node):
    #     return node

    # def transform_Closure(self, node):
    #     return node

    # def transform_If(self, node):
    #     return node

    # def transform_Lambda(self, node):
    #     return node

    # def transform_Let(self, node):
    #     return node

    # def transform_Symbol(self, node):
    #     return node

    # def transform_Tuple(self, node):
    #     return node
