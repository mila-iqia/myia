from typing import Dict, List

from .ast import \
    Transformer, GenSym, MyiaASTNode, \
    Symbol, Value, Lambda, Let, Apply, Tuple, Closure
from .interpret import \
    global_env, impl, \
    PrimitiveImpl, FunctionImpl, ClosureImpl
from .front import Env
from .symbols import builtins, bsym
from copy import copy


builtins.zero = bsym('zero')
builtins.merge = bsym('merge')
builtins.J = bsym('J')
builtins.Jinv = bsym('Jinv')


######################################
# Decorator for gradient definitions #
######################################


def rgrad(sym):
    # Copy symbol to grad namespace
    rsym = Symbol(sym.label, namespace='grad:builtin')

    def decorator(fn):
        prim = global_env[sym]
        assert isinstance(prim, PrimitiveImpl)

        # Wrap the primitive and a closure-converted backpropagator
        # in a combined method that follows the protocol
        G = GenSym()
        args = [G.sym(a) for a in prim.argnames]
        forward = Apply(sym, *args)
        backward = Closure(rsym, args)
        ast = Lambda(args, Tuple([forward, backward]), G)
        impl = FunctionImpl(ast, global_env)
        prim.grad = impl
        impl.primal = prim

        global_env[rsym] = PrimitiveImpl(fn)
        return impl

    return decorator


################################################
# Implementation of primitives needed for Grad #
################################################


@impl(builtins.zero)
def zero(x):
    if isinstance(x, (int, float)):
        return 0
    elif isinstance(x, tuple):
        return tuple(gzero(a) for a in x)
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
        return make_grad(x)
    elif isinstance(x, ClosureImpl):
        # ??
        return ClosureImpl(J(x.fn), J(x.args))
    else:
        raise TypeError(f'Invalid argument for J: {x}')


@impl(builtins.Jinv)
def Jinv(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(Jinv(a) for a in x)
    elif isinstance(x, (PrimitiveImpl, FunctionImpl)):
        return x.primal
    elif isinstance(x, ClosureImpl):
        return ClosureImpl(Jinv(x.fn), Jinv(x.args))
    else:
        raise TypeError(f'Invalid argument for Jinv: {x}')


###########################################
# Gradients of primitives needed for Grad #
###########################################


@rgrad(builtins.zero)
def gzero(x, d):
    return ((), zero(x))


@rgrad(builtins.merge)
def gmerge(x, y, d):
    return ((), d, d)


@rgrad(builtins.J)
def gJ(x, d):
    return ((), Jinv(d),)


@rgrad(builtins.Jinv)
def gJinv(x, d):
    return ((), J(d))


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
            return [(self.tagged_var(var),
                     Closure(self.tagged_var(value.fn), args))]

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

        elif isinstance(value, Apply):
            # x = f(y) ==> (f_sen, y_sen) += x_bprop(x_sen)
            args = [value.fn, *value.args]
            increment = Apply(self.backpropagator_var(var),
                              self.sensitivity_var(var))
            return self.accum(args, increment)

        elif isinstance(value, Closure):
            # x = Closure(f, w, z) ==> (w_sen, z_sen) += x_sen
            return self.accum(value.args, self.tagged_var(var))

        elif isinstance(value, Tuple):
            return self.accum(value.values, self.tagged_var(var))

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
            sens = list(map(self.sensitivity_var, vars))
            new_sens = list(map(self.new_sensitivity_var, vars))
            tmp = self.gensym('tmp')
            group = Tuple(sens)
            app = Apply(builtins.merge, group, value)
            rval = [(tmp, app)]
            for i, new_sen in enumerate(new_sens):
                rval.append((new_sen, Apply(builtins.index, tmp, Value(i))))
            return rval
        else:
            sen = self.sensitivity_var(var)
            new_sen = self.new_sensitivity_var(var)
            app = Apply(builtins.merge, sen, value)
            return [(new_sen, app)]

    def tagged_var(self, v):
        # Maps v to the v_up variable i.e. the tagged variable for v
        assert isinstance(v, Symbol)
        if v.namespace in {'global', 'builtin'}:
            return Apply(builtins.J, v)
        else:
            return copy(self.tagged_map.setdefault(v, self.gensym(v, '↑')))

    def sensitivity_var(self, v):
        # Maps v to the v_sen variable i.e. the gradient of v
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
        assert isinstance(let, Let)  # TODO: could be symbol too

        # Create this sensitivity variable first (it's an argument).
        out_sen = self.new_sensitivity_var(let.body)

        forward = []
        backward = []
        for s, v in let.bindings:
            forward += self.phi(s, v)

        # zeros = [self.zero_init(s) for s in args] \
        #     + [self.zero_init(s) for s, _ in let.bindings]

        for s, v in reversed(let.bindings):
            backward += self.rho(s, v)

        backp_bargs = [self.backpropagator_var(s) for s, _ in let.bindings]
        backp_cargs = [self.tagged_var(s) for s, _ in let.bindings]
        backp_rargs = [self.tagged_var(arg) for arg in args]
        backp_args = backp_bargs + backp_cargs + backp_rargs
        backp_ret = Tuple([
            Tuple([self.sensitivity_var(arg.label) for arg in backp_cargs]),
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
