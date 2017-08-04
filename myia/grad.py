from typing import Dict, List

from .ast import \
    Transformer, GenSym, MyiaASTNode, \
    Symbol, Value, Lambda, Let, Apply, Tuple, Closure
from .interpret import \
    root_globals, impl, myia_impl, evaluate, \
    PrimitiveImpl, FunctionImpl, ClosureImpl
from .front import \
    ParseEnv, parse_function, get_global_parse_env
from .symbols import builtins, bsym, gsym
from copy import copy
from .compile import a_normal
from .util import Props
from .buche import buche


builtins.fill = gsym('fill')
builtins.zero = gsym('zero')
builtins.one = gsym('one')
builtins.merge = gsym('merge')
builtins.J = gsym('J')
builtins.JX = gsym('JX')
builtins.Jinv = gsym('Jinv')


######################################
# Decorator for gradient definitions #
######################################


def macro_grad_for(nclos_args):
    def macro_grad(*args):
        return Tuple([Tuple(args[:nclos_args]), *args[nclos_args:]])
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


def rgrad(sym):
    assert isinstance(sym, Symbol)

    def decorator(orig_fn):
        prim = root_globals[sym]
        assert isinstance(prim, PrimitiveImpl)

        _cache = {}

        # Wrap the primitive and a closure-converted backpropagator
        # in a combined method that follows the protocol
        def mkgrad(nargs_closure):
            cached = _cache.get(nargs_closure, None)
            if cached:
                return cached

            # Copy symbol to grad namespace
            rsym = Symbol(sym,
                          version=nargs_closure + 1,
                          namespace='builtin',
                          relation='♢*')

            r, genv = parse_function(
                orig_fn,
                macros={'GRAD': macro_grad_for(nargs_closure)}
            )
            fn = evaluate(r, genv)
            G = GenSym()
            args = [G.sym(a) for a in prim.argnames]
            forward = Apply(builtins.J,
                            Apply(sym, *[Apply(builtins.Jinv, a)
                                         for a in args]))
            backward = Closure(rsym, args)
            ast = Lambda(args, Tuple([forward, backward]), G)
            ast.global_env = get_global_parse_env('__root__')
            impl_sym = ast.global_env.gen('TMP')
            ast.global_env[impl_sym] = ast
            ast.ref = impl_sym
            ast.primal = sym
            impl = FunctionImpl(ast, (root_globals,))
            root_globals[impl_sym] = impl
            root_globals[rsym] = fn
            _cache[nargs_closure] = impl
            return impl

        prim.grad = mkgrad

        return impl

    return decorator


################################################
# Implementation of primitives needed for Grad #
################################################


@impl(builtins.fill)
def fill(x, value):
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


@impl(builtins.zero)
def zero(x):
    return fill(x, 0)


@impl(builtins.one)
def one(x):
    return fill(x, 1)


@impl(builtins.merge)
def merge(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x + y
    elif type(x) is not type(y):
        raise TypeError(f'Cannot merge {x} and {y} (not same type).')
    elif isinstance(x, tuple):
        assert len(x) == len(y)
        return tuple(merge(a, b) for a, b in zip(x, y))
    elif x is None:
        return None
    else:
        raise TypeError(f'Cannot merge values of type {type(x)}')


def JGrad(x):
    _cache = {}

    def make_grad(nargs_closure):
        gfn = _cache.get(nargs_closure, None)
        if gfn:
            return gfn

        G = Grad(
            name = x.ast.ref or x.ast.gen('???'),
            primal = a_normal(x.ast),
            nargs_closure = nargs_closure,
        )
        g = G.transform()

        bindings = {}
        bindings.update(G.global_env.bindings)
        for env in reversed(x.envs):
            bindings.update(env)

        gfn = evaluate(bindings[g])
        _cache[nargs_closure] = gfn
        return gfn
    return make_grad


@impl(builtins.JX)
def JX(x, nargs_closure):
    if isinstance(x, PrimitiveImpl):
        assert x.grad is not None
        return x.grad(nargs_closure)
    elif isinstance(x, FunctionImpl):
        if not x.grad:
            x.grad = JGrad(x)
        return x.grad(nargs_closure)
    else:
        raise TypeError(f'JX applied on wrong type: {x}')


@impl(builtins.J)
def J(x):
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
def Jinv(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(Jinv(a) for a in x)
    elif isinstance(x, (FunctionImpl, PrimitiveImpl)):
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


@rgrad(builtins.zero)
def gzero(x, d):
    return GRAD(zero(x))


@rgrad(builtins.merge)
def gmerge(x, y, d):
    return GRAD(d, d)


@rgrad(builtins.JX)
def gJX(x, n, d):
    return GRAD(Jinv(d), 0)


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


@rgrad(builtins.lazy_if)
def glazy_if(c, t, f, dz):
    if c:
        return GRAD(
            False,
            t()[1](dz)[0],
            zero(Jinv(f))
        )
    else:
        return GRAD(
            False,
            zero(Jinv(t)),
            f()[1](dz)[0]
        )


@rgrad(builtins.half_lazy_if)
def ghalf_lazy_if(c, t, f, dz):
    if c:
        return GRAD(
            (),
            False,
            t()[1](dz)[0],
            zero(Jinv(f))
        )
    else:
        return GRAD(
            False,
            zero(Jinv(t)),
            dz
        )


@rgrad(builtins.switch)
def gswitch(c, t, f, dz):
    if c:
        return GRAD(
            zero(Jinv(c)),  # False
            dz,
            zero(Jinv(f))
        )
    else:
        return GRAD(
            zero(Jinv(c)),  # False
            zero(Jinv(t)),
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
                      zero(Jinv(pair[1])))
    rval = map(f, enumerate(tup))
    return GRAD(rval, 0)


@rgrad(builtins.len)
def glen(xs, dz):
    return GRAD(zero(Jinv(xs)))


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
    d = map(bprops[0], dz)
    df = reduce(merge, map(first, d))
    dxs = map(second, d)
    return GRAD(df, dxs)


@rgrad(builtins.enumerate)
def genumerate(xs, dz):
    return GRAD(map(second, dz))


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
                 # global_env: ParseEnv,
                 nargs_closure = 0) -> None:
        self.name = name
        assert(isinstance(primal, Lambda))
        self.primal = primal
        self.gensym = primal.gen
        # self.global_env = global_env or ParseEnv(namespace='global')
        assert primal.global_env
        self.global_env = primal.global_env
        self.tagged_map: Dict[Symbol, Symbol] = {}
        self.sensitivity_map: Dict[Symbol, Symbol] = {}
        self.backpropagator_map: Dict[Symbol, Symbol] = {}
        self.zeroes: List[MyiaASTNode] = []
        self.nargs_closure = nargs_closure

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

            # These assertions ensure that value.fn is resolvable.
            assert isinstance(value.fn, Symbol)
            if value.fn.namespace not in {'global', 'builtin'}:
                raise Exception(
                    'First argument to Closure'
                    ' should always be a global variable.'
                )

            args = [self.tagged_var(a) for a in value.args]

            fn = evaluate(value.fn, self.global_env)
            if isinstance(fn, PrimitiveImpl):
                # fn = Apply(builtins.JX, value.fn, Value(len(args)))
                # expr = Closure(fn, args)
                fn = evaluate(value.fn, get_global_parse_env('__root__'))
                gfn = JX(fn, len(value.args))
                expr = Closure(gfn.ast.ref, args)
            else:
                # sym = self.global_env.gen('TMPG')
                G = Grad(fn.ast.ref, a_normal(fn.ast), len(value.args))
                grad = G.transform()
                # self.global_env[sym] = grad
                expr = Closure(grad, args)

            # fn = evaluate(value.fn, self.global_env.bindings)
            # g = JX(fn, len(value.args))
            # assert g.primal
            # assert g.ast.ref
            # expr = Closure(g.ast.ref, args)

            # fn = Apply(builtins.JX, value.fn, Value(len(args)))
            # expr = Closure(fn, args)

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
            tmp = self.gensym('tmp')
            rval = [(tmp, value)]
            for i, v in enumerate(vars):
                if isinstance(v, Value):
                    # No accumulation in non-variables.
                    continue
                sen = self.sensitivity_var(v)
                new_sen = self.new_sensitivity_var(v)
                rval.append((new_sen,
                             Apply(builtins.merge, sen,
                                   Apply(builtins.index, tmp, Value(i)))))
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
        backp_all_ret = [self.sensitivity_var(arg) for arg in args]
        backp_ret = Tuple([
            # Tuple([self.sensitivity_var(arg.label) for arg in backp_cargs]),
            Tuple(backp_all_ret[:self.nargs_closure]),
            *backp_all_ret[self.nargs_closure:]
        ])
        backp_fn = Lambda([*map(copy, backp_args), out_sen],
                          Let(self.zeroes + backward, backp_ret),
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
                       Tuple([self.tagged_var(let.body), backp_clsym]))

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
