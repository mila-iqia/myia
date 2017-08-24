
from typing import Dict
from ..symbols import builtins, Keyword
from ..stx import Tuple, maptup2
# from .unify import Variable, OrVariable
from unification import var, Var, isvar, unify, reify
from unification.core import _unify, _reify
from unification.utils import transitive_get as walk
from ..interpret.runtime import symbol_associator
from itertools import product
from ..front import parse_function
from ..util.buche import buche


############################
# Top of the value lattice #
############################


ANY = Keyword('ANY')


#########################
# Unification extension #
#########################


class Union:
    def __new__(cls, opts):
        if len(set(opts)) == 1:
            # Union([a]) => a
            for o in opts:
                if not isinstance(o, Union):
                    # Just a trick to return the one element in
                    # opts even if it's not indexable
                    return o
        return super().__new__(cls)

    def __init__(self, opts):
        self.opts = set()
        for opt in opts:
            if isinstance(opt, Union):
                self.opts.update(opt.opts)
            else:
                self.opts.add(opt)

    def __repr__(self):
        return f'Union({", ".join(map(str, self.opts))})'


def reify_union(u, d):
    return Union([reify(o, d) for o in u.opts])


def unify_set(s1, s2, d):
    if len(s1) != len(s2):
        return False
    i = s1 & s2
    a = s1 - i
    b = s2 - i
    aa = sorted(a, key=lambda x: 1 if isinstance(x, Var) else -1)
    bb = sorted(b, key=lambda x: 1 if isinstance(x, Var) else -1)
    for x in aa:
        for y in bb:
            d2 = unify(x, y, d)
            if d2:
                break
        else:
            return False
        bb.remove(y)
        d = d2
    return d


def unify_union(u1: Union, u2: Union, d: dict):
    return unify_set(u1.opts, u2.opts, d)


def unify_union_var(u: Union, v: Var, d: dict):
    # unify(X, Union(a, b, X)) ==> unify(X, Union(a, b))
    if v in u.opts:
        return Union(u.opts.difference({v}))
    else:
        v = walk(v, d)
        if isvar(v):
            return {**d, v: u}
        return unify(u, v)


def unify_var_union(v, u, d: dict):
    return unify_union_var(u, v, d)


_unify.add((set, set, dict), unify_set)
_unify.add((Union, Union, dict), unify_union)
unify.add((Union, Var, dict), unify_union_var)
unify.add((Var, Union, dict), unify_var_union)
_reify.add((Union, dict), reify_union)


############
# Builtins #
############


class BuiltinCollection:
    """
    Implements a "module" of sorts. It has no methods,
    only fields that are populated by ``impl``.
    """
    pass


# Myia's global variables. Used for evaluation.
aroot_globals = {
    builtins.myia_builtins: BuiltinCollection()
}


class AImpl:
    def __init__(self):
        self.mem = {}

    def __call__(self, *args, proj=None):
        assert proj is None
        return self.run(*args)
        # args2 = [arg.opts if isinstance(arg, Union) else [arg]
        #          for arg in args]
        # results = []
        # for args in product(*args2):
        #     try:
        #         res = self.mem[args]
        #     except TypeError:
        #         res = self.run(*args)
        #     except KeyError:
        #         self.mem[args] = var()
        #         res = self.run(*args)
        #         self.mem[args] = res
        #     results.append(res)
        # if len(results) == 1:
        #     return results[0]
        # else:
        #     return Union(results)


class PrimitiveAImpl(AImpl):
    def __init__(self, run, sym=None):
        super().__init__()
        self.sym = sym
        self.run = run


class FunctionAImpl(AImpl):
    __cache__: Dict = {}

    def __new__(cls, lbda, genv):
        if lbda in cls.__cache__:
            return cls.__cache__[lbda]
        else:
            inst = super().__new__(cls)
            cls.__cache__[lbda] = inst
            return inst

    def __init__(self, lbda, genv):
        super().__init__()
        self.lbda = lbda
        self.genv = genv

    def __call__(self, *args, proj=None):
        return self.run(*args, proj=proj)

    def run(self, *args, proj=None):
        node = self.lbda
        env = {sym: value if isinstance(value, AValue) else AValue(value)
               for sym, value in zip(node.args, args)}
        return AbstractInterpreter(env, self.genv).eval(node.body, proj)


class ClosureAImpl(AImpl):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self, *args, proj=None):
        return self.run(*args, proj=proj)

    def run(self, *args, proj=None):
        return self.fn(*self.args, *args, proj=proj)


##########################
# Implementation helpers #
##########################


@symbol_associator("aimpl_")
def aimpl(sym, name, fn):
    prim = PrimitiveAImpl(fn, sym)
    aroot_globals[sym] = prim
    return prim


def std_aimpl(fn):
    def deco(*args):
        unwrapped = [arg[None] for arg in args]
        rval = fn(*unwrapped)
        return AValue(rval)
        # if any(isinstance(arg, Var) for arg in args):
        #     return var()
        # else:
        #     return fn(*args)
    deco.__name__ = fn.__name__
    return aimpl(deco)


@std_aimpl
def aimpl_add(x, y):
    return x + y


@std_aimpl
def aimpl_subtract(x, y):
    return x - y


@std_aimpl
def aimpl_dot(x, y):
    return x @ y


@std_aimpl
def aimpl_index(xs, idx):
    return xs[idx]


@std_aimpl
def aimpl_shape(xs):
    return xs.shape


@std_aimpl
def aimpl_assert_true(v, message):
    assert v, message


@std_aimpl
def aimpl_equal(x, y):
    return x == y


@std_aimpl
def aimpl_greater(x, y):
    return x > y


@aimpl
def aimpl_identity(x):
    return x[None]


@aimpl
def aimpl_switch(cond, t, f):
    cond = cond[None]
    if isinstance(cond, bool):
        return t[None] if cond else f[None]
    elif isinstance(cond, Var):
        return Union([t, f])
    else:
        raise TypeError(f'Cannot switch on {cond}')


##############
# Projectors #
##############


projector_set = {aimpl_shape}


projectors: Dict = {}


def proj(psym):
    projs = projectors.setdefault(psym, {})

    @symbol_associator('proj_')
    def pimpl(sym, name, fn):
        fsym, fenv = parse_function(fn)
        projs[aroot_globals[sym]] = FunctionAImpl(fenv[fsym], aroot_globals)
        return fn

    return pimpl


@proj(builtins.shape)
def proj_add(x, y):
    assert shape(x) == shape(y)
    return shape(x)


@proj(builtins.shape)
def proj_dot(x, y):
    # sx = shape(x)
    # sy = shape(y)
    # assert sx[1] == sy[0]
    # return sx[0], sy[1]
    a, b = shape(x)
    c, d = shape(y)
    assert b == c
    return a, d


########################
# Abstract interpreter #
########################


class AValue:
    def __init__(self, interpreter, node=None):
        if node is None:
            self.interpreter = None
            self.node = None
            values = interpreter
            if isinstance(values, dict):
                self.values = values
            elif isinstance(values, AValue):
                self.node = values.node
                self.interpreter = values.interpreter
                self.values = values.values
            else:
                self.values = {None: values}
        else:
            self.interpreter = interpreter
            self.node = node
            self.values = {}

    def __getitem__(self, proj):
        if proj not in self.values:
            if self.node:
                v = self.interpreter.eval(self.node, proj)
            elif proj is None:
                v = self.values[None]
            else:
                v = aroot_globals[proj](self)
            while isinstance(v, AValue):
                v = v[None]
            self.values[proj] = v
        return self.values[proj]

    def __call__(self, proj):
        rval = self[proj]
        if proj is None:
            return self
        else:
            return rval


# class Delayed:
#     def __init__(self, interpreter, node):
#         self.interpreter = interpreter
#         self.node = node

#     def __call__(self, proj):
#         return self.interpreter.eval(self.node, proj)

def find_projector(proj, fn):
    if isinstance(fn, PrimitiveAImpl):
        try:
            return projectors[proj][fn]
        except KeyError:
            raise Exception(f'Missing prim projector "{proj}" for {fn}.')
    elif isinstance(fn, ClosureAImpl):
        return ClosureAImpl(find_projector(proj, fn.fn), fn.args)
    elif isinstance(fn, FunctionAImpl):
        def fn2(*args, **kw):
            return fn(*args, proj=proj)
        return fn2
    else:
        raise Exception(f'Cannot project "{proj}" with {fn}.')


class AbstractInterpreter:
    def __init__(self,
                 env,
                 global_env=aroot_globals):
        self.global_env = global_env
        self.env = env

    def eval(self, node, proj=None):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'eval_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type for evaluation: {}".format(cls)
            )
        try:
            rval = method(node, proj)
        except Exception as exc:
            level = getattr(exc, 'level', 0)
            exc.level = level + 1
            node.annotations = node.annotations | {'error', f'error{level}'}
            raise exc from None
        return rval

    def eval_Apply(self, node, proj):
        fn = self.eval(node.fn)

        # args = list(map(self.eval, node.args))
        args = [AValue(self, arg) for arg in node.args]

        if fn in projector_set:
            assert proj is None
            assert len(node.args) == 1
            assert fn.sym
            return self.eval(node.args[0], proj=fn.sym)

        # def helper(fn, args):
        #     if isinstance(fn, Union):
        #         return Union(helper(poss, args) for poss in fn.opts)
        #     elif isinstance(fn, Var):
        #         # raise TypeError('Cannot call a Var.')
        #         return var()
        #     else:
        #         return fn(*args)

        # return helper(fn, args)

        if proj is None:
            return fn(*args)
        else:
            pfn = find_projector(proj, fn)
            return pfn(*args)

    def eval_Begin(self, node, proj):
        for stmt in node.stmts[:-1]:
            self.eval(stmt)
        return self.eval(node.stmts[-1], proj)

    def eval_Closure(self, node, proj):
        fn = self.eval(node.fn)
        args = [AValue(self, arg) for arg in node.args]
        return ClosureAImpl(fn, args)

    def eval_Lambda(self, node, proj):
        fn = FunctionAImpl(node, self.global_env)
        self.global_env[node.ref] = fn
        return fn

    def eval_Let(self, node, proj):
        def stash(k, v):
            self.env[k] = v

        for k, v in node.bindings:
            if isinstance(k, Tuple):
                v = self.eval(v)
                maptup2(stash, k, v)
            else:
                self.env[k] = AValue(self, v)
        return self.eval(node.body, proj)

    def eval_Symbol(self, node, proj):
        try:
            v = self.env[node]
        except KeyError:
            v = self.global_env[node]
        if isinstance(v, AValue):
            return v[proj]
        else:
            assert proj is None
            return v

    def eval_Tuple(self, node, proj):
        assert proj is None
        # return tuple(self.eval(x) for x in node.values)
        return tuple(AValue(self, x) for x in node.values)

    def eval_Value(self, node, proj):
        assert proj is None
        return node.value
