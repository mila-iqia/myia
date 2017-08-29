
from typing import Dict
from ..symbols import builtins, Keyword
from ..stx import Tuple, Lambda, maptup2
# from .unify import Variable, OrVariable
from unification import var, Var, isvar, unify, reify
from unification.core import _unify, _reify
from unification.utils import transitive_get as walk
from itertools import product
from ..front import parse_function
from ..util.buche import buche
from copy import copy
from ..impl.main import impl_bank


########
# Misc #
########


aroot_globals = impl_bank['abstract']
projector_set = set()


def load():
    from ..impl.impl_abstract import _
    from ..impl.proj_shape import _
    from ..impl.proj_type import _
    for p in ('shape', 'type'):
        projector_set.add(aroot_globals[builtins[p]])


class Unsatisfiable(Exception):
    pass


class Not:
    def __init__(self, unif):
        self.unif = unif

    def __repr__(self):
        return f'~{self.unif}'


collector = None


class AbstractCollect:
    def __init__(self):
        self.cache = []
        self.watchlists = {}
        self.errors = []

    def __enter__(self):
        global collector
        collector = self

    def __exit__(self, exc_type, exc, tb):
        global collector
        collector = None


############
# Keywords #
############


# Top of the value lattice
ANY = Keyword('ANY')
# Principal value
VALUE = Keyword('VALUE')
# Unifications
UNIFY = Keyword('UNIFY')
# Error associated to the operation
ERROR = Keyword('ERROR')


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


def merge(*ds):
    """
    Merge unification dictionaries (TODO: improve this
    naive implementation).
    """
    if len(ds) == 0:
        return {}
    rval, *ds = ds
    for d in ds:
        for k, v in d.items():
            if not rval:
                return rval
            rval = unify(k, v, rval)
    return rval


############
# Builtins #
############


class AImpl:
    def __init__(self):
        pass


class PrimitiveAImpl(AImpl):
    def __init__(self, run, sym=None):
        super().__init__()
        self.sym = sym
        self._run = run

    def __call__(self, *args, proj=VALUE):
        assert proj is VALUE
        return self._run(*args)

    def __str__(self):
        return f'Prim({self.sym})'


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

    def __call__(self, *args, proj=VALUE):
        node = self.lbda
        # env = {sym: value if isinstance(value, AbstractValue) \
        #                   else AbstractValue(value)
        #        for sym, value in zip(node.args, args)}
        env = {sym: wrap_abstract(value)
               for sym, value in zip(node.args, args)}
        return AbstractInterpreter(env, self.genv).eval(node.body, proj)


class ClosureAImpl(AImpl):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self, *args, proj=VALUE):
        return self.fn(*self.args, *args, proj=proj)


########################
# Abstract interpreter #
########################


class AbstractData:
    def __init__(self, values=None):
        self.values = values or {}

    def __getitem__(self, proj):
        if proj not in self.values:
            if proj is UNIFY:
                u = {}
                self.values[proj] = u
                return u
            res = self.acquire(proj)
            if proj is VALUE and isinstance(res, AbstractData):
                self.values = copy(res.values)
                return self[VALUE]
            else:
                self.values[proj] = res
        return self.values[proj]

    def __call__(self, proj):
        rval = self[proj]
        if proj is VALUE:
            return self
        else:
            return rval

    def __repr__(self):
        return repr(self.values)


class AbstractComputation(AbstractData):
    def __init__(self, interpreter, node):
        super().__init__()
        self.interpreter = interpreter
        self.node = node

    def acquire(self, proj):
        return self.interpreter.eval(self.node, proj)


class AbstractValue(AbstractData):
    def __init__(self, value):
        if isinstance(value, dict):
            assert not isinstance(value.get(VALUE, None), Union)
            super().__init__(value)
        else:
            assert not isinstance(value, Union)
            super().__init__({VALUE: value})

    def acquire(self, proj):
        if proj is VALUE:
            return self.values[VALUE]
        else:
            return aroot_globals[proj](self)


unify.add((AbstractData, AbstractData, dict),
          lambda a, b, d: unify(a[None], b[None], d))
unify.add((AbstractData, object, dict),
          lambda a, x, d: unify(a[None], x, d))
unify.add((object, AbstractData, dict),
          lambda x, a, d: unify(x, a[None], d))


def wrap_abstract(v):
    if isinstance(v, AbstractData):
        return v
    else:
        return AbstractValue(v)


def unwrap_abstract(v):
    if isinstance(v, AbstractData):
        return v[VALUE]
    else:
        return v


def find_projector(proj, fn):
    if proj is VALUE:
        return fn
    if isinstance(fn, PrimitiveAImpl):
        try:
            # return projectors[proj][fn]
            return impl_bank['project'][proj][fn]
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

    def eval(self, node, proj=VALUE):
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
        fn = unwrap_abstract(self.eval(node.fn))

        # args = list(map(self.eval, node.args))
        args = [AbstractComputation(self, arg) for arg in node.args]

        if fn in projector_set:
            assert proj is VALUE
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

        return find_projector(proj, fn)(*args)

    def eval_Begin(self, node, proj):
        for stmt in node.stmts[:-1]:
            self.eval(stmt)
        return self.eval(node.stmts[-1], proj)

    def eval_Closure(self, node, proj):
        fn = self.eval(node.fn)
        args = [AbstractComputation(self, arg) for arg in node.args]
        return ClosureAImpl(fn, args)

    def eval_Lambda(self, node, proj):
        fn = FunctionAImpl(node, self.global_env)
        self.global_env[node.ref] = fn
        return fn

    def eval_Let(self, node, proj):
        def _maptup(vals1, vals2):
            if isinstance(vals1, Tuple):
                vals2 = unwrap_abstract(vals2)
                assert len(vals1.values) == len(vals2)
                return Tuple(_maptup(x, y)
                             for x, y in zip(vals1.values, vals2))
            else:
                self.env[vals1] = vals2

        for k, v in node.bindings:
            if isinstance(k, Tuple):
                v = self.eval(v)
                _maptup(k, v)
            else:
                self.env[k] = AbstractComputation(self, v)
        return self.eval(node.body, proj)

    def eval_Symbol(self, node, proj):
        try:
            v = self.env[node]
        except KeyError:
            v = self.global_env[node]
        if isinstance(v, AbstractData):
            return v[proj]
        else:
            assert proj is VALUE
            return v

    def eval_Tuple(self, node, proj):
        assert proj is VALUE
        return tuple(AbstractComputation(self, x) for x in node.values)

    def eval_Value(self, node, proj):
        assert proj is VALUE
        return node.value


def abstract_evaluate(lbda, args, proj=None):
    load()
    parse_env = lbda.global_env
    assert parse_env is not None
    fn = AbstractInterpreter(parse_env).eval(lbda)
    with AbstractCollect():
        return fn(*args, proj=proj)
