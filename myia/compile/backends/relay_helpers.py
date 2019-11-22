"""Collection of helpers for the Relay backend.

Most of those should go away as Relay main development progresses.
"""

import numpy as np
from tvm import relay
from tvm.relay import adt, transform

from ...abstract import (
    AbstractArray,
    AbstractError,
    AbstractFunction,
    AbstractHandle,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    TypedPrimitive,
    VirtualFunction,
    broaden,
)
from ...utils import overload
from ...xtype import Bool, EnvType, Nil, UniverseType, type_to_np_dtype

union_type = relay.GlobalTypeVar('$_union_adt')
empty_union = adt.Constructor("empty", [], union_type)
tag_map = {}
rev_tag_map = {}


def get_union_ctr(tag, t):
    """Get the relay constructor for a union tag."""
    if tag not in tag_map:
        assert t is not None
        rt = to_relay_type(t)
        ctr = adt.Constructor(f"c{tag}", [rt], union_type)
        tag_map[tag] = ctr
        rev_tag_map[ctr] = tag
    return tag_map[tag]


def get_myia_tag(ctr):
    """Return the myia tag for a constructor."""
    return rev_tag_map[ctr]


env_val = relay.GlobalTypeVar('$_env_val')

env_type = relay.GlobalTypeVar('$_env_adt')
a = relay.ty.TypeVar("a")
empty_env = adt.Constructor("empty_env", [], env_type)
cons_env = adt.Constructor("cons_env", [relay.ty.scalar_type('int64'),
                                        a, env_type(a)], env_type)


class TypeHelper:
    """Class to help manage and generate helper types."""

    def __init__(self):
        """Initialize the caches."""
        self.env_val_map = {}

    def initialize(self, mod, mng):
        """Add stub types to the module."""
        mod[env_type] = adt.TypeData(env_type, [a], [empty_env, cons_env])
        for node in mng.all_nodes:
            if isinstance(node.abstract, AbstractTaggedUnion):
                for opt in node.abstract.options:
                    get_union_ctr(*opt)
        mod[union_type] = adt.TypeData(
            union_type, [], list(tag_map.values()))

    def finalize(self, mod):
        """Fill in stub type definitions."""
        mod[env_val] = adt.TypeData(
            env_val, [], list(m['ctr'] for m in self.env_val_map.values()))
        for m in self.env_val_map.values():
            for k, v in m.items():
                if k == 'ctr':
                    continue
                mod[v[0]] = v[1]

    def _get_env_val_map(self, t):
        t = broaden(t)
        rt = to_relay_type(t)
        if t not in self.env_val_map:
            name = f"v{len(self.env_val_map)}"
            self.env_val_map[t] = {
                'ctr': adt.Constructor(name, [rt], env_val)
            }
        return self.env_val_map[t], rt

    def get_env_update(self, val_t):
        """Return a function to update a grad env."""
        m, rt = self._get_env_val_map(val_t)
        res = m.get('env_update', None)
        if res is None:
            res = self._make_env_update(m, rt)
        return res[0]

    def get_env_find(self, val_t):
        """Return a function to get a value from a grad env."""
        m, rt = self._get_env_val_map(val_t)
        res = m.get('env_find', None)
        if res is None:
            res = self._make_env_find(m, rt)
        return res[0]

    def _make_env_update(self, m, rval_t):
        ctr = m['ctr']
        gv = relay.GlobalVar(f"$_env_update<{ctr.name_hint}>")

        env = relay.Var("env", env_type(env_val()))
        key = relay.Var("key", relay.ty.scalar_type('int64'))
        val = relay.Var("val", rval_t)

        k = relay.Var("k")
        v = relay.Var("v")
        r = relay.Var("r")

        empty_clause = adt.Clause(
            adt.PatternConstructor(empty_env, []),
            cons_env(key, ctr(val), env))
        cons_clause = adt.Clause(
            adt.PatternConstructor(cons_env, [adt.PatternVar(k),
                                              adt.PatternVar(v),
                                              adt.PatternVar(r)]),
            relay.If(relay.equal(key, k),
                     cons_env(key, ctr(val), env),
                     cons_env(k, v, relay.Call(gv, [r, key, val]))))
        body = adt.Match(env, [empty_clause, cons_clause])
        fn = relay.Function([env, key, val], body, env_type(env_val()))
        m['env_update'] = (gv, fn)
        return gv, fn

    def _make_env_find(self, m, rval_t):
        ctr = m['ctr']
        gv = relay.GlobalVar(f"$_env_find<{ctr.name_hint}>")

        env = relay.Var("env", env_type(env_val()))
        key = relay.Var("key", relay.ty.scalar_type('int64'))
        dft = relay.Var("dft", rval_t)

        k = relay.Var("k")
        v = relay.Var("v")
        r = relay.Var("r")
        x = relay.Var("x")

        extract_clause = adt.Clause(
            adt.PatternConstructor(ctr, [adt.PatternVar(x)]),
            x)

        empty_clause = adt.Clause(
            adt.PatternConstructor(empty_env, []),
            dft)
        cons_clause = adt.Clause(
            adt.PatternConstructor(cons_env, [adt.PatternVar(k),
                                              adt.PatternVar(v),
                                              adt.PatternVar(r)]),

            relay.If(relay.equal(key, k),
                     adt.Match(v, [extract_clause], complete=False),
                     relay.Call(gv, [r, key, dft])))
        body = adt.Match(env, [empty_clause, cons_clause])
        fn = relay.Function([env, key, dft], body, rval_t)
        m['env_find'] = (gv, fn)
        return gv, fn


@overload(bootstrap=True)
def to_relay_type(self, a: AbstractScalar):
    """Convert a myia abstract to a Relay type."""
    tp = a.xtype()
    if issubclass(tp, Bool):
        return relay.ty.scalar_type('bool')
    elif issubclass(tp, Nil):
        return relay.ty.TupleType([])
    elif issubclass(tp, EnvType):
        return env_type(env_val())
    elif issubclass(tp, UniverseType):
        return relay.ty.scalar_type('uint64')
    else:
        return relay.ty.scalar_type(type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTuple):
    return relay.ty.TupleType([self(e) for e in a.elements])


@overload  # noqa: F811
def to_relay_type(self, a: AbstractArray):
    tp = a.element.xtype()
    return relay.ty.TensorType(a.xshape(), type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractFunction):
    sings = list(self(sing) for sing in a.get_sync())
    for sing in sings[1:]:
        assert sing == sings[0]
    return sings[0]


@overload  # noqa: F811
def to_relay_type(self, a: (VirtualFunction, TypedPrimitive)):
    return relay.ty.FuncType([self(aa) for aa in a.args],
                             self(a.output))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTaggedUnion):
    return union_type()


@overload  # noqa: F811
def to_relay_type(self, a: AbstractHandle):
    return relay.ty.RefType(relay.ty.TupleType(
        [relay.ty.scalar_type('uint64'), self(a.element)]))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractError):
    return relay.ty.scalar_type('uint16')


def dead_value(t):
    """Make a value of the specified type."""
    if isinstance(t, AbstractError):
        return relay.const(0xDEAD, 'uint16')
    return _placeholder_body(to_relay_type(t))


def handle_wrapper(fn, handle_cst, handle_params):
    """Wraps a model function to perform handle updates."""
    def wrapper(*args):
        handle_instances = list(handle_cst)
        handle_instances.extend(get(args[i]) for i, get in handle_params)
        res = fn(*args)
        u = res[0]
        res = res[1] if len(res) == 2 else res[1:]
        for h, v in zip(handle_instances, u):
            h.value = type(h.value)(h.value.fields[0], v)
        return (), res
    if len(handle_cst) + len(handle_params) == 0:
        return fn
    else:
        return wrapper


def _placeholder_body(type):
    if isinstance(type, relay.TensorType):
        sh = [sh.value for sh in type.shape]
        return relay.const(np.array(np.random.rand(*sh)).astype(type.dtype),
                           dtype=type.dtype)
    elif isinstance(type, relay.TupleType):
        return relay.Tuple([_placeholder_body(f) for f in type.fields])
    elif isinstance(type, relay.FuncType):
        params = []
        for arg_ty in type.arg_types:
            params.append(relay.var("p", type_annotation=arg_ty))

        return relay.Function(
            params,
            _placeholder_body(type.ret_type),
            ret_type=type.ret_type)
    elif isinstance(type, relay.TypeCall):
        if type.func == union_type:
            return empty_union()
        elif type.func == env_type:
            return empty_env()
        else:  # pragma: no cover
            raise ValueError(f"Can't build value for adt: {type.func}")
    elif isinstance(type, relay.RefType):
        return relay.RefCreate(_placeholder_body(type.value))
    else:  # pragma: no cover
        raise ValueError(f"Can't build value of type {type}")


def add_functions(mod, funcs):
    """Workaround for type checker and mutually recursive functions."""
    for gv in funcs:
        func = funcs[gv]
        body = _placeholder_body(func.ret_type)
        mod[gv] = relay.Function(func.params, body, func.ret_type)

    for gv in funcs:
        mod[gv] = funcs[gv]


pass_set = transform.Sequential(
    passes=[
        transform.SimplifyInference(),
        transform.CanonicalizeOps(),
        transform.CanonicalizeCast(),
        transform.FuseOps(3),
        # transform.CombineParallelConv2d(),
        transform.AlterOpLayout(),
        # transform.RewriteAnnotatedOps(???),
    ],
    opt_level=0
)


def optimize(mod):
    """Optimize all the functions in a module.

    Modules are the only mutable piece of Relay.  We write an
    optimization pass over the module which destructively updates each
    function while optimizing.
    """
    return pass_set(mod)


__all__ = [
    'TypeHelper',
    'add_functions',
    'dead_value',
    'empty_env',
    'get_myia_tag',
    'get_union_ctr',
    'optimize',
    'to_relay_type',
]
