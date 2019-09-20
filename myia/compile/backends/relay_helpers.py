"""Collection of helpers for the Relay backend.

Most of those should go away as Relay main development progresses.
"""

import numpy as np
from tvm import relay
from tvm.relay import adt, transform

from ...abstract import (
    AbstractArray,
    AbstractFunction,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    PartialApplication,
    TypedPrimitive,
    VirtualFunction,
)
from ...xtype import Bool, Nil, type_to_np_dtype
from ...utils import overload

union_type = relay.GlobalTypeVar('$_union_adt')
empty_union = adt.Constructor("empty", [], union_type)
tag_map = {}
rev_tag_map = {}


def get_union_ctr(tag, t):
    """Get the relay constructor for a tag."""
    if tag not in tag_map:
        rt = to_relay_type(t)
        tag_map[tag] = adt.Constructor(f"c{tag}", [rt], union_type)
        rev_tag_map[tag_map[tag]] = tag
    return tag_map[tag]


env_val = relay.GlobalTypeVar('$_env_val')
env_val_map = {}
rev_env_val_map = {}


def get_env_val_map(t):
    if t not in env_val_map:
        rt = to_relay_type(t)
        env_val_map[t] = {
            'ctr': adt.Constructor(f"v{len(env_val_map)}", [rt], env_val)
        }
        rev_env_val_map[env_val_map[t]['ctr']] = t
    return env_val_map[t]


def get_env_update(val_t):
    m = get_env_val_map(val_t)
    return m['env_update'][0]


env_type = relay.GlobalTypeVar('$_env_adt')
empty_env = adt.Constructor("empty_env", [], env_type)
cons_env = adt.Constructor("cons_env", [relay.ty.scalar_type('int32'),
                                        env_val(), env_type()], env_type)

def add_env_types(mod):
    """Add types and functions to a relay module."""
    mod[env_val] = adt.TypeData(env_val, [],
                                list(m['ctr'] for m in env_val_map.values()))
    mod[env_type] = adt.TypeData(env_type, [], [empty_env, cons_env])
    for m in env_val_map.values():
        for k, v in m.items():
            if k == 'ctr':
                continue
            mod[v[0]] = v[1]


@overload(bootstrap=True)
def to_relay_type(self, a: AbstractScalar):
    """Convert a myia abstract to a Relay type."""
    tp = a.xtype()
    if issubclass(tp, Bool):
        return relay.ty.scalar_type('bool')
    elif issubclass(tp, Nil):
        return relay.ty.TupleType([])
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
def to_relay_type(self, a: PartialApplication):
    tp = self(a.fn)
    return relay.ty.FuncType(tp.arg_types[len(a.args):], tp.ret_type)


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTaggedUnion):
    return union_type()


def _placeholder_body(type):
    if isinstance(type, relay.TensorType):
        sh = [int(sh) for sh in type.shape]
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
        else:
            raise ValueError(f"Can't build value for union: {type.func}")
    else:
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


def setup_env_val(val_t):
    """Build and cache all the functions that go with a type."""
    make_env_update(val_t)


def make_env_update(val_t):
    """Define a function to update a grad env."""
    gv = relay.GlobalVar(f"$_update_env<{val_t}>")

    m = get_env_val_map(val_t)
    upd = m.get('update_env', None)

    if upd is not None:
        return upd

    ctr = m['ctr']

    env = relay.Var("env", env_type())
    key = relay.Var("key", relay.ty.scalar_type('int32'))
    val = relay.Var("val", to_relay_type(val_t))

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
    fn = relay.Function([env, key, val], body, env_type())
    m['update_env'] = (gv, fn)
    return gv, fn


__all__ = [
    'build_module',
    'optimize',
]
