"""Collection of helpers for the Relay backend.

Most of those should go away as Relay main development progresses.
"""

import numpy as np
from tvm import relay
from tvm.relay import adt
from tvm.runtime.object import Object

from ...abstract import (
    AbstractArray,
    AbstractError,
    AbstractHandle,
    AbstractRandomState,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractType,
    TypedPrimitive,
    VirtualFunction2,
)
from ...operations import primitives as P
from ...utils import overload
from ...xtype import Bool, EnvType, Nil, UniverseType, type_to_np_dtype

union_type = relay.GlobalTypeVar("$_union_adt")
empty_union = adt.Constructor("empty", [], union_type)
tag_map = {None: empty_union}
rev_tag_map = {}


def get_union_ctr(tag, t):
    """Get the relay constructor for a union tag."""
    if tag not in tag_map:
        assert t is not None
        rt = to_relay_type(t)
        ctr = adt.Constructor(f"c{tag}", [rt], union_type)
        tag_map[tag] = ctr
    return tag_map[tag]


def fill_reverse_tag_map():
    """Fill the back-conversion map.

    Do this after a compilation step involving the constructors you
    need since the tags are not set otherwise.
    """
    for tag, ctr in tag_map.items():
        if ctr.tag != -1:
            rev_tag_map[ctr.tag] = tag


def get_myia_tag(rtag):
    """Return the myia tag for a constructor.

    This will fail if you haven't properly called fill_reverse_tag_map().
    """
    return rev_tag_map[rtag]


option_type = relay.GlobalTypeVar("_option")
a = relay.ty.TypeVar("a")
nil = adt.Constructor("None", [], option_type)
some = adt.Constructor("Some", [a], option_type)
env_type = relay.GlobalTypeVar("env_type")
dead_env = adt.Constructor("DeadEnv", [], env_type)


class TypeHelper:
    """Class to help manage and generate helper types."""

    def __init__(self):
        """Initialize the caches."""
        self.env_val_map = {}

    def initialize(self, mod, mng):
        """Add types to the module."""
        if mng is not None:
            for node in mng.all_nodes:
                if isinstance(node.abstract, AbstractTaggedUnion):
                    for opt in node.abstract.options:
                        get_union_ctr(*opt)
                elif node.is_apply(P.env_setitem):
                    key = node.inputs[2]
                    tt = to_relay_type(node.inputs[3].abstract)
                    assert key.is_constant()
                    self.env_val_map[key.value] = tt
        env_val_keys = sorted(list(self.env_val_map.keys()))

        for i, k in enumerate(env_val_keys):
            self.env_val_map[k] = (i, self.env_val_map[k])

        mod[union_type] = adt.TypeData(union_type, [], list(tag_map.values()))
        mod[option_type] = adt.TypeData(option_type, [a], [nil, some])
        self.env_ctr = adt.Constructor("v", [self._build_env_type()], env_type)
        mod[env_type] = adt.TypeData(env_type, [], [self.env_ctr, dead_env])

    def build_default_env_val(self):
        """Build the default value for env, which is empty."""
        return self.env_ctr(relay.Tuple([nil() for _ in self.env_val_map]))

    def _build_env_type(self):
        map = dict((i, tt) for (i, tt) in self.env_val_map.values())

        return relay.ty.TupleType(
            [option_type(map[i]) for i in range(len(map))]
        )

    def do_env_update(self, env_, key, val):
        """Build the code to update the env."""
        v = relay.var("v")
        cl = adt.Clause(
            adt.PatternConstructor(self.env_ctr, [adt.PatternVar(v)]), v
        )
        env = adt.Match(env_, [cl], complete=False)

        map = dict((i, k) for k, (i, _) in self.env_val_map.items())
        new_env = relay.Tuple(
            [
                some(val) if map[i] == key else relay.TupleGetItem(env, i)
                for i in range(len(map))
            ]
        )
        return self.env_ctr(new_env)

    def do_env_find(self, env, key, dft):
        """Build the code to find a value in env."""
        v = relay.var("v")
        cl = adt.Clause(
            adt.PatternConstructor(self.env_ctr, [adt.PatternVar(v)]), v
        )
        env_v = adt.Match(env, [cl], complete=False)

        val = relay.TupleGetItem(env_v, self.env_val_map[key][0])
        x = relay.var("x")
        nil_c = adt.Clause(adt.PatternConstructor(nil, []), dft)
        some_c = adt.Clause(
            adt.PatternConstructor(some, [adt.PatternVar(x)]), x
        )
        return adt.Match(val, [some_c, nil_c])


@overload(bootstrap=True)
def to_relay_type(self, a: AbstractScalar):
    """Convert a myia abstract to a Relay type."""
    tp = a.xtype()
    if issubclass(tp, Bool):
        return relay.ty.scalar_type("bool")
    elif issubclass(tp, Nil):
        return relay.ty.TupleType([])
    elif issubclass(tp, EnvType):
        return env_type()
    elif issubclass(tp, UniverseType):
        return relay.ty.TupleType([])
    else:
        return relay.ty.scalar_type(type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractRandomState):
    return relay.ty.TupleType(
        [
            # key
            relay.ty.scalar_type("uint32"),
            # counter
            relay.ty.scalar_type("uint32"),
        ]
    )


@overload  # noqa: F811
def to_relay_type(self, a: AbstractType):
    # Abstract types are not currently used in the graph,
    # they are replaced with other calls,
    # and appear here just as unused graph parameters.
    # So, let's just replace them with an integer type as placeholder.
    return relay.ty.scalar_type("int32")


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTuple):
    return relay.ty.TupleType([self(e) for e in a.elements])


@overload  # noqa: F811
def to_relay_type(self, a: AbstractArray):
    tp = a.element.xtype()
    return relay.ty.TensorType(a.xshape(), type_to_np_dtype(tp))


@overload  # noqa: F811
def to_relay_type(self, a: (VirtualFunction2, TypedPrimitive)):
    return relay.ty.FuncType([self(aa) for aa in a.args], self(a.output))


@overload  # noqa: F811
def to_relay_type(self, a: AbstractTaggedUnion):
    return union_type()


@overload  # noqa: F811
def to_relay_type(self, a: AbstractHandle):
    return relay.ty.RefType(self(a.element))


def dead_value(t):
    """Make a value of the specified type."""
    assert not isinstance(t, AbstractError)
    return _placeholder_body(to_relay_type(t))


def handle_wrapper(fn, handle_params):
    """Wraps a model function to perform handle updates."""

    def wrapper(*args):
        handle_instances = list(args[i] for i in handle_params)
        res = fn(*args)
        u = res[0]
        res = res[1] if len(res) == 2 else res[1:]
        for h, v in zip(handle_instances, u):
            h.value = v
        return (), res

    if len(handle_params) == 0:
        return fn
    else:
        return wrapper


def _placeholder_body(type):
    if isinstance(type, relay.TensorType):
        sh = [sh.value for sh in type.shape]
        return relay.const(
            np.array(np.random.rand(*sh)).astype(type.dtype), dtype=type.dtype
        )
    elif isinstance(type, relay.TupleType):
        return relay.Tuple([_placeholder_body(f) for f in type.fields])
    elif isinstance(type, relay.FuncType):
        params = []
        for arg_ty in type.arg_types:
            params.append(relay.var("p", type_annotation=arg_ty))

        return relay.Function(
            params, _placeholder_body(type.ret_type), ret_type=type.ret_type
        )
    elif isinstance(type, relay.RefType):
        return relay.RefCreate(_placeholder_body(type.value))
    elif isinstance(type, Object):
        if type.func == union_type:
            return empty_union()
        elif type.func == env_type:
            return dead_env()
        else:  # pragma: no cover
            raise ValueError(f"Can't build value for adt: {type.func}")
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


__all__ = [
    "TypeHelper",
    "add_functions",
    "dead_value",
    "fill_reverse_tag_map",
    "get_myia_tag",
    "get_union_ctr",
    "to_relay_type",
]
