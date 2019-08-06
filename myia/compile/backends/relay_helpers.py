"""Collection of helpers for the Relay backend."""

from tvm import relay
from tvm.relay import transform
import numpy as np


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


def build_module(funcs):
    """Workaround for type checker and mutually recursive functions."""
    mod = relay.Module({})
    for gv in funcs:
        func = funcs[gv]
        body = _placeholder_body(func.ret_type)
        mod[gv] = relay.Function(func.params, body, func.ret_type)

    for gv in funcs:
        mod[gv] = funcs[gv]

    return mod


pass_set = transform.Sequential(
    passes=[
        transform.SimplifyInference(),
        transform.CanonicalizeOps(),
        transform.CanonicalizeCast(),
        transform.FuseOps(3),
        transform.AlterOpLayout(),
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
