"""Collection of helpers for the Relay backend.

Most of those should go away as Relay main development progresses.
"""

import numpy as np
from tvm import relay
from tvm.relay import transform, adt

union_type = relay.GlobalTypeVar('$_union_adt')
empty_union = adt.Constructor("c_empty", [], union_type)
tag_map = {} #{None: empty_union}
rev_tag_map = {}


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


__all__ = [
    'build_module',
    'optimize',
]
