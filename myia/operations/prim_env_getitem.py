"""Definitions for the primitive `env_getitem`."""

from .. import xtype
from ..lib import bprop_to_grad_transform, standard_prim
from ..operations import env_setitem, zeros_like
from . import primitives as P


def pyimpl_env_getitem(env, key, default):
    """Implement `env_getitem`."""
    return env.get(key, default)


@standard_prim(P.env_getitem)
async def infer_env_getitem(
    self, engine, env: xtype.EnvType, key: xtype.SymbolicKeyType, dflt
):
    """Infer the return type of primitive `env_getitem`."""
    expected = key.xvalue().abstract
    engine.abstract_merge(expected, dflt)
    return expected


@bprop_to_grad_transform(P.env_getitem)
def bprop_env_getitem(env, key, default, out, dout):
    """Back propagator for env_getitem."""
    return (
        env_setitem(zeros_like(env), key, dout),
        zeros_like(key),
        zeros_like(default),
    )


__operation_defaults__ = {
    "name": "env_getitem",
    "registered_name": "env_getitem",
    "mapping": P.env_getitem,
    "python_implementation": pyimpl_env_getitem,
}


__primitive_defaults__ = {
    "name": "env_getitem",
    "registered_name": "env_getitem",
    "type": "backend",
    "python_implementation": pyimpl_env_getitem,
    "inferrer_constructor": infer_env_getitem,
    "grad_transform": bprop_env_getitem,
}
