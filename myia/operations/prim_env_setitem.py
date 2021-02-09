"""Definitions for the primitive `env_setitem`."""

from .. import xtype
from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import zeros_like
from . import primitives as P


def pyimpl_env_setitem(env, key, x):
    """Implement `env_setitem`."""
    return env.set(key, x)


@standard_prim(P.env_setitem)
async def infer_env_setitem(
    self, engine, env: xtype.EnvType, key: xtype.SymbolicKeyType, value
):
    """Infer the return type of primitive `env_setitem`."""
    expected = key.xvalue().abstract
    engine.abstract_merge(expected, value)
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.EnvType})


@bprop_to_grad_transform(P.env_setitem)
def bprop_env_setitem(env, key, value, out, dout):
    """Back propagator for env_setitem."""
    return (
        P.env_setitem(dout, key, zeros_like(value)),
        zeros_like(key),
        zeros_like(value),
    )


__operation_defaults__ = {
    "name": "env_setitem",
    "registered_name": "env_setitem",
    "mapping": P.env_setitem,
    "python_implementation": pyimpl_env_setitem,
}


__primitive_defaults__ = {
    "name": "env_setitem",
    "registered_name": "env_setitem",
    "type": "backend",
    "python_implementation": pyimpl_env_setitem,
    "inferrer_constructor": infer_env_setitem,
    "grad_transform": bprop_env_setitem,
}
