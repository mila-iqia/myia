"""Definitions for the primitive `env_getitem`."""

from .. import xtype
from ..lib import standard_prim
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
    "grad_transform": None,
}
