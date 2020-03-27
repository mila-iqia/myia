"""Definitions for the primitive `env_add`."""

from .. import xtype
from ..lib import ANYTHING, TYPE, VALUE, AbstractScalar, standard_prim
from . import primitives as P


def pyimpl_env_add(env1, env2):
    """Implement `env_add`."""
    return env1.add(env2)


@standard_prim(P.env_add)
async def infer_env_add(self, engine, env1, env2):
    """Infer the return type of primitive `env_add`."""
    return AbstractScalar({VALUE: ANYTHING, TYPE: xtype.EnvType})


__operation_defaults__ = {
    "name": "env_add",
    "registered_name": "env_add",
    "mapping": P.env_add,
    "python_implementation": pyimpl_env_add,
}


__primitive_defaults__ = {
    "name": "env_add",
    "registered_name": "env_add",
    "type": "backend",
    "python_implementation": pyimpl_env_add,
    "inferrer_constructor": infer_env_add,
    "grad_transform": None,
}
