"""Definitions for the primitive `hastype`."""

from .. import lib, xtype
from ..lib import (
    TYPE,
    VALUE,
    AbstractScalar,
    hastype_helper,
    standard_prim,
    type_to_abstract,
)
from . import primitives as P


@standard_prim(P.hastype)
async def infer_hastype(self, engine, value, model: lib.AbstractType):
    """Infer the return type of primitive `hastype`."""
    a = type_to_abstract(model.element)
    return AbstractScalar({
        VALUE: hastype_helper(value, a),
        TYPE: xtype.Bool,
    })


__operation_defaults__ = {
    'name': 'hastype',
    'registered_name': 'hastype',
    'mapping': P.hastype,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'hastype',
    'registered_name': 'hastype',
    'type': 'inference',
    'python_implementation': None,
    'inferrer_constructor': infer_hastype,
    'grad_transform': None,
}
