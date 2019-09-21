"""Definitions for the primitive `record_setitem`."""

from .. import lib, xtype
from ..lib import MyiaAttributeError, MyiaTypeError, standard_prim, typecheck
from . import primitives as P


@standard_prim(P.record_setitem)
async def infer_record_setitem(self, engine,
                               data: lib.AbstractClassBase,
                               attr: xtype.String,
                               value):
    """Infer the return type of primitive `record_setitem`."""
    attr_v = self.require_constant(attr, argnum=2)
    if attr_v not in data.attributes:
        raise MyiaAttributeError(f'Unknown field in {data}: {attr_v}')
    model = data.user_defined_version()
    expected = model.attributes[attr_v]
    if not typecheck(expected, value):
        raise MyiaTypeError(f'Expected field {attr_v} to have type {expected}')
    return type(data)(
        data.tag,
        {**data.attributes, attr_v: value},
        constructor=data.constructor
    )


__operation_defaults__ = {
    'name': 'record_setitem',
    'registered_name': 'record_setitem',
    'mapping': P.record_setitem,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'record_setitem',
    'registered_name': 'record_setitem',
    'type': 'inference',
    'python_implementation': None,
    'inferrer_constructor': infer_record_setitem,
    'grad_transform': None,
}
