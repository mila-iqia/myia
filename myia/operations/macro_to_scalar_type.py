"""
Implementation of 'to_scalar_type' operation.

Convert given data into an appropriate abstract scalar.
If data is already an abstract type, return the result of method xvalue().
If data is already an abstract scalar with a number type, return it as is.
If data is an abstract scalar with a string type, try to infer type
from given string and return an abstract scalar with inferred type.
Used in myia/operations/op_full.py
"""

import numpy as np

from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    AbstractType,
    Constant,
    macro,
)
from ..xtype import Number, String, pytype_to_myiatype


@macro
async def to_scalar_type(info, data):
    """
    Convert given data to abstract scalar.

    :param data: arbitrary data to convert
    :return: an abstract scalar object if data can be converted,
        otherwise raise an exception.
    """
    sync_data = await data.get()
    if isinstance(sync_data, AbstractType):
        sync_data = sync_data.xvalue()
    if isinstance(sync_data, AbstractScalar):
        xtype = sync_data.xtype()
        if issubclass(xtype, Number):
            return Constant(sync_data)
        elif xtype is String:
            myia_type = pytype_to_myiatype(np.dtype(sync_data.xvalue()).type)
            return Constant(AbstractScalar({VALUE: ANYTHING, TYPE: myia_type}))
    else:
        raise Exception(
            'Unable to convert data to scalar type: %s' % sync_data)


__operation_defaults__ = {
    'name': 'to_scalar_type',
    'registered_name': 'to_scalar_type',
    'mapping': to_scalar_type,
    'python_implementation': None,
}
