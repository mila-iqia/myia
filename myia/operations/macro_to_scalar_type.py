"""
Implementation of 'to_scalar_type' operation.

Convert given data into an appropriate abstract scalar.
If data is already an abstract type, return the result of method xvalue().
If data is already an abstract scalar with a number type, return it as is.
If data is an abstract scalar with a string type, try to infer type
from given string and return an abstract scalar with inferred type.
Used in myia/operations/op_full.py
"""
import inspect

import numpy as np

from myia.utils.errors import MyiaTypeError

from ..lib import (
    ANYTHING,
    TYPE,
    VALUE,
    AbstractScalar,
    AbstractType,
    Constant,
    force_pending,
    macro,
)
from ..xtype import Number, String, pytype_to_myiatype


def string_to_np_dtype(string):
    """Convert given string to numpy d-type. Return None if parsing failed."""
    try:
        # If Numpy cannot parse given string, it will raise a TypeError.
        np_dtype = np.dtype(string)
    except TypeError:
        pass
    else:
        # We accept only:
        # - booleans,
        # - signed integers,
        # - unsigned integers,
        # - floating values
        # - complex values.
        if np_dtype.kind in 'biufc':
            return np_dtype.type
    return None


@macro
async def to_scalar_type(info, data):
    """
    Convert given data to abstract scalar.

    :param data: arbitrary data to convert
    :return: an abstract scalar object if data can be converted,
        otherwise raise an exception.
    """
    sync_data = await data.get()

    # We expect either:
    # - an abstract string containing an abstract scalar with scalar type
    # - an abstract scalar containing a string to be parsed to a scalar type

    if isinstance(sync_data, AbstractType):
        abstract_scalar = sync_data.xvalue()
        if isinstance(abstract_scalar, AbstractScalar):
            xtype = await force_pending(abstract_scalar.xtype())
            if inspect.isclass(xtype) and issubclass(xtype, Number):
                return Constant(abstract_scalar)

    elif isinstance(sync_data, AbstractScalar):
        xtype = await force_pending(sync_data.xtype())
        if xtype is String:
            np_dtype = string_to_np_dtype(sync_data.xvalue())
            if np_dtype:
                myia_type = pytype_to_myiatype(np_dtype)
                return Constant(
                    AbstractScalar({VALUE: ANYTHING, TYPE: myia_type}))

    # In any other case, we raise an exception.
    raise MyiaTypeError(
        'Unable to convert data to scalar type: %s' % sync_data)


__operation_defaults__ = {
    'name': 'to_scalar_type',
    'registered_name': 'to_scalar_type',
    'mapping': to_scalar_type,
    'python_implementation': None,
}
