"""Implementation of the 'hasattr' operation."""

from .. import lib, operations
from ..lib import Constant, macro
from .macro_getattr import _attr_case, _union_make


@macro
async def hasattr_(info, r_data, r_attr):
    """Check that an object has an attribute."""
    data, attr = await info.abstracts()

    if isinstance(data, lib.AbstractUnion):
        return await _union_make(info, operations.hasattr)

    attr_v = await info.build(r_attr)
    case, *_ = await _attr_case(info, data, attr_v)

    return Constant(case == 'field' or case == 'method')


__operation_defaults__ = {
    'name': 'hasattr',
    'registered_name': 'hasattr',
    'mapping': hasattr_,
    'python_implementation': hasattr,
}
