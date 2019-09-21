"""Implementation of the 'is' operation."""

import operator

from ..lib import Constant, MyiaTypeError, macro
from ..xtype import Bool, Nil, NotImplementedType
from . import primitives as P


@macro
async def is_(info, a, b):
    """Implement the is operator."""
    anode, bnode = info.nodes()
    at = (await a.get()).xtype()
    bt = (await b.get()).xtype()
    if at is Nil or bt is Nil:
        return Constant(at is bt)
    elif at is NotImplementedType or bt is NotImplementedType:
        return Constant(at is bt)
    elif at is Bool and bt is Bool:
        return info.graph.apply(P.bool_eq, anode, bnode)
    elif at is Bool or bt is Bool:
        return Constant(False)
    else:
        raise MyiaTypeError(
            f'The operator "is" is only implemented for booleans ' +
            f'and singletons such as None or NotImplemented.'
        )


__operation_defaults__ = {
    'name': 'is',
    'registered_name': 'is_',
    'mapping': is_,
    'python_implementation': operator.is_,
}
