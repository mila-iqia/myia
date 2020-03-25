"""Implementation of the 'isinstance' operation."""

from functools import reduce

from .. import lib
from ..lib import Constant, MyiaTypeError, macro, type_to_abstract
from . import primitives as P


@macro
async def isinstance_(info, r_data, r_type):
    """Map isinstance to hastype."""
    ts = await info.build(r_type)
    if not isinstance(ts, tuple):
        ts = (ts,)
    for t in ts:
        if not isinstance(t, lib.AbstractValue):
            raise MyiaTypeError(
                "isinstance expects a Python type" " or a tuple of Python types"
            )
    hastypes = [
        info.graph.apply(P.hastype, r_data.node, Constant(type_to_abstract(t)))
        for t in ts
    ]
    return reduce(lambda x, y: info.graph.apply(P.bool_or, x, y), hastypes)


__operation_defaults__ = {
    "name": "isinstance",
    "registered_name": "isinstance",
    "mapping": isinstance_,
    "python_implementation": isinstance,
}
