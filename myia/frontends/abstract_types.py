"""Abstract Types for All Frontends."""

from ..abstract.data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
)
from ..xtype import Bool, NDArray

AA_bool = AbstractArray(
    AbstractScalar({TYPE: Bool, VALUE: ANYTHING}),
    {SHAPE: ANYTHING, TYPE: ANYTHING},
)


AS = AbstractScalar({TYPE: ANYTHING, VALUE: ANYTHING})


AA = AbstractArray(ANYTHING, {SHAPE: ANYTHING, TYPE: NDArray})


__all__ = ["AA_bool", "AS", "AA"]
