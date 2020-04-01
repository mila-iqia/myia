"""Abstract Types for All Frontends."""

from ..abstract.data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
)
from ..xtype import Bool

AA_bool = AbstractArray(
    AbstractScalar({TYPE: Bool, VALUE: ANYTHING}),
    {SHAPE: ANYTHING, TYPE: ANYTHING},
)


AS = AbstractScalar({TYPE: ANYTHING, VALUE: ANYTHING})


__all__ = ["AA_bool", "AS"]
