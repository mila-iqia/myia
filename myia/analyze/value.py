"""Value inference."""
from typing import Callable

from myia.prim import Primitive, ops as P
from myia.utils import Named, Registry

ESTIMATORS: Registry[Primitive, Callable] = Registry()
register_estimator = ESTIMATORS.register


NO_VALUE = Named('NO_VALUE')


@register_estimator(P.mul)
def _est_mul(x, y):
    if x == 0 or y == 0:
        return 0
    else:
        return NO_VALUE
