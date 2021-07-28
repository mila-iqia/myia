"""Supplementary inferrers for testing.

Help to test master operations.
Inferrers might be moved to standar inferrers later if correctly tested.
"""

import math
from myia import inferrers
from myia.infer.inferrers import X
from myia.infer.infnode import dispatch_inferences, signature
from myia.testing.common import Float, Int, Nil, Object, tuple_of
from myia.testing.master_placeholders import tuple_setitem


def add_testing_inferrers():
    """Add supplementary inferrers for testing."""
    inferrers.update(
        {
            # master operation inferrers
            tuple_setitem: signature(tuple_of(), Int, X, ret=tuple_of()),
            math.log: dispatch_inferences(
                (bool, Float), (Int, Float), (Float, float)
            ),
            # type inferrers
            object: signature(ret=Object),
            type(None): signature(ret=Nil),
        }
    )
