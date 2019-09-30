"""Edge cases that may pop up but it's not clear where else to test them."""

import numpy as np

from myia import myia
from myia.lib import core
from myia.operations import array_map

from .multitest import eqtest


def test_static_inline_array_map():
    @core(static_inline=True)
    def inl(x, y):
        return x + y

    @myia
    def f(xs, ys):
        return array_map(inl, xs, ys)

    assert eqtest(f(np.ones((2, 2)), np.ones((2, 2))), 2 * np.ones((2, 2)))
