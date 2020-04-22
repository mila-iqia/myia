"""Edge cases that may pop up but it's not clear where else to test them."""

from dataclasses import dataclass

import numpy as np
import pytest

from myia import myia
from myia.lib import core
from myia.operations import array_map
from myia.pipeline import standard_pipeline, steps

from .multitest import eqtest

pipeline2 = standard_pipeline.insert_after("parse", resolve=steps.step_resolve)


def test_static_inline_array_map():
    @core(static_inline=True)
    def inl(x, y):
        return x + y

    @myia
    def f(xs, ys):
        return array_map(inl, xs, ys)

    assert eqtest(f(np.ones((2, 2)), np.ones((2, 2))), 2 * np.ones((2, 2)))


def test_call_opdef():
    from myia.operations.utils import to_opdef

    @to_opdef
    def f(x, y):
        return x + y

    with pytest.raises(TypeError):
        f(1, 2)

    @myia(pipeline=pipeline2)
    def g(x, y):
        return f(x, y)

    assert g(1, 2) == 3


@dataclass(frozen=True)
class Linear:
    """Linear layer."""

    W: object

    def apply(self):
        """Apply the layer."""
        return self.W
