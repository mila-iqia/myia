import pytest

from myia.operations import tagged
from myia.testing.common import Point
from myia.testing.multitest import Multiple, mt, run

run_relay_debug = run.configure(
    backend=Multiple(
        pytest.param(
            ("relay", {"exec_kind": "debug"}),
            id="relay-cpu-debug",
            marks=pytest.mark.relay,
        )
    )
)


@mt(run_relay_debug(0, 1.7, Point(3, 4), (8, 9)),)
def test_tagged(c, x, y, z):
    if c > 0:
        return tagged(x)
    elif c == 0:
        return tagged(y)
    else:
        return tagged(z)
