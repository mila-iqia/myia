
from myia.operations import ioprint, print as myia_print
from myia.lib import new_universe
from myia.operations.op_print import print_handle
from myia.xtype import UniverseType

from ..common import i64, f64
from ..multitest import infer, mt, run_debug


@mt(
    infer(i64, i64, result=i64),
    infer(i64, (f64, f64), result=i64),

    run_debug(0, (8, 9, 10), result=1),
)
def test_ioprint(iostate, obj):
    return ioprint(iostate, obj)



def _test_print_check(args, res):
    U0, *args = args
    U1, x = res
    s0 = print_handle.state
    assert x is None
    assert U1.get(print_handle) == s0 + len(args)
    return True


@mt(
    infer(UniverseType, i64, i64, result=(UniverseType, None)),
    infer(UniverseType, i64, i64, f64, result=(UniverseType, None)),

    run_debug(new_universe, 66, 89, 41.5, result=_test_print_check)
)
def test_print(U, *entries):
    return myia_print(U, *entries)
