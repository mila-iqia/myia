import io
import math
import operator
import sys

import pytest

from myia.compile.backends.python.python import compile_graph
from myia.parser import parse
from myia.parser_opt import apply_parser_opts
from myia.utils.info import enable_debug


def parse_and_compile(function, debug=True, parser_opt=False, optimize=False):
    if debug:
        with enable_debug():
            graph = parse(function)
    else:
        graph = parse(function)
    if parser_opt:
        graph = apply_parser_opts(graph)
    output = io.StringIO()
    fn = compile_graph(graph, debug=output, optimize=optimize)
    output = output.getvalue()
    print()
    print(output)
    return fn, output


# NB: Need to be global for test_recursion to work.
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


def test_nonlocal_handle():
    def f():
        a = 0

        def g():
            nonlocal a
            a = 1

        g()
        return a

    fn, output = parse_and_compile(f)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f():
  a = make_handle(None)
  _1 = global_universe_setitem(a, 0)

  def g():
    _2 = global_universe_setitem(a, 1)
    return None

  _3 = g()
  return global_universe_getitem(a)
"""
    )
    assert f() == 1


def test_nonlocal_handle_optimized():
    # NB: Graph contains 2 calls to universe_setitem on same make_handle, so parser opt can't optimize graph.
    # Backend opt can still remove universe usages.

    def f():
        a = 0

        def g():
            nonlocal a
            a = 1

        g()
        return a

    fn, output = parse_and_compile(f, optimize=True)
    assert (
        output
        == """def f():
  a = 0

  def g():
    nonlocal a
    a = 1
    return None

  _1 = g()
  return a
"""
    )
    assert f() == 1


def test_operations():
    def f(x, y):
        a = x + y
        b = x - y
        c = 2 * x + 3 * a - 5 * b
        return c + b + y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x, y):
  a = x + y
  b = x - y
  _1 = 2 * x
  _2 = 3 * a
  _3 = _1 + _2
  _4 = 5 * b
  c = _3 - _4
  _5 = c + b
  return _5 + y
"""
    )
    assert f(1, 2) == fn(1, 2)


def test_simple():
    def f(x):
        return x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  return x
"""
    )
    assert f(-1) == fn(-1)
    assert f((2, 3)) == fn((2, 3))


def test_add():
    def f(x, y):
        return x + y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x, y):
  return x + y
"""
    )
    assert f(3, 4) == fn(3, 4) == 7


def test_seq():
    def f(x):
        x = x + 1
        return 0

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _x_2 = x + 1
  return 0
"""
    )
    assert f(11) == fn(11) == 0


def test_seq2():
    def f(x):
        return x + x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  return x + x
"""
    )
    assert f(11) == fn(11) == 22


def test_compare():
    def f(x):
        return x > 0

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  return x > 0
"""
    )
    assert f(10) is fn(10) is True
    assert f(-5) is fn(-5) is False


def test_unary():
    def f(x):
        return -x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  return -x
"""
    )
    assert f(2) == fn(2) == -2
    assert f(-3) == fn(-3) == 3


def test_ifexp():
    def f(x, y, b):
        return x if b else y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x, y, b):
  _1 = bool(b)

  def if_false_f(phi_x, phi_y):
    return phi_y

  def if_true_f(_phi_x_2, _phi_y_2):
    return _phi_x_2

  _2 = if_true_f if _1 else if_false_f
  return _2(x, y)
"""
    )
    assert f(2, 3, 0) == fn(2, 3, 0) == 3
    assert f(2, 3, 1) == fn(2, 3, 1) == 2


def test_boolop():
    def f(a, b, c):
        return a and b or c

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(a, b, c):
  _1 = bool(a)

  def if_false_f(phi_b):
    return False

  def if_true_f(_phi_b_2):
    return _phi_b_2

  _2 = if_true_f if _1 else if_false_f
  _3 = _2(b)
  _4 = bool(_3)

  def _if_false_f_2(phi_c):
    return phi_c

  def _if_true_f_2(_phi_c_2):
    return True

  _5 = _if_true_f_2 if _4 else _if_false_f_2
  return _5(c)
"""
    )
    assert f(1, 2, 3) == 2 and fn(1, 2, 3) is True
    assert f(1, 0, 4) == fn(1, 0, 4) == 4


def test_compare2():
    def f(x):
        return 0 < x < 42

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _1 = 0 < x
  _2 = bool(_1)

  def if_false_f(phi_x):
    return False

  def if_true_f(_phi_x_2):
    return _phi_x_2 < 42

  _3 = if_true_f if _2 else if_false_f
  return _3(x)
"""
    )
    assert f(1) is fn(1) is True
    assert f(30) is fn(30) is True
    assert f(0) is fn(0) is False
    assert f(100) is fn(100) is False


def test_if():
    def f(b, x, y):
        if b:
            return x
        else:
            return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  _1 = bool(b)

  def if_false_f(phi_x, phi_y):
    return phi_y

  def if_true_f(_phi_x_2, _phi_y_2):
    return _phi_x_2

  _2 = if_true_f if _1 else if_false_f
  return _2(x, y)
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(5, 1, 2) == fn(5, 1, 2) == 1


def test_if2():
    def f(b, x, y):
        if b:
            return x
        return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  def if_after_f(phi_y):
    return phi_y

  _1 = bool(b)

  def if_false_f(phi_x, _phi_y_2):
    return if_after_f(_phi_y_2)

  def if_true_f(_phi_x_2, _phi_y_3):
    return _phi_x_2

  _2 = if_true_f if _1 else if_false_f
  return _2(x, y)
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_while():
    def f(b, x, y):
        while b:
            return x
        return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  def while_f(phi_b, phi_x, phi_y):
    def while_after_f(_phi_y_2):
      return _phi_y_2

    def else_while_f(_phi_x_2, _phi_y_3):
      return while_after_f(_phi_y_3)

    def body_while_f(_phi_x_3, _phi_y_4):
      return _phi_x_3

    _1 = body_while_f if phi_b else else_while_f
    return _1(phi_x, phi_y)

  return while_f(b, x, y)
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_while2():
    def f(x):
        while x:
            x = x - 1
        return x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  def while_f(phi_x):
    def while_after_f(_phi_x_2):
      return _phi_x_2

    def else_while_f(_phi_x_3):
      return while_after_f(_phi_x_3)

    def body_while_f(_phi_x_4):
      _x_2 = _phi_x_4 - 1
      return while_f(_x_2)

    _1 = body_while_f if phi_x else else_while_f
    return _1(phi_x)

  return while_f(x)
"""
    )
    assert f(0) == fn(0) == 0
    assert f(100) == fn(100) == 0


def test_recursion():
    fn, output = parse_and_compile(factorial)
    assert output == (
        """from test_code_format import factorial

def factorial(n):
  _1 = n < 2
  _2 = bool(_1)

  def if_false_factorial(phi_n):
    _3 = phi_n - 1
    _4 = factorial(_3)
    return phi_n * _4

  def if_true_factorial(_phi_n_2):
    return 1

  _5 = if_true_factorial if _2 else if_false_factorial
  return _5(n)
"""
    )
    assert fn(5) == 120


def test_no_debug():
    def f(x):
        val = 10
        if x % 2 == 0:
            a = 0
            for i in range(x):
                a = a + i
        else:
            a = x ** 2 + 2 * x + 1
        return factorial(x) - a + val

    fn, output = parse_and_compile(f, debug=False)

    assert output == (
        """from test_code_format import factorial
from myia.basics import myia_iter
from myia.basics import myia_hasnext
from myia.basics import myia_next

def _1(_2):
  _3 = _2 % 2
  _4 = _3 == 0

  def _5(_6, _7, _8):
    _9 = factorial(_6)
    _10 = _9 - _7
    return _10 + _8

  _11 = bool(_4)

  def _12(_13, _14):
    _15 = _13 ** 2
    _16 = 2 * _13
    _17 = _15 + _16
    _18 = _17 + 1
    return _5(_13, _18, _14)

  def _19(_20, _21):
    _22 = range(_20)

    def _23(_24, _25, _26):
      return _5(_24, _25, _26)

    _27 = myia_iter(_22)

    def _28(_29, _30, _31, _32):
      _33 = myia_hasnext(_29)

      def _34(_35, _36, _37):
        return _23(_35, _36, _37)

      def _38(_39, _40, _41):
        _42 = myia_next(_29)
        _43 = _42[0]
        _44 = _42[1]
        _45 = _40 + _43
        return _28(_44, _39, _45, _41)

      _46 = _38 if _33 else _34
      return _46(_30, _31, _32)

    return _28(_27, _20, 0, _21)

  _47 = _19 if _11 else _12
  return _47(_2, 10)
"""
    )

    assert fn(1) == 7
    assert fn(5) == 94
    assert fn(10) == 3628765


def test_constants():
    def f():
        a = 3
        b = -2.33
        c = -1.44e-9
        d = True
        e = "a string"
        g = (5, 7.7, -1, False)
        h = [5, 7.7, -1, False]
        i = {}
        j = {"a": "1", True: 2}
        k = dict()
        m = dict(a=1, b=2)
        n = operator.add
        p = math.sin
        return a, b, c, d, e, g, h, i, j, k, m, n, p

    fn, output = parse_and_compile(f)
    assert (
        output
        == """import operator
import math

def f():
  b = -2.33
  c = -1.44e-09
  _1 = -1
  g = (5, 7.7, _1, False)
  _2 = -1
  h = [5, 7.7, _2, False]
  i = {}
  j = {'a': '1', True: 2}
  k = dict()
  _3 = {'a': 1, 'b': 2}
  _4 = ()
  m = dict(*_4, **_3)
  n = operator.add
  p = math.sin
  return (3, b, c, True, 'a string', g, h, i, j, k, m, n, p)
"""
    )
    assert fn()[4] == "a string"


def test_print():
    def f(x):
        print("X is", x)
        return x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _1 = print('X is', x)
  return x
"""
    )
    default_buf = sys.stdout
    sys.stdout = io.StringIO()
    assert fn(2) == 2
    output = sys.stdout.getvalue()
    sys.stdout = default_buf
    assert output == "X is 2\n"


def test_if_with_constant_strings():
    # Test if inline return correctly handles literal strings.

    def f(x):
        return "morning" if x < 12 else "evening"

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _1 = x < 12
  _2 = bool(_1)

  def if_false_f():
    return 'evening'

  def if_true_f():
    return 'morning'

  _3 = if_true_f if _2 else if_false_f
  return _3()
"""
    )
    assert fn(2) == "morning"
    assert fn(15) == "evening"


def test_inline_operators_with_string():
    def f(x):
        return x + ", world!"

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  return x + ', world!'
"""
    )
    assert fn("Hello") == "Hello, world!"


def test_universe_on_string():
    # NB: Parser opt is enough to remove universe usages. No need of backend opt.

    def f():
        x = "hello"

        def g():
            return x

        return g()

    fn, output = parse_and_compile(f)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f():
  x = make_handle(None)
  _1 = global_universe_setitem(x, 'hello')

  def g():
    return global_universe_getitem(x)

  return g()
"""
    )
    assert fn() == "hello"


def test_universe_on_string_parser_opt():
    # NB: Parser opt is enough to remove universe usages. No need of backend opt.

    def f():
        x = "hello"

        def g():
            return x

        return g()

    fn, output = parse_and_compile(f, parser_opt=True)
    assert (
        output
        == """def f():
  def g():
    return 'hello'

  return g()
"""
    )
    assert fn() == "hello"


def test_no_return():
    # NB: Parser opt is enough to remove universe usages. No need of backend opt.

    def f(x):
        y = 2 * x

        def g(i):
            j = i + x + y  # noqa: F841

        z = g(0)  # noqa: F841

    fn, output = parse_and_compile(f)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(x):
  y = make_handle(None)
  _x_2 = make_handle(None)
  _1 = global_universe_setitem(_x_2, x)
  _2 = global_universe_getitem(_x_2)
  _3 = 2 * _2
  _4 = global_universe_setitem(y, _3)

  def g(i):
    _5 = global_universe_getitem(_x_2)
    _6 = i + _5
    _7 = global_universe_getitem(y)
    j = _6 + _7
    return None

  z = g(0)
  return None
"""
    )
    assert fn(1) is None


def test_no_return_parser_opt():
    # NB: Parser opt is enough to remove universe usages. No need of backend opt.

    def f(x):
        y = 2 * x

        def g(i):
            j = i + x + y  # noqa: F841

        z = g(0)  # noqa: F841

    fn, output = parse_and_compile(f, parser_opt=True)
    assert (
        output
        == """def f(x):
  _1 = 2 * x

  def g(i):
    _2 = i + x
    j = _2 + _1
    return None

  z = g(0)
  return None
"""
    )
    assert fn(1) is None


thingy = 10


def test_global_integer():
    def f(x):
        return x * thingy

    fn, output = parse_and_compile(f)
    assert (
        output
        == """# Dynamic external import: thingy

def f(x):
  return x * thingy
"""
    )
    assert fn(2) == 20


def test_varargs_kwargs():
    # Test function header code with complex parameter list.
    # We expect parameter list in generated function code to be
    # the same as in original function.

    def f(a, b, *args, c, d, **kwargs):
        return {"a": a, "b": b, "args": args, "c": c, "d": d, "kwargs": kwargs}

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(a, b, *args, c, d, **kwargs):
  return {'a': a, 'b': b, 'args': args, 'c': c, 'd': d, 'kwargs': kwargs}
"""
    )
    assert fn(1, 2, 3, 4, 5, d=9, c=11, x=33) == {
        "a": 1,
        "b": 2,
        "args": (3, 4, 5),
        "c": 11,
        "d": 9,
        "kwargs": {"x": 33},
    }


def test_posonly_args():
    def f(a, b, c, /, d, e):
        return a + b + c, d - e

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(a, b, c, /, d, e):
  _1 = a + b
  _2 = _1 + c
  _3 = d - e
  return (_2, _3)
"""
    )
    assert fn(1, 2, 3, e=-1, d=-2) == (6, -1)


def test_kwonly_args():
    def f(a, b, *, c, d):
        return a + b, c + 10 * d

    fn, output = parse_and_compile(f)

    # Fn call should fail if we don't pass keyword arguments.
    with pytest.raises(
        TypeError, match="takes 2 positional arguments but 4 were given"
    ):
        fn(2, 3, 4, 5)

    assert fn(2, 3, d=5, c=4) == (5, 54)
