import io

import pytest
import operator
import math

from myia.compile.backends.python.python import compile_graph
from myia.ir.print import str_graph
from myia.parser import parse
from myia.utils.info import enable_debug


def parse_and_compile(function, debug=True):
    if debug:
        with enable_debug():
            graph = parse(function)
    else:
        graph = parse(function)
    print(str_graph(graph))
    output = io.StringIO()
    fn = compile_graph(graph, debug=output)
    output = output.getvalue()
    print()
    print(output)
    return fn, output


# NB: Need to be global for test_recursion to work.
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


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
  _apply1 = 2 * x
  _apply2 = 3 * a
  _apply3 = _apply1 + _apply2
  _apply4 = 5 * b
  c = _apply3 - _apply4
  _apply5 = c + b
  return _apply5 + y
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(x, y, b):
  _apply1 = typeof(x)
  _apply2 = typeof(y)
  _x_2 = make_handle(_apply1)
  _y_2 = make_handle(_apply2)
  _apply3 = universe_setitem(_x_2, x)
  _apply4 = universe_setitem(_y_2, y)
  _apply5 = bool(b)

  def if_false_f():
    return universe_getitem(_y_2)

  def if_true_f():
    return universe_getitem(_x_2)

  _apply6 = if_true_f if _apply5 else if_false_f
  return _apply6()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(a, b, c):
  _apply1 = typeof(b)
  _apply2 = typeof(c)
  _b_2 = make_handle(_apply1)
  _c_2 = make_handle(_apply2)
  _apply3 = universe_setitem(_b_2, b)
  _apply4 = universe_setitem(_c_2, c)
  _apply5 = bool(a)

  def if_false_f():
    return False

  def if_true_f():
    return universe_getitem(_b_2)

  _apply6 = if_true_f if _apply5 else if_false_f
  _apply7 = _apply6()
  _apply8 = bool(_apply7)

  def _if_false_f_2():
    return universe_getitem(_c_2)

  def _if_true_f_2():
    return True

  _apply9 = _if_true_f_2 if _apply8 else _if_false_f_2
  return _apply9()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(x):
  _apply1 = typeof(x)
  _x_2 = make_handle(_apply1)
  _apply2 = universe_setitem(_x_2, x)
  _apply3 = universe_getitem(_x_2)
  _apply4 = 0 < _apply3
  _apply5 = bool(_apply4)

  def if_false_f():
    return False

  def if_true_f():
    _apply6 = universe_getitem(_x_2)
    return _apply6 < 42

  _apply7 = if_true_f if _apply5 else if_false_f
  return _apply7()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(b, x, y):
  _apply1 = typeof(x)
  _apply2 = typeof(y)
  _x_2 = make_handle(_apply1)
  _y_2 = make_handle(_apply2)
  _apply3 = universe_setitem(_x_2, x)
  _apply4 = universe_setitem(_y_2, y)
  _apply5 = bool(b)

  def if_false_f():
    return universe_getitem(_y_2)

  def if_true_f():
    return universe_getitem(_x_2)

  _apply6 = if_true_f if _apply5 else if_false_f
  return _apply6()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(b, x, y):
  _apply1 = typeof(x)
  _apply2 = typeof(y)
  _x_2 = make_handle(_apply1)
  _y_2 = make_handle(_apply2)
  _apply3 = universe_setitem(_x_2, x)
  _apply4 = universe_setitem(_y_2, y)

  def if_after():
    return universe_getitem(_y_2)

  _apply5 = bool(b)

  def if_false_f():
    return if_after()

  def if_true_f():
    return universe_getitem(_x_2)

  _apply6 = if_true_f if _apply5 else if_false_f
  return _apply6()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(b, x, y):
  _apply1 = typeof(b)
  _apply2 = typeof(x)
  _b_2 = make_handle(_apply1)
  _apply3 = typeof(y)
  _x_2 = make_handle(_apply2)
  _apply4 = universe_setitem(_b_2, b)
  _y_2 = make_handle(_apply3)
  _apply5 = universe_setitem(_x_2, x)
  _apply6 = universe_setitem(_y_2, y)

  def while_header():
    def while_after():
      return universe_getitem(_y_2)

    _apply7 = universe_getitem(_b_2)

    def while_else():
      return while_after()

    def while_body():
      return universe_getitem(_x_2)

    _apply8 = while_body if _apply7 else while_else
    return _apply8()

  return while_header()
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
        == """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem

def f(x):
  _apply1 = typeof(x)
  _x_2 = make_handle(_apply1)
  _apply2 = universe_setitem(_x_2, x)

  def while_header():
    def while_after():
      return universe_getitem(_x_2)

    _apply3 = universe_getitem(_x_2)

    def while_else():
      return while_after()

    def while_body():
      _apply4 = universe_getitem(_x_2)
      _apply5 = _apply4 - 1
      _apply6 = universe_setitem(_x_2, _apply5)
      return while_header()

    _apply7 = while_body if _apply3 else while_else
    return _apply7()

  return while_header()
"""
    )
    assert f(0) == fn(0) == 0
    assert f(100) == fn(100) == 0


def test_recursion():
    fn, output = parse_and_compile(factorial)
    assert output == (
        """# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem
# Dynamic external import: factorial

def factorial(n):
  _apply1 = typeof(n)
  _n_2 = make_handle(_apply1)
  _apply2 = universe_setitem(_n_2, n)
  _apply3 = universe_getitem(_n_2)
  _apply4 = _apply3 < 2
  _apply5 = bool(_apply4)

  def if_false_factorial():
    _apply6 = universe_getitem(_n_2)
    _apply7 = universe_getitem(_n_2)
    _apply8 = _apply7 - 1
    _apply9 = factorial(_apply8)
    return _apply6 * _apply9

  def if_true_factorial():
    return 1

  _apply10 = if_true_factorial if _apply5 else if_false_factorial
  return _apply10()
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

    assert output == ("""# Dynamic external import: typeof
# Dynamic external import: make_handle
# Dynamic external import: universe_setitem
# Dynamic external import: universe_getitem
# Dynamic external import: factorial
# Dynamic external import: python_iter
# Dynamic external import: python_hasnext
# Dynamic external import: python_next

def _graph1(_parameter1):
  _apply1 = typeof(_parameter1)
  _apply2 = typeof(10)
  _apply3 = make_handle(_apply1)
  _apply4 = make_handle(_apply2)
  _apply5 = universe_setitem(_apply3, _parameter1)
  _apply6 = universe_setitem(_apply4, 10)
  _apply7 = universe_getitem(_apply3)
  _apply8 = _apply7 % 2
  _apply9 = _apply8 == 0

  def _graph2():
    _apply10 = typeof(0)
    _apply11 = universe_getitem(_apply3)
    _apply12 = make_handle(_apply10)
    _apply13 = factorial(_apply11)
    _apply14 = universe_getitem(_apply12)
    _apply15 = _apply13 - _apply14
    _apply16 = universe_getitem(_apply4)
    return _apply15 + _apply16

  _apply17 = bool(_apply9)

  def _graph3():
    _apply18 = universe_getitem(_apply3)
    _apply19 = _apply18 ** 2
    _apply20 = universe_getitem(_apply3)
    _apply21 = 2 * _apply20
    _apply10 = typeof(0)
    _apply22 = _apply19 + _apply21
    _apply12 = make_handle(_apply10)
    _apply23 = _apply22 + 1
    _apply24 = universe_setitem(_apply12, _apply23)
    return _graph2()

  def _graph4():
    _apply10 = typeof(0)
    _apply12 = make_handle(_apply10)
    _apply25 = universe_setitem(_apply12, 0)
    _apply26 = universe_getitem(_apply3)
    _apply27 = range(_apply26)

    def _graph5():
      return _graph2()

    _apply28 = python_iter(_apply27)

    def _graph6(_parameter2):
      _apply29 = python_hasnext(_parameter2)

      def _graph7():
        return _graph5()

      def _graph8():
        _apply30 = python_next(_parameter2)
        _apply31 = _apply30[0]
        _apply32 = _apply30[1]
        _apply33 = universe_getitem(_apply12)
        _apply34 = _apply33 + _apply31
        _apply35 = universe_setitem(_apply12, _apply34)
        return _graph6(_apply32)

      _apply36 = _graph8 if _apply29 else _graph7
      return _apply36()

    return _graph6(_apply28)

  _apply37 = _graph4 if _apply17 else _graph3
  return _apply37()
""")

    # TODO There is currently a bug with universe_getitem called with an unreachable key.
    # Thus, compiled code can't currently run.
    # assert fn(1) == 7
    # assert fn(5) == 94
    # assert fn(10) == 3628765


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
        l = dict(a=1, b=2)
        m = operator.add
        n = math.sin
        return a, b, c, d, e, g, h, i, j, k, l, m, n

    fn, output = parse_and_compile(f)
    assert output == """# Dynamic external import: operator
# Dynamic external import: math

def f():
  b = -2.33
  c = -1.44e-09
  _apply1 = -1
  g = (5, 7.7, _apply1, False)
  _apply2 = -1
  h = [5, 7.7, _apply2, False]
  i = {}
  j = {'a': '1', True: 2}
  k = dict()
  _apply3 = {'a': 1, 'b': 2}
  _apply4 = ()
  l = dict(*_apply4, **_apply3)
  m = operator.add
  n = math.sin
  return (3, b, c, True, 'a string', g, h, i, j, k, l, m, n)
"""
    assert fn()[4] == "a string"
