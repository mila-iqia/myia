import io
import math
import operator
import sys

from myia.compile.backends.python.python import compile_graph
from myia.parser import parse
from myia.utils.info import enable_debug


def parse_and_compile(function, debug=True, optimize=True):
    if debug:
        with enable_debug():
            graph = parse(function)
    else:
        graph = parse(function)
    output = io.StringIO()
    fn = compile_graph(graph, debug=output, optimize=optimize)
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

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(x, y, b):
  _1 = type(x)
  _2 = type(y)
  _x_2 = make_handle(_1)
  _y_2 = make_handle(_2)
  _3 = global_universe_setitem(_x_2, x)
  _4 = global_universe_setitem(_y_2, y)
  _5 = bool(b)

  def if_false_f():
    return global_universe_getitem(_y_2)

  def if_true_f():
    return global_universe_getitem(_x_2)

  _6 = if_true_f if _5 else if_false_f
  return _6()
"""
    )
    assert f(2, 3, 0) == fn(2, 3, 0) == 3
    assert f(2, 3, 1) == fn(2, 3, 1) == 2


def test_ifexp_optimized():
    def f(x, y, b):
        return x if b else y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x, y, b):
  _x_2 = x
  _y_2 = y
  _1 = bool(b)

  def if_false_f():
    return _y_2

  def if_true_f():
    return _x_2

  _2 = if_true_f if _1 else if_false_f
  return _2()
"""
    )
    assert f(2, 3, 0) == fn(2, 3, 0) == 3
    assert f(2, 3, 1) == fn(2, 3, 1) == 2


def test_boolop():
    def f(a, b, c):
        return a and b or c

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(a, b, c):
  _1 = type(b)
  _2 = type(c)
  _b_2 = make_handle(_1)
  _c_2 = make_handle(_2)
  _3 = global_universe_setitem(_b_2, b)
  _4 = global_universe_setitem(_c_2, c)
  _5 = bool(a)

  def if_false_f():
    return False

  def if_true_f():
    return global_universe_getitem(_b_2)

  _6 = if_true_f if _5 else if_false_f
  _7 = _6()
  _8 = bool(_7)

  def _if_false_f_2():
    return global_universe_getitem(_c_2)

  def _if_true_f_2():
    return True

  _9 = _if_true_f_2 if _8 else _if_false_f_2
  return _9()
"""
    )
    assert f(1, 2, 3) == 2 and fn(1, 2, 3) is True
    assert f(1, 0, 4) == fn(1, 0, 4) == 4


def test_boolop_optimized():
    def f(a, b, c):
        return a and b or c

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(a, b, c):
  _b_2 = b
  _c_2 = c
  _1 = bool(a)

  def if_false_f():
    return False

  def if_true_f():
    return _b_2

  _2 = if_true_f if _1 else if_false_f
  _3 = _2()
  _4 = bool(_3)

  def _if_false_f_2():
    return _c_2

  def _if_true_f_2():
    return True

  _5 = _if_true_f_2 if _4 else _if_false_f_2
  return _5()
"""
    )
    assert f(1, 2, 3) == 2 and fn(1, 2, 3) is True
    assert f(1, 0, 4) == fn(1, 0, 4) == 4


def test_compare2():
    def f(x):
        return 0 < x < 42

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(x):
  _1 = type(x)
  _x_2 = make_handle(_1)
  _2 = global_universe_setitem(_x_2, x)
  _3 = global_universe_getitem(_x_2)
  _4 = 0 < _3
  _5 = bool(_4)

  def if_false_f():
    return False

  def if_true_f():
    _6 = global_universe_getitem(_x_2)
    return _6 < 42

  _7 = if_true_f if _5 else if_false_f
  return _7()
"""
    )
    assert f(1) is fn(1) is True
    assert f(30) is fn(30) is True
    assert f(0) is fn(0) is False
    assert f(100) is fn(100) is False


def test_compare2_optimized():
    def f(x):
        return 0 < x < 42

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _x_2 = x
  _1 = _x_2
  _2 = 0 < _1
  _3 = bool(_2)

  def if_false_f():
    return False

  def if_true_f():
    _4 = _x_2
    return _4 < 42

  _5 = if_true_f if _3 else if_false_f
  return _5()
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

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(b, x, y):
  _1 = type(x)
  _2 = type(y)
  _x_2 = make_handle(_1)
  _y_2 = make_handle(_2)
  _3 = global_universe_setitem(_x_2, x)
  _4 = global_universe_setitem(_y_2, y)
  _5 = bool(b)

  def if_false_f():
    return global_universe_getitem(_y_2)

  def if_true_f():
    return global_universe_getitem(_x_2)

  _6 = if_true_f if _5 else if_false_f
  return _6()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(5, 1, 2) == fn(5, 1, 2) == 1


def test_if_optimized():
    def f(b, x, y):
        if b:
            return x
        else:
            return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  _x_2 = x
  _y_2 = y
  _1 = bool(b)

  def if_false_f():
    return _y_2

  def if_true_f():
    return _x_2

  _2 = if_true_f if _1 else if_false_f
  return _2()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(5, 1, 2) == fn(5, 1, 2) == 1


def test_if2():
    def f(b, x, y):
        if b:
            return x
        return y

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(b, x, y):
  _1 = type(x)
  _2 = type(y)
  _x_2 = make_handle(_1)
  _y_2 = make_handle(_2)
  _3 = global_universe_setitem(_x_2, x)
  _4 = global_universe_setitem(_y_2, y)

  def if_after_f():
    return global_universe_getitem(_y_2)

  _5 = bool(b)

  def if_false_f():
    return if_after_f()

  def if_true_f():
    return global_universe_getitem(_x_2)

  _6 = if_true_f if _5 else if_false_f
  return _6()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_if2_optimized():
    def f(b, x, y):
        if b:
            return x
        return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  _x_2 = x
  _y_2 = y

  def if_after_f():
    return _y_2

  _1 = bool(b)

  def if_false_f():
    return if_after_f()

  def if_true_f():
    return _x_2

  _2 = if_true_f if _1 else if_false_f
  return _2()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_while():
    def f(b, x, y):
        while b:
            return x
        return y

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(b, x, y):
  _1 = type(b)
  _2 = type(x)
  _b_2 = make_handle(_1)
  _3 = type(y)
  _x_2 = make_handle(_2)
  _4 = global_universe_setitem(_b_2, b)
  _y_2 = make_handle(_3)
  _5 = global_universe_setitem(_x_2, x)
  _6 = global_universe_setitem(_y_2, y)

  def while_f():
    def while_after_f():
      return global_universe_getitem(_y_2)

    _7 = global_universe_getitem(_b_2)

    def else_while_f():
      return while_after_f()

    def body_while_f():
      return global_universe_getitem(_x_2)

    _8 = body_while_f if _7 else else_while_f
    return _8()

  return while_f()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_while_optimized():
    def f(b, x, y):
        while b:
            return x
        return y

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(b, x, y):
  _b_2 = b
  _x_2 = x
  _y_2 = y

  def while_f():
    def while_after_f():
      return _y_2

    _1 = _b_2

    def else_while_f():
      return while_after_f()

    def body_while_f():
      return _x_2

    _2 = body_while_f if _1 else else_while_f
    return _2()

  return while_f()
"""
    )
    assert f(0, 1, 2) == fn(0, 1, 2) == 2
    assert f(3, 1, 2) == fn(3, 1, 2) == 1


def test_while2():
    def f(x):
        while x:
            x = x - 1
        return x

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f(x):
  _1 = type(x)
  _x_2 = make_handle(_1)
  _2 = global_universe_setitem(_x_2, x)

  def while_f():
    def while_after_f():
      return global_universe_getitem(_x_2)

    _3 = global_universe_getitem(_x_2)

    def else_while_f():
      return while_after_f()

    def body_while_f():
      _4 = global_universe_getitem(_x_2)
      _5 = _4 - 1
      _6 = global_universe_setitem(_x_2, _5)
      return while_f()

    _7 = body_while_f if _3 else else_while_f
    return _7()

  return while_f()
"""
    )
    assert f(0) == fn(0) == 0
    assert f(100) == fn(100) == 0


def test_while2_optimized():
    def f(x):
        while x:
            x = x - 1
        return x

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f(x):
  _x_2 = x

  def while_f():
    def while_after_f():
      return _x_2

    _1 = _x_2

    def else_while_f():
      return while_after_f()

    def body_while_f():
      nonlocal _x_2
      _2 = _x_2
      _3 = _2 - 1
      _x_2 = _3
      return while_f()

    _4 = body_while_f if _1 else else_while_f
    return _4()

  return while_f()
"""
    )
    assert f(0) == fn(0) == 0
    assert f(100) == fn(100) == 0


def test_recursion():
    fn, output = parse_and_compile(factorial, optimize=False)
    assert output == (
        """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem
from test_code_format import factorial

def factorial(n):
  _1 = type(n)
  _n_2 = make_handle(_1)
  _2 = global_universe_setitem(_n_2, n)
  _3 = global_universe_getitem(_n_2)
  _4 = _3 < 2
  _5 = bool(_4)

  def if_false_factorial():
    _6 = global_universe_getitem(_n_2)
    _7 = global_universe_getitem(_n_2)
    _8 = _7 - 1
    _9 = factorial(_8)
    return _6 * _9

  def if_true_factorial():
    return 1

  _10 = if_true_factorial if _5 else if_false_factorial
  return _10()
"""
    )
    assert fn(5) == 120


def test_recursion_optimized():
    fn, output = parse_and_compile(factorial)
    assert output == (
        """from test_code_format import factorial

def factorial(n):
  _n_2 = n
  _1 = _n_2
  _2 = _1 < 2
  _3 = bool(_2)

  def if_false_factorial():
    _4 = _n_2
    _5 = _n_2
    _6 = _5 - 1
    _7 = factorial(_6)
    return _4 * _7

  def if_true_factorial():
    return 1

  _8 = if_true_factorial if _3 else if_false_factorial
  return _8()
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

    fn, output = parse_and_compile(f, debug=False, optimize=False)

    assert output == (
        """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem
from test_code_format import factorial
from myia.basics import myia_iter
from myia.basics import myia_hasnext
from myia.basics import myia_next

def _1(_2):
  _3 = type(_2)
  _4 = type(10)
  _5 = make_handle(_3)
  _6 = make_handle(_4)
  _7 = global_universe_setitem(_5, _2)
  _8 = global_universe_setitem(_6, 10)
  _9 = global_universe_getitem(_5)
  _10 = _9 % 2
  _11 = _10 == 0

  def _12():
    _13 = type(0)
    _14 = global_universe_getitem(_5)
    _15 = make_handle(_13)
    _16 = factorial(_14)
    _17 = global_universe_getitem(_15)
    _18 = _16 - _17
    _19 = global_universe_getitem(_6)
    return _18 + _19

  _20 = bool(_11)

  def _21():
    _22 = global_universe_getitem(_5)
    _23 = _22 ** 2
    _24 = global_universe_getitem(_5)
    _25 = 2 * _24
    _13 = type(0)
    _26 = _23 + _25
    _15 = make_handle(_13)
    _27 = _26 + 1
    _28 = global_universe_setitem(_15, _27)
    return _12()

  def _29():
    _13 = type(0)
    _15 = make_handle(_13)
    _30 = global_universe_setitem(_15, 0)
    _31 = global_universe_getitem(_5)
    _32 = range(_31)

    def _33():
      return _12()

    _34 = myia_iter(_32)

    def _35(_36):
      _37 = myia_hasnext(_36)

      def _38():
        return _33()

      def _39():
        _40 = myia_next(_36)
        _41 = _40[0]
        _42 = _40[1]
        _43 = global_universe_getitem(_15)
        _44 = _43 + _41
        _45 = global_universe_setitem(_15, _44)
        return _35(_42)

      _46 = _39 if _37 else _38
      return _46()

    return _35(_34)

  _47 = _29 if _20 else _21
  return _47()
"""
    )

    # TODO There is currently a bug with universe_getitem called with an unreachable key.
    # Thus, compiled code can't currently run.
    # assert fn(1) == 7
    # assert fn(5) == 94
    # assert fn(10) == 3628765


def test_no_debug_optimized():
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
  _3 = _2
  _4 = 10
  _5 = _3
  _6 = _5 % 2
  _7 = _6 == 0

  def _8():
    _9 = _3
    _10 = factorial(_9)
    _12 = _11
    _13 = _10 - _12
    _14 = _4
    return _13 + _14

  _15 = bool(_7)

  def _16():
    _17 = _3
    _18 = _17 ** 2
    _19 = _3
    _20 = 2 * _19
    _21 = _18 + _20
    _22 = _21 + 1
    _11 = _22
    return _8()

  def _23():
    _11 = 0
    _24 = _3
    _25 = range(_24)

    def _26():
      return _8()

    _27 = myia_iter(_25)

    def _28(_29):
      _30 = myia_hasnext(_29)

      def _31():
        return _26()

      def _32():
        nonlocal _11
        _33 = myia_next(_29)
        _34 = _33[0]
        _35 = _33[1]
        _36 = _11
        _37 = _36 + _34
        _11 = _37
        return _28(_35)

      _38 = _32 if _30 else _31
      return _38()

    return _28(_27)

  _39 = _23 if _15 else _16
  return _39()
"""
    )

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
    def f():
        x = "hello"

        def g():
            return x

        return g()

    fn, output = parse_and_compile(f, optimize=False)
    assert (
        output
        == """from myia.basics import make_handle
from myia.basics import global_universe_setitem
from myia.basics import global_universe_getitem

def f():
  _1 = type('hello')
  x = make_handle(_1)
  _2 = global_universe_setitem(x, 'hello')

  def g():
    return global_universe_getitem(x)

  return g()
"""
    )
    assert fn() == "hello"


def test_universe_on_string_optimized():
    def f():
        x = "hello"

        def g():
            return x

        return g()

    fn, output = parse_and_compile(f)
    assert (
        output
        == """def f():
  x = 'hello'

  def g():
    return x

  return g()
"""
    )
    assert fn() == "hello"


def test_no_return():
    def f(x):
        y = 2 * x

        def g(i):
            j = i + x + y  # noqa: F841

        z = g(0)  # noqa: F841

    fn, output = parse_and_compile(f, optimize=True)
    assert (
        output
        == """def f(x):
  _x_2 = x
  _1 = _x_2
  _2 = 2 * _1
  y = _2

  def g(i):
    _3 = _x_2
    _4 = i + _3
    _5 = y
    j = _4 + _5
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

    fn, output = parse_and_compile(f, optimize=True)
    assert (
        output
        == """# Dynamic external import: thingy

def f(x):
  return x * thingy
"""
    )
    assert fn(2) == 20
