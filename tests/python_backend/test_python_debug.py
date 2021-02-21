import io

from myia import myia


def test_debug():
    output = io.StringIO()

    @myia(backend="python", backend_options={"debug": output})
    def f(a, b):
        c = 2 * a
        d = a + b
        return 2 * a * b + (c - d) * (a - b)

    f(1, 2)
    code = output.getvalue()
    assert "def main(" in code
    print(code)
