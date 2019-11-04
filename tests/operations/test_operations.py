from myia import myia


@myia
def run_lsfhit(a, b):
    return a << b


@myia
def run_rshift(a, b):
    return a >> b


@myia
def run_bit_and(a, b):
    return a & b


@myia
def run_bit_or(a, b):
    return a | b


@myia
def run_bit_xor(a, b):
    return a ^ b


def test_bitwise_operations():
    assert run_lsfhit(3, 2) == 12
    assert run_rshift(12, 2) == 3
    assert run_bit_and(5, 7) == 5
    assert run_bit_or(5, 2) == 7
    assert run_bit_xor(10, 8) == 2
