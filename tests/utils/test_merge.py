
from myia.utils import DELETE, Merge, Override, Reset, TypeMap, cleanup, merge


def test_merge():

    assert merge(1, 2) == 2
    assert merge([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert merge((1, 2), (3, 4)) == (1, 2, 3, 4)
    assert merge({1, 2}, {3, 4}) == {1, 2, 3, 4}

    a = dict(a=1, b=2, c=dict(d=3, e=[4, 5]))
    b = dict(a=1, b=2, c=dict(e=[6, 7], f=8))
    c = dict(a=1, b=2, c=dict(d=3, e=[4, 5, 6, 7], f=8))

    assert merge(a, b) == c

    dlt = dict(c=DELETE, d=3)
    assert merge(a, dlt) == dict(a=1, b=2, d=3)


def test_merge_subclass():

    tm = TypeMap({int: "int"})
    mtm = merge(tm, {str: "str"})
    assert isinstance(mtm, TypeMap)
    assert mtm == TypeMap({int: "int", str: "str"})


def test_merge_modes():

    for x, y in [({1, 2}, {3, 4}),
                 ([1, 2], [3, 4]),
                 ((1, 2), (3, 4))]:

        assert merge(x, y, mode='reset') == y
        assert merge(x, Reset(y)) == y
        assert merge(x, Reset(y), mode='merge') == y

        assert merge(x, y, mode='override') == y
        assert merge(x, Override(y)) == y
        assert merge(x, Override(y), mode='merge') == y

    a = {'a': 1}
    b = {'b': 2}
    c = {'a': 1, 'b': 2}

    assert merge(a, b, mode='reset') == b
    assert merge(a, b, mode='override') == c

    a = {'a': [1, 2], 'b': [3, 4]}
    b = {'a': [5, 6], 'b': Override([7, 8])}
    c = {'a': [1, 2, 5, 6], 'b': [7, 8]}
    d = {'a': [5, 6], 'b': [7, 8]}

    assert merge(a, b) == c
    assert merge(a, b, mode='override') == d


def test_cleanup():
    a = dict(a=1, b=[2, Merge(3)], c=Override(4), d=DELETE)
    assert cleanup(a) == dict(a=1, b=[2, 3], c=4)


def test_cleanup_subclass():
    a = TypeMap({int: Merge("int")})
    ca = cleanup(a)
    assert isinstance(ca, TypeMap)
    assert ca == TypeMap({int: "int"})
