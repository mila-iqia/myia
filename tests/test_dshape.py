import pytest
from myia.infer import ANYTHING
from myia.dshape import NOSHAPE, TupleShape, ListShape, ClassShape, \
    find_matching_shape, shape_cloner, InferenceError


def test_shape_cloner():
    s1 = ClassShape({'x': NOSHAPE,
                     'y': TupleShape([(1, 2, 3), ListShape((4, 5))])})

    s2 = ClassShape({'x': NOSHAPE,
                     'y': TupleShape([(2, 3, 4), ListShape((5, 6))])})

    assert shape_cloner(s1) == s1

    @shape_cloner.variant
    def adder(self, t: int):
        return t + 1

    assert adder(s1) == s2


def test_find_matching_shape():
    assert find_matching_shape(
        [TupleShape([(1, 2), NOSHAPE]), TupleShape([(1, 7), NOSHAPE])]
    ) == TupleShape([(1, ANYTHING), NOSHAPE])

    assert find_matching_shape(
        [ClassShape({'x': (1, 2)}), ClassShape({'x': (1, 7)})]
    ) == ClassShape({'x': (1, ANYTHING)})


def test_find_matching_shape_errors():
    # We test some cases that are hard to reproduce via from_value()
    with pytest.raises(InferenceError):
        find_matching_shape([(2, 3), 2])

    with pytest.raises(InferenceError):
        find_matching_shape([(2, 3), (1, 2, 3)])

    with pytest.raises(InferenceError):
        find_matching_shape([NOSHAPE, (1, 2, 3)])

    with pytest.raises(InferenceError):
        find_matching_shape([ListShape([NOSHAPE]), (1, 2, 3)])

    with pytest.raises(InferenceError):
        find_matching_shape([TupleShape([NOSHAPE]), (1, 2, 3)])

    with pytest.raises(InferenceError):
        find_matching_shape([TupleShape([NOSHAPE]),
                             TupleShape([NOSHAPE, NOSHAPE])])

    with pytest.raises(InferenceError):
        find_matching_shape(
            [ClassShape({'x': (1, 2)}), NOSHAPE]
        )

    with pytest.raises(InferenceError):
        find_matching_shape(
            [ClassShape({'x': (1, 2)}), ClassShape({'x': (1, 2), 'y': (3, 4)})]
        )
