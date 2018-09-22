import pytest
from myia.dshape import find_matching_shape, InferenceError


def test_find_matching_shape():
    # We test some cases that are hard to reproduce via from_value()
    with pytest.raises(InferenceError):
        find_matching_shape([(2, 3), 2])

    with pytest.raises(InferenceError):
        find_matching_shape([(2, 3), (1, 2, 3)])
