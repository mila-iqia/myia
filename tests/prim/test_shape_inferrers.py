from myia.prim.shape_inferrers import find_matching_shape, InferenceError

def test_find_matching_shape():
    # We test some cases that are hard to reproduce via from_value()
    try:
        find_matching_shape([(2, 3), 2])
        assert False
    except InferenceError:
        pass

    try:
        find_matching_shape([(2, 3), (1, 2, 3)])
        assert False
    except InferenceError:
        pass
