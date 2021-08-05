"""MUlti-testing utilities."""
from collections import Counter

import pytest
from ovld import ovld

from myia.abstract import data
from myia.infer.infnode import infer_graph
from myia.parser import parse
from myia.testing.common import A
from myia.utils.info import enable_debug


def mt(*testers):
    """Multitest.

    All testers in the list of tests will be run on the same function.

    Arguments:
        testers: list of callable objects.
            Each tester will be called with function to test (`tester(fn)`)
    """

    def deco(fn):
        def runtest(tester):
            tester(fn)()

        nb_tester_instances = Counter()
        pytests = []
        for tester in testers:
            tester_name = tester.__name__
            nb_tester_instances.update([tester_name])
            pytests.append(
                pytest.param(
                    tester,
                    marks=getattr(pytest.mark, tester_name),
                    id=f"{tester_name}{nb_tester_instances[tester_name]}",
                )
            )
        return pytest.mark.parametrize("tester", pytests)(runtest)

    return deco


@ovld
def _check_inference(self, x: object, y: object):  # noqa: F811
    return x is y


@ovld
def _check_inference(  # noqa: F811
    self, x: data.AbstractUnion, y: data.AbstractUnion
):
    return (
        type(x) is type(y)
        and len(x.options) == len(y.options)
        and all(vx is vy for vx, vy in zip(x.options, y.options))
    )


@ovld
def _check_inference(  # noqa: F811
    self, x: (data.AbstractAtom, data.AbstractStructure), y: data.AbstractUnion
):
    return x in y.options


@ovld
def _check_inference(  # noqa: F811
    self, x: data.AbstractAtom, y: data.AbstractAtom
):
    # compare value track manually.
    xval = x.tracks.value
    yval = y.tracks.value
    return (
        type(x) is type(y)
        and x.tracks.interface is y.tracks.interface
        and (xval == data.ANYTHING or yval == data.ANYTHING or xval == yval)
    )


@ovld
def _check_inference(  # noqa: F811
    self, x: data.AbstractDict, y: data.AbstractDict
):
    return (
        len(x.elements) == len(y.elements)
        and all(kx is ky for kx, ky in zip(x.keys, y.keys))
        and all(self(vx, vy) for vx, vy in zip(x.values, y.values))
    )


def infer(*args, result=None):
    """Inference tester.

    Arguments:
        args: The argspec for the function.
        result: The expected result, or an exception subclass or instance.
            If result is an exception subclass, test is expected to fail
            by raising given exception.
            If result is an exception instance, test is expected to fail
            by raising given exception type, and str(result) will be used
            as a regex expression to search in raised exception message.

    Returns:
        callable: a decorator that will receive a function to test
            and must returns the final wrapped function to run.
    """
    args = tuple(
        arg if isinstance(arg, data.AbstractValue) else A(arg) for arg in args
    )

    exc_type = None
    exc_match = None
    if not isinstance(result, data.AbstractValue):
        if isinstance(result, Exception):
            exc_type = type(result)
            exc_match = str(result)
        elif isinstance(result, type) and issubclass(result, Exception):
            exc_type = result
        else:
            result = A(result)

    def deco(fn):
        def wrapper(*a, **k):
            with enable_debug():
                graph = parse(fn)

                if isinstance(result, data.AbstractValue):
                    ret_graph = infer_graph(graph, args)
                    ret_type = ret_graph.return_.abstract
                    assert _check_inference(
                        ret_type, result
                    ), f"Expected {result}, got {ret_type}"
                else:
                    with pytest.raises(exc_type, match=exc_match):
                        infer_graph(graph, args)

        return wrapper

    deco.__name__ = infer.__name__

    return deco
