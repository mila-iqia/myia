
from types import FunctionType

import numpy as np
import pytest

from myia.lib import concretize_abstract, from_value
from myia.pipeline import standard_debug_pipeline, standard_pipeline
from myia.utils import keyword_decorator, merge, overload

from .common import to_abstract_test


@overload
def eqtest(t1: tuple, t2, **kwargs):
    return (isinstance(t2, tuple)
            and all(eqtest(x1, x2, **kwargs) for x1, x2 in zip(t1, t2)))


@overload  # noqa: F811
def eqtest(a1: (np.ndarray, int, float), a2,
           rtol=1e-5, atol=1e-8, **kwargs):
    return np.allclose(a1, a2, rtol=rtol, atol=atol)


@overload  # noqa: F811
def eqtest(x: object, y, **kwargs):
    return x == y


infer_pipeline = standard_pipeline.select(
    'resources', 'parse', 'infer'
)

run_pipeline = standard_pipeline

run_debug_pipeline = standard_debug_pipeline


def mt(*tests, **kwargs):
    def deco(fn):
        def runtest(test):
            test.run(fn)
        pytests = []
        for test in tests:
            test = test.configure(**kwargs)
            pytests.append(test.make_param())
        runtest = pytest.mark.parametrize('test', pytests)(runtest)
        return runtest
    return deco


class MyiaFunctionTest:

    def __init__(self, runtest, spec):
        self.spec = spec
        self.runtest = runtest

    def configure(self, **spec):
        return MyiaFunctionTest(
            self.runtest,
            spec={**self.spec, **spec},
        )

    def check(self, run, expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            try:
                res = run()
            except expected as err:
                pass
            else:
                raise Exception(
                    f'Expected an error of type {expected}, '
                    f'but got result {res}'
                )
        else:
            res = run()
            if isinstance(expected, FunctionType):
                if not expected(res):
                    raise Exception(
                        f'Failed the result check function'
                    )

            elif not eqtest(res, expected):
                raise Exception(
                    f'Mismatch: expected {expected}, got {res}'
                )

    def make_param(self):
        p = pytest.param(self,
                         marks=self.spec.get('marks', []),
                         id=self.spec.get('id', None))
        return p

    def run(self, fn):
        spec = dict(self.spec)
        for key in ('marks', 'id'):
            if key in spec:
                del spec[key]
        return self.runtest(self, fn, **spec)

    def __call__(self, fn):
        return mt(self)(fn)


class MyiaFunctionTestFactory:
    def __init__(self, runtest, spec={}):
        self.runtest = runtest
        self.spec = spec

    def configure(self, **spec):
        return MyiaFunctionTestFactory(self.runtest, merge(self.spec, spec))

    def xfail(self, *args, **kwargs):
        return self(*args, **kwargs, marks=[pytest.mark.xfail])

    def __call__(self, *args, **kwargs):
        kwargs = merge(self.spec, kwargs)
        kwargs['args'] = args
        return MyiaFunctionTest(self.runtest, kwargs)


@keyword_decorator
def myia_function_test(fn, **kwargs):
    return MyiaFunctionTestFactory(fn, kwargs)


@myia_function_test(marks=[pytest.mark.infer], id='infer')
def infer(self, fn, args, result=None, pipeline=infer_pipeline):
    args = [to_abstract_test(arg) for arg in args]

    def out():
        pip = pipeline.make()
        res = pip(input=fn, argspec=args)
        rval = res['outspec']
        rval = concretize_abstract(rval)
        return rval

    self.check(out, to_abstract_test(result))


@myia_function_test(marks=[pytest.mark.run], id='run')
def run(self, fn, args, result=None, abstract=None,
        validate=True, pipeline=run_pipeline):

    if abstract is None:
        argspec = tuple(from_value(arg, broaden=True)
                        for arg in args)
    else:
        argspec = tuple(to_abstract_test(a) for a in abstract)

    if not validate:
        pipeline = pipeline.configure(validate=False)

    def out():
        pip = pipeline.make()
        mfn = pip(input=fn, argspec=argspec)
        rval = mfn['output'](*args)
        return rval

    if result is None:
        result = fn(*args)

    self.check(out, result)


run_debug = run.configure(pipeline=run_debug_pipeline, validate=False)
