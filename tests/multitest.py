
from itertools import product
from types import FunctionType

import numpy as np
import pytest

from myia.lib import concretize_abstract, from_value
from myia.pipeline import standard_debug_pipeline, standard_pipeline
from myia.utils import keyword_decorator, merge, overload

from .common import to_abstract_test

ParameterSet = type(pytest.param(1234))


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


class Multiple:
    def __init__(self, *params, **options):
        self.options = []
        for i, param in enumerate(params):
            assert isinstance(param, ParameterSet)
            self.options.append(
                (self, param.id, param.marks, param.values[0])
            )
        self.options += [(self, id, [], v) for id, v in options.items()]


def mt(*tests, **kwargs):
    def deco(fn):
        def runtest(test):
            test.run(fn)
        pytests = []
        for test in tests:
            test = test.configure(**kwargs)
            pytests += test.generate_params()
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

    def generate_params(self):
        marks = self.spec.get('marks', [])
        id = self.spec.get('id', 'test')
        spec = dict(self.spec)
        for key in ('marks', 'id'):
            if key in spec:
                del spec[key]

        multis = [(k, v) for k, v in self.spec.items()
                  if isinstance(v, Multiple)]
        options = list(product(*[v.options for _, v in multis]))
        params = []
        for option in options:
            curr_spec = dict(spec)
            curr_ids = []
            curr_marks = list(marks)
            for (spec_k, _), opt_info in zip(multis, option):
                mul, opt_id, opt_marks, value = opt_info
                curr_spec[spec_k] = value
                curr_ids.append(opt_id)
                curr_marks += opt_marks
            curr_ids.append(id)
            p = pytest.param(self.configure(**curr_spec),
                             marks=curr_marks, id="-".join(curr_ids))
            params.append(p)

        return params

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
