from itertools import product
from types import FunctionType

import numpy as np
import pytest
from ovld import ovld

from myia.compile.backends import get_backend_names
from myia.lib import concretize_abstract, from_value
from myia.pipeline import standard_debug_pipeline, standard_pipeline
from myia.utils import keyword_decorator, merge

from .common import to_abstract_test

ParameterSet = type(pytest.param(1234))


@ovld
def eqtest(t1: tuple, t2, **kwargs):
    return isinstance(t2, tuple) and all(
        eqtest(x1, x2, **kwargs) for x1, x2 in zip(t1, t2)
    )


@ovld  # noqa: F811
def eqtest(a1: (np.ndarray, int, float), a2, rtol=1e-5, atol=1e-8, **kwargs):
    try:
        return np.allclose(a1, a2, rtol=rtol, atol=atol)
    except TypeError:
        return False


@ovld  # noqa: F811
def eqtest(x: type(None), y, **kwargs):
    if y is None:
        return True
    else:
        return (y == 0).all()


@ovld  # noqa: F811
def eqtest(x: object, y, **kwargs):
    return x == y


@overload  # noqa: F811
def to_numpy(value: object):
    """Convert a value to a numpy array. Used in _run below."""
    return value


infer_pipeline = standard_pipeline.select("resources", "parse", "infer")


class Multiple:
    def __init__(self, *params, **options):
        self.options = []
        for i, param in enumerate(params):
            assert isinstance(param, ParameterSet)
            self.options.append((self, param.id, param.marks, param.values[0]))
        self.options += [(self, id, [], v) for id, v in options.items()]


def mt(*tests, **kwargs):
    """Multitest.

    All MyiaFunctionTest instances in the list of tests will be run on the same
    function. If kwargs are provided, they will be given to all the tests.
    """

    def deco(fn):
        def runtest(test):
            test.run(fn)

        pytests = []
        for test in tests:
            test = test.configure(**kwargs)
            pytests += test.generate_params()
        runtest = pytest.mark.parametrize("test", pytests)(runtest)
        return runtest

    return deco


class MyiaFunctionTest:
    """Test a Myia function on a set of arguments.

    Arguments:
        runtest: The function to use to run the test.
        spec: The kwargs to give to the function.
    """

    def __init__(self, runtest, spec):
        """Initialize a MyiaFunctionTest."""
        self.spec = spec
        self.runtest = runtest

    def configure(self, **spec):
        """Configure this test with new kwargs."""
        return MyiaFunctionTest(self.runtest, spec={**self.spec, **spec})

    def check(self, run, args, expected, **kwargs):
        """Check the result of run() against expected.

        Expected can be either:

        * A value, which will be compared using eqtest.
        * A subclass of Exception, which run() is expected to raise.
        * A callable, which can run custom checks.
        """
        message = None
        if isinstance(expected, Exception):
            message = expected.args[0]
            expected = type(expected)
        if isinstance(expected, type) and issubclass(expected, Exception):
            try:
                res = run(args)
            except expected as err:
                if message is not None and message not in err.args[0]:
                    raise
            else:
                raise
        else:
            res = run(args)
            if isinstance(expected, FunctionType):
                if not expected(args, res):
                    raise Exception(f"Failed the result check function")

            elif not eqtest(res, expected, **kwargs):
                raise Exception(f"Mismatch: expected {expected}, got {res}")

    def generate_params(self):
        """Generate pytest parameters.

        If any of the kwargs is an instance of Multiple, we will generate tests
        for each possible value it can take. If there are multiple Multiples,
        we will test a cartesian product of them.
        """
        marks = self.spec.get("marks", [])
        id = self.spec.get("id", "test")
        spec = dict(self.spec)
        for key in ("marks", "id"):
            if key in spec:
                del spec[key]

        multis = [
            (k, v) for k, v in self.spec.items() if isinstance(v, Multiple)
        ]
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
            p = pytest.param(
                MyiaFunctionTest(self.runtest, curr_spec),
                marks=curr_marks,
                id="-".join(curr_ids),
            )
            params.append(p)

        return params

    def run(self, fn):
        """Run the test on the given function."""
        return self.runtest(self, fn, **self.spec)

    def __call__(self, fn):
        """Decorate a Myia function."""
        return mt(self)(fn)


class MyiaFunctionTestFactory:
    """Represents a decorator to perform a particular test on a function.

    For example, @infer, @run, etc. are instances of this. Calling a
    MyiaFunctionTestFactory produces a MyiaFunctionTest, e.g. infer(1, 2, 3)
    creates a MyiaFunctionTest.

    Arguments:
        runtest: The function to run the test.
        spec: Default kwargs for the function.
    """

    def __init__(self, runtest, spec={}):
        """Initialize a MyiaFunctionTestFactory."""
        self.runtest = runtest
        self.spec = spec

    def configure(self, **spec):
        """Configure the factory with more kwargs."""
        return MyiaFunctionTestFactory(self.runtest, merge(self.spec, spec))

    def xfail(self, *args, **kwargs):
        """Add the xfail mark when the test is created."""
        return self(*args, **kwargs, marks=[pytest.mark.xfail])

    def __call__(self, *args, **kwargs):
        """Create a MyiaFunctionTest."""
        kwargs = merge(self.spec, kwargs)
        kwargs["args"] = args
        return MyiaFunctionTest(self.runtest, kwargs)


@keyword_decorator
def myia_function_test(fn, **kwargs):
    """Create a MyiaFunctionTestFactory from a function."""
    return MyiaFunctionTestFactory(fn, kwargs)


@myia_function_test(marks=[pytest.mark.infer], id="infer")
def infer(self, fn, args, result=None, pipeline=infer_pipeline):
    """Inference test.

    Arguments:
        fn: The Myia function to test.
        args: The argspec for the function.
        result: The expected result, or an exception subclass.
    """
    args = [to_abstract_test(arg) for arg in args]

    def out(args):
        pip = pipeline.make()
        res = pip(input=fn, argspec=args)
        rval = res["outspec"]
        rval = concretize_abstract(rval)
        return rval

    self.check(out, args, to_abstract_test(result))


@myia_function_test(marks=[pytest.mark.run], id="run")
def _run(
    self,
    fn,
    args,
    result=None,
    abstract=None,
    broad_specs=None,
    validate=True,
    pipeline=standard_pipeline,
    backend=None,
    numpy_compat=False,
    **kwargs,
):
    """Test a Myia function.

    Arguments:
        fn: The Myia function to test.
        args: The args for the function.
        result: The expected result, or an exception subclass. If result is
            None, we will call the Python version of the function to compare
            with.
        abstract: The argspec. If None, it will be derived automatically from
            the args.
        broad_specs: For each argument, whether to broaden the type. By
            default, broaden all arguments.
        validate: Whether to run the validation step.
        pipeline: The pipeline to use.
    """

    if backend:
        backend_name = backend[0]
        backend_options = backend[1]

        pipeline = pipeline.configure(
            {
                "resources.backend.name": backend_name,
                "resources.backend.options": backend_options,
            }
        )

    if abstract is None:
        if broad_specs is None:
            broad_specs = (True,) * len(args)
        argspec = tuple(
            from_value(arg, broaden=bs) for bs, arg in zip(broad_specs, args)
        )
    else:
        argspec = tuple(to_abstract_test(a) for a in abstract)

    if not validate:
        pipeline = pipeline.configure(validate=False)

    def out(args):
        pip = pipeline.make()
        mfn = pip(input=fn, argspec=argspec)
        rval = mfn["output"](*args)
        return rval

    if result is None:
        result = fn(*args)

    self.check(out, args, result, **kwargs)

    if numpy_compat:
        args_torch = args
        args = ()
        for _ in args_torch:
            args += (to_numpy(_),)

        if abstract is None:
            if broad_specs is None:
                broad_specs = (True,) * len(args)
            argspec = tuple(
                from_value(arg, broaden=bs)
                for bs, arg in zip(broad_specs, args)
            )
        else:
            argspec = tuple(to_abstract_test(a) for a in abstract)

        out(args)


_loaded_backends = get_backend_names()
_pytest_parameters = {}


def _generate_pytest_parameter(backend, target, options, identifier=None):
    if identifier is None:
        identifier = "%s-%s" % (backend, target)
    marks = [getattr(pytest.mark, backend)]
    if target == "gpu":
        marks.append(getattr(pytest.mark, target))
    return pytest.param((backend, options), id=identifier, marks=marks)


def register_backend_testing(backend, target, options, identifier=None):
    """register some backend options to test a backend.

    :param backend: name of backend to register parameters
    :param target: device to register parameters.
        Currently either "cpu" or "gpu".
    :param options: dictionary of options to pass to backend loader
    :param identifier: a unique identifier for this testing.
        By default: "<backend>-<target>"
    :type backend: str
    :type target: str
    :type options: dict
    :type identifier: str
    """
    if backend not in _loaded_backends:
        raise RuntimeError("Unknown backend: %s" % backend)
    if target not in ("cpu", "gpu"):
        raise RuntimeError("Unsupported target: %s" % target)
    _pytest_parameters.setdefault(backend, {}).setdefault(target, []).append(
        _generate_pytest_parameter(backend, target, options, identifier)
    )


def _get_backend_testing_parameters():
    for backend, targets in _pytest_parameters.items():
        for target, params in targets.items():
            for param in params:
                yield backend, target, param


register_backend_testing("relay", "cpu", {"target": "cpu", "device_id": 0})
register_backend_testing(
    "relay", "gpu", {"target": "cuda", "device_id": 0}, "relay-cuda"
)
register_backend_testing("pytorch", "cpu", {"device": "cpu"})
register_backend_testing("pytorch", "gpu", {"device": "cuda"}, "pytorch-cuda")


class __Module:
    """Helper to provide some module attributes at runtime.

    Those attributes depend on backend testing options, which may
    not be all known when module is loaded (e.g. if another backend
    add some backend testings), so it should be better to generate
    these attributes at runtime instead of hardcoding them.
    """

    @property
    def backend_gpu(self):
        return Multiple(*list(_get_backend_testing_parameters()))

    @property
    def backend_all(self):
        return Multiple(
            *[
                param
                for backend, target, param in _get_backend_testing_parameters()
                if target == "cpu"
            ]
        )

    @property
    def run(self):
        return _run.configure(backend=self.backend_all)

    @property
    def run_gpu(self):
        return _run.configure(backend=self.backend_gpu)

    @property
    def run_debug(self):
        return self.run.configure(
            pipeline=standard_debug_pipeline, validate=False, backend=False
        )


__module = __Module()


def __getattr__(name):
    return getattr(__module, name)
