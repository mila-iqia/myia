
from myia.front import parse_function, MyiaSyntaxError
from myia.interpret import evaluate
import pytest
import inspect
import tests.functions as fns

_functions = {}

class TestParse:
    # Populated with tests for the functions in the functions module
    # by _create_tests_for
    pass


def _create_tests_for(fn):
    ann = getattr(fn, 'annotations', {})
    g = globals()

    def test(self):
        if 'stxerr' in ann:
            try:
                parse_function(fn)
            except MyiaSyntaxError as e:
                assert ann['stxerr'] in e.message
            else:
                raise Exception('{} should raise syntax error with message containing "{}"'.format(fn.__name__, ann['stxerr']))
        else:
            data = parse_function(fn)
            _functions.update(data)

            tests = list(ann.get('tests', []))
            test = ann.get('test', None)
            if test is not None:
                tests.append(test)
            assert len(tests) > 0, "At least one inputs/output pair should be provided to test this function."
            for ins, out in tests:
                if not isinstance(ins, tuple):
                    ins = ins,
                python_result = fn(*ins)
                myia_result = evaluate(data[fn.__name__], _functions)(*ins)
                assert python_result == out
                assert myia_result == out
            return fn

    if ann.get('xfail', False):
        test = pytest.mark.xfail(test)
    setattr(TestParse, 'test({})'.format(fn.__name__), test)


def _create_tests():
    fnames = [fn for fn in dir(fns) if fn.startswith("fn_")]
    for fname in fnames:
        fn = getattr(fns, fname)
        _create_tests_for(fn)


_create_tests()
