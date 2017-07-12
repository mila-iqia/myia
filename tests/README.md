
# Tests

## Running

Run tests with:

```bash
$ pytest
```

Make sure that pytest is installed (`pip install pytest`) and myia is in your PYTHONPATH.

## Adding tests

`pytest` automatically collects all functions called `test_*` and all `test_*` methods of `Test*` classes in all `test_*.py` files.

`test_front.py` tests the parsing and evaluating of various functions by myia's frontend. You can use the `myia_test` decorator to add more tests.
