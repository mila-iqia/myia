
# Myia Alpha documentation


## Install

You will need to clone the repository and install the dependencies using `conda` and `pip`:

```bash
git clone https://github.com/mila-iqia/myia.git
cd myia
conda install -c abergeron -c pytorch --file=requirements.conda
pip install -r requirements.txt
pip install -e . --no-deps
```


## Run the tests

Run the test using `pytest`. This should also test the examples.


## Current API

[The MLP example](https://github.com/mila-iqia/myia/blob/master/examples/mlp.py) examplifies most of the current API.

* Decorate the top level function you wish to run with `@myia`. Any other functions called by that function will be found and  processed automatically.
* Use `grad(func)(arg1, arg2, ...)` to compute `dfunc/darg1`. You can also provide an argument name as the second argument to `grad`.
* Use `value_and_grad` with the same signature to get the value and the gradient in a tuple.


## Limitations

* Myia is a functional language. Destructive operations such as `a += b` or `a[i] = b` are not allowed.
* A lot of operations are still missing, notably array indexing and convolutions.
* `print` statements won't work. In fact, the parser will ignore any statements that are not connected to the output and there will be no warnings about it, so be aware of that.

