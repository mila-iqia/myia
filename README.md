# Myia

[![Build Status](https://travis-ci.com/mila-udem/myia.svg?token=p8b613NdVqVa9KeL48d5&branch=master)](https://travis-ci.com/mila-udem/myia)

## What is it?

Myia is a new deep learning framework that aims to combine:

* Flexibility and ease of use.
* Automatic differentiation.
* Advanced optimization and high performance.

It works by parsing a subset of the Python language, running type and shape inference as well as static analysis, and plugging into various backends in order to produce performant code.

**Current state of the project (2017/12/07):** not in a usable state. The internal representation is being redesigned, the optimization engine and backend are very rough. Most of the pieces are there in some form, however.


## Usage

**This is pre-alpha software**, so don't rely on it for anything important, and don't expect it to be fast.

Myia works by decorating a function with `@myia`:

```python
from myia.front import myia

@myia
def square(x):
    return x * x

print(square(1234))
```

Myia will compile that function for you, as well as any functions called by that function (so you only need to decorate the top-level function(s) you are going to call directly).

## Tests

Run tests with:

```bash
$ ./run_tests.sh
```

Make sure that pytest is installed (`pip install pytest`).

## Planned features and comparison

Feature                        | Myia | Theano | TensorFlow | PyTorch | Autograd | CUDA | Torch
------------------------------ | ---- | ------ | ---------- | ------- | -------- | ---- | -----
Array type                     | ✓    | ✓      | ✓          | ✓       | ✓        | ✗    | ✓
GPU support                    | ✓    | ✓      | ✓          | ✓       | ✗        | ✓    | ✓
Reverse-mode AD                | ✓    | ✓      | ✓          | ✓       | ✓        | ✗    | ✓
Python integration             | ✓    | ✓      | ✓          | ✓       | ✓        | ✗    | ✗
Control flow statements        | ✓    | ✗      | ✗          | ✓       | ✓        | ✓    | ✗
Optimizing compiler            | ✓    | ✓      | ✓          | ✗       | ✗        | ✓    | ✗
Optimizing numerical stability | ✓    | ✓      | ✗          | ✗       | ✗        | ✗    | ✗
Runtime debugging              | ✓    | ✗      | ✗          | ✓       | ✓        | ✗    | ✗

* **Array type**: For user-friendliness and type safety we want built-in array types. Theano and TensorFlow both support gradual typing (shapes can be optionally given) with type inference, although more formal and extensive type systems which include shape types, symmetric matrices, etc. would be better.
* **GPU support**: For performance reasons we want to be able to use parallel architectures e.g. GPUs through CUDA
* **Reverse-mode AD**: Deep learning research relies heavily on calculating gradients for functions with large input spaces
* **Python integration**: Python is the go-to programming language for scientists, with a huge number of libraries that can be used for visualization, data preprocessing, etc.
* **Control flow statements**: Although libraries like Theano and TensorFlow have support for control flow, it is in the form of function calls instead of statements. Moreover, some support (e.g. `continue` and `break` statements) is missing, and the internal representation of control flow (such as loops) limits and complicates the optimizing compiler.
* **Optimizing compiler**: For high-performance and portability a pipeline involving code generation for potentially multiple backends is preferred over an interpreted language
* **Optimizing numerical stability**: Since the program formulation is intended to be more mathematical than numerical, we want our compiler to optimize for numerical stability
* **Runtime debugging**: Certain numerical errors are easiest to debug when the user can easily print or log intermediate values, or drop into a debugger when certain conditions arise. Theano and TensorFlow provide custom debugging tools for this, but they are generally not as convenient as dropping into `pdb` in PyTorch or Autograd. Since Myia is envisioned as a subset of Python, it could support a Python-based mode similar to these frameworks.
