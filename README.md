# Myia

[![Build Status](https://travis-ci.com/mila-udem/myia.svg?token=p8b613NdVqVa9KeL48d5&branch=master)](https://travis-ci.com/mila-udem/myia)

Myia is a research/prototyping project to develop a new toolchain for array programming with GPU support and automatic differentiation.

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

## Command line

At this stage of development, the following commands are available:

```bash
$ python -m myia -h
Myia

Usage:
  myia parse [FILE] [-e <expr>]
  myia (-h | --help)
  myia --version

Options:
  -h --help     Show this screen.
  -e <expr>     Provide an expression to evaluate.
  --version     Show version.
```

