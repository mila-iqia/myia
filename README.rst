Myia
====

Myia is a new differentiable programming language. It aims to support large scale high performance computations (e.g. linear algebra) and their gradients. The main application Myia aims to support is research in artificial intelligence, in particular deep learning algorithms.

* Define a model using a subset of Python, which is compiled to Myia (interfaces in other languages than Python may follow). This subset is general purpose and includes looping constructs and recursion. It excludes side effects and inplace operations.

* Ask for the derivative of your model. Derivatives are fully supported for all control flow and all differentiable primitives.

* Compile to efficient CPU and GPU code that optimizes use of your resources.

If you want to play with the current implementation, you can check out `ALPHA.md <https://github.com/mila-iqia/myia/blob/master/ALPHA.md>`_

A short document explaining some of Myia's inner workings is available `here <https://github.com/mila-iqia/myia/blob/master/devdoc.md>`_


Status
------

Myia is currently under development and is not yet ready for use. We are optimistic about having an alpha version to play with around the start of 2020.

See `Roadmap`_.


Motivation
----------

Development in artificial intelligence has been undergoing a boom in the past decade, chiefly due to the success of deep neural networks. The training of a neural network is a sort of *differentiable program*: one writes a program to compute the output and a cost, and then one computes the derivative of that cost with respect to the model's parameters to determine how they should be updated. 

Differentiation can be automated, but mainstream programming languages offer no support for this, hence the need for libraries or programming languages that can reliably support these applications.

The current leading solutions for deep learning fall in two camps:

**Computation graph-based solutions** such as TensorFlow, Theano and MXNet support automatic differentiation and are very well optimized, but they are not fully general, with only limited support for loops and none for general recursion. Thus models like recursive neural networks are tricky and awkward to write.

**Operator overloading solutions** such as PyTorch or Autograd use a dynamic approach to automatic differentiation which makes them much more general, but they are tightly coupled to the Python language and cannot reap the benefits of an optimizing compiler. They also involve a certain quantity of overhead per operation which discourages composing small cheap operations.

Myia's solution is to define a **strongly-typed, general-purpose intermediate representation** with an IR-level automatic differentiation transformation, which can then be compiled and optimized for various targets, thereby getting the best of both leading approaches.


Roadmap
-------

Current
~~~~~~~

* **Parser**: Supports ``def``, ``if``, ``for``, ``while``, operators, function calls, ``class`` and methods (limited support).
* **Intermediate representation**: Implemented, with an array of utilities.
* **Debug VM**: Faithfully runs the IR.
* **VM**: Works on the simplified/optimized IR.
* **Primitives**: Scalar primitives work, as well as map, reduce, broadcasting, 2D convolutions, concat/split, and many other operations.
* **Type system**: Types are inferred without the need for annotations. Shapes can also be inferred. Myia supports recursive ADTs (e.g. tree data structures).
* **Optimization**: Pattern-based optimizations, inlining, constant propagation, common subexpression elimination, closure conversion.
* **Automatic differentiation**: Second order differentiation is not yet in working order.
* **GPU support**: Using Relay or PyTorch.

In development
~~~~~~~~~~~~~~

* **Compiler optimization**: The compiler currently needs to be optimized to reduce compile times.
* **Auto-monadization**: We are working to support print statements and random number generation through an auto-monadization system that can automatically keep track of the IO or RNG state.

Next steps
~~~~~~~~~~

* **Error messages**: We need to make sure that every likely mistake leads to an understandable and traceable error diagnosis.

Near future
~~~~~~~~~~~

* **Serialization**: Serializing optimized graphs will allow for greater performance across runs and greater portability across systems.
* **Debugger**: Intent is to have a step debugger for Myia. There used to be a working one for a previous version of the IR, so this should not pose a problem.
* **More Python syntax**: ``break/continue``.

After Beta
~~~~~~~~~~

* **Even more Python syntax**: Support for these features is not certain.

  * Augmented assignment (under restrictions)
  * ``yield`` and ``await``

* **Support other languages**: Which ones depend on demand. A new language is also a possibility.

Publications
~~~~~~~~~~~~

* `Automatic differentiation in ML: Where we are and where we should be going (NeurIPS 2018) <https://papers.nips.cc/paper/8092-automatic-differentiation-in-ml-where-we-are-and-where-we-should-be-going>`_
