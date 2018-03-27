Myia
====

Myia is a new differentiable programming language. It aims to support large scale high performance computations (e.g. linear algebra) and their gradients. The main application Myia aims to support is research in artificial intelligence, in particular deep learning algorithms.

* Define a model using a subset of Python, which is compiled to Myia (interfaces in other languages than Python may follow). This subset is general purpose and includes looping constructs and recursion. It excludes side effects and inplace operations.

* Ask for the derivative of your model. Derivatives are fully supported for all control flow and all differentiable primitives.

* Compile to efficient CPU and GPU code that optimizes use of your resources.


Status
------

Myia is currently under development and is not yet ready for use. As of 2018/03/26 we anticipate we may be able to offer a beta for 2018/07/01. We will update that estimate regularly.

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

* **Parser**: Supports ``def``, ``if``, ``while``, operators, function calls.
* **Intermediate representation**: Implemented, with an array of utilities.
* **Debug VM**: Faithfully runs the IR.
* **Primitives**: Only scalar primitives currently work.
* **Automatic differentiation**: Works, tested up to second order, PR `#22`_ (awaiting review).

In development
~~~~~~~~~~~~~~

* **Type system**: Implementation in progress (`#27`_, `#41`_, `#50`_)
* **Value propagation**: Under development.
* **Inlining**: Requires graph cloning code awaiting review in `#46`_
* **Optimization**: in ``breuleux:opt``, requires cleanup, no PR yet.

  * **Pattern optimizer**: Works.
  * **Inlining**: Works, using the algorithm in `#46`_.
  * **Constant propagation**: Works.

Next steps
~~~~~~~~~~

* **Broadcasting**: We are trying to figure out the best primitives to support this feature. Discussion in `#43`_.
* **Array primitives**: Need to implement ``map`` and ``reduce`` as well as their backpropagators.
* **GPU support**: We currently plan to integrate NNVM as a backend to compile subgraphs of array primitives.

Near future
~~~~~~~~~~~

* **Low level VM**: We need a VM that is more efficient and portable than the debug VM.
* **Debugger**: Intent is to have a step debugger for Myia. There used to be a working one for a previous version of the IR, so this should not pose a problem.
* **More Python syntax**: ``for``, ``lambda``, ``and``/``or``, ``break/continue``.

After Beta
~~~~~~~~~~

* **Even more Python syntax**: Support for these features is not certain.

  * ``class`` (under restrictions)
  * Augmented assignment (under restrictions)
  * ``yield`` and ``await``

* **Support other languages**: Which ones depend on demand. A new language is also a possibility.

.. _#22: https://github.com/mila-udem/myia/pull/22
.. _#27: https://github.com/mila-udem/myia/pull/27
.. _#41: https://github.com/mila-udem/myia/pull/41
.. _#43: https://github.com/mila-udem/myia/pull/43
.. _#46: https://github.com/mila-udem/myia/pull/46
.. _#50: https://github.com/mila-udem/myia/pull/50
