Roadmap
=======

A short outline of what I belive are some fundamental design choices for
the next Theano. These are based on discussions I had at
FAIR (Jeff Johnson, Soumith Chintala) when they were
working on a TensorFlow equivalent and PyTorch, with Alex Wiltschko at Twitter
(including some discussions with Zachary DeVito from Stanford), with the
Theano developers, Olexa Bilaniuk, James Bergstra, and several others.

A view important guiding principles and features:

-  **Modularity**; Alex Wiltschko originally championed this, and I
   think it is extremely important. Instead of the monolithic systems we
   have seen so far we need a series of separate projects with clearly
   defined interfaces. This involves developing a language-independent
   graph representation.
-  **Runtime *and* compile time**; so far systems can be defined as
   either compile-time (Theano, TensorFlow, MXNet) or runtime (Chainer,
   Torch, torch-autograd). The former can more easily apply
   optimizations, the latter are easier to debug and more easily allows
   for complicated control flow. Why not do both (like Lisp)?
-  **Checkpointing**; a well-known technique that has never made it into
   machine learning (except for some non-general bits and pieces). But
   it could be extremely powerful. Imagine never running out of memory
   again! This is a feature more than a design principle, but it could
   be a major selling point.

The different components:

1. API
------

The API is language-specific. It transforms the users input into some
intermediate representation.

.. code:: python

    from numpy.random import rand
    from myia import grad, sum, tanh
    
    def f(x):
        return sum(tanh(x))
    
    df = grad(f)
    grad, loss = df(rand(3, 3))

2. Intermediate representation
------------------------------

A language-agnostic representation of computation graphs.

::

    1 new x1(ndim=2,dtype=float64)  # Define a new tensor variable with 2 dimensions and a data type
    2 tanh x2 x1  # Apply the hyperbolic tangent operator to x1 and store the result in x2
    3 sum x3 x2  # Sum the elements of x2 and store them in x3
    4 return x3  # Return the value of x3

A fair amount of thought and consideration should be put into which
information this graph should contain, and what it should not contain.
This decides what the API, IR, and compiler/interpreter are responsible
for. For example:

1. Should the IR be in `static single assignment
   form <https://en.wikipedia.org/wiki/Static_single_assignment_form>`__?
2. What kind of type system should this IR have? I believe `intersection
   typing <https://en.wikipedia.org/wiki/Type_system#Intersection_types>`__
   is a natural choice e.g. ``ndim=2`` for a general matrix, but
   ``ndim=2,shape=(3,3),symmetric=true`` for a symmetric 3x3 matrix.
   Note that we would be using
   `*generic* <https://en.wikipedia.org/wiki/Generic_programming>`__
   intersection types.
3. Should we introduce "hints" e.g. unstructured annotations that can
   help the interpreter/compiler with knowing how to execute certain
   commands? Useful hints could be "do this operation in-place" or
   "choose this particular convolution implementation". These hints can
   be freely ignored.
4. What operators should we support? For example, should ``lstm`` be an
   operator (since an LSTM kernel exists in cuDNN)?
5. Should the IR define the data type of each variable (below), or just
   input variables (above), or should we do type inference entirely at
   runtime (i.e. data types should be hints)?

   ::

       1 new x1(ndim=2,dtype=float64)
       2 tanh x2(ndim=2,dtype=float64) x1
       3 sum x3(ndim=0,dtype=float64) x2
       4 return x3

6. Should the return types of operators be part of the IR? For example,
   is the fact that ``type(tanh(x)) == type(x)`` (assuming ``x`` is
   floating point) part of the IR specification or a property of the
   operator itself that the compiler/interpreter applies.
7. What control flow statements should be part of the IR and how should
   they be represented? We can look at TensorFlow here, and `Click's
   'sea of nodes'
   representation <http://grothoff.org/christian/teaching/2007/3353/papers/click95simple.pdf>`__.
8. How do we deal with platform specific operations? Should the IR be
   entirely platform agnostic? This  forces the interpreter and compiler
   to decide when to transfer to the GPU or not, which is a bad idea
   (and the cause of a lot of slowdowns with Theano). Instead, the user
   should be able to control this (as in Torch). Should they then take
   the form of operators, or types, or both?

   ::

       # Using operators
       1 new x1(ndim=2,dtype=float64)
       2 togpu x2(ndim=2,dtype=float64) x1
       3 sum x3(ndim=0,dtype=float64) x2  # Compiler needs to figure out what's on the GPU and not
       # Using types
       1 new x1(ndim=2,dtype=float64)
       2 cast x2(ndim=2,dtype=float64,device=gpu1)
       3 sum x3(ndim=2,dtype=float64,device=gpu1) x2  # Compiler just looks at the type

Important guidelines when making these decisions should be: The
computation graph needs to contain all the information required to
perform the correct computation, but it should make minimal assumptions
otherwise to allow the compiler to optimize. I'm inclined to say that
the IR should be mathematical more than programmatical (except for the
hints); it shouldn't care about *how* the computation is executed or how
memory is managed, only about describing the execution in mathematical
terms. It shouldn't make any assumptions about the platform it is
running on (could be FPGA, CPU, GPU, a cluster).

3a. Runtime engine (automatic differentation)
---------------------------------------------

I would argue the system should have a runtime engine based on automatic
differentiation implemented with operator overloading (cf.
torch-autograd, PyTorch, Chainer). This allows for control flows within
the host language and easy debugging within the host language as well.

.. code:: python

    from myia import run
    from numpy.random import rand
    
    def f(x):
        if sum(x) > 0.1:
            return x
        else:
            return x + 1
        
    run(f)(rand(3, 3))

The ``run`` API would work by producing a single line of the IR (section
2) at a time and sending it to the AD/runtime execution engine. So when
``run(f)`` gets called, the line
``1 new x1(ndim=2,dtype=float64,shape=(3,3)`` is sent to the engine and
executed1. The ``sum(x)`` results in the instruction ``2 sum x2 x1``
which returns a 0-dimensional scalar with a value. The comparison
operator of this object is overloaded of course so that ``sum(x) > 0.1``
can evaluate normally.

In short, the runtime/AD engine simply receives one instruction at a
time and executes them immediately (possibly asynchronosly). Most
importantly, if the engine is informed that a variable is
differentiable, the engine is also in charge of keeping track of the
operators applied and their inputs, so that it can perform reverse
gradient computation.

1. Perhaps returning a pointer to which the actual NumPy data can be
   copied, or perhaps there should be a ``new from`` operator that
   accepts a memory pointer.

3b. Compiler
------------

3b.1 Gradient
~~~~~~~~~~~~~

Given an IR and a series of variables to differentiate, this module
spits out a new IR that includes the gradient computation of the
differentiable variables. Note that this is closely related to the IR
specification itself, since the gradient of each operator and its
properties should probably be part of the IR specification.

3b.2 Optimizers
~~~~~~~~~~~~~~~

An optimizer simply takes an IR and spits out a new, optimized IR that
performs the same computation. It can perform all the typical
optimizations. It can optionally consider the hints as well. Since the
IR is a sea-of-nodes style graph representation (with control flow being
represented with nodes), optimizations such as dead-code elimination are
trivial.

3b.3 Compiler
~~~~~~~~~~~~~

Given an entire optimized computation graph, this module is expected to
somehow execute it. There can be many different compilers: MPI-enabled
multi-node schedulers, FPGA-enabled compilers, etc. The only interface
restriction is that they must accept the IR and a set of inputs, and are
expected to produce the correct outputs.

The compiler's prerogative is to match the operations described in the
IR to the correct kernels at its disposal. It relies on the type system
here (using kernels for the correct data type, but also the correct
shape e.g. to select convolution kernels, to use the fact that matrices
are symmetric, or even to choose kernels based on auto-tuning, etc.).

The compiler is in charge of memory management as well (potentially
using e.g. CNMeM, CUB, or Torch's memory allocator).

4. Kernels
----------

The actual implementation of the operators should be a set of kernels
that can easily be re-used by the different compilers and interpreters.
For a large part, these kernels are already present in cuBLAS and cuDNN,
or they can be re-used from e.g. Torch (TH, THNN, THC, THCUNN). These
kernels should be BLAS-style, making zero assumptions about memory
management, execution environment, etc. They should simply take pointers
to the input and output, possibly some flags, and perform the required
computation.
