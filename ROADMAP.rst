====
Myia
====

Myia (pronouned *my-ah*) is the daughter of Theano, and the project name for a set of ideas that we are exploring for a potential successor of Theano.

Deep learning frameworks
========================

Deep learning frameworks encompass a wide variety of features and objectives.

* Multi-dimensional array primitives
* High-level mathematical functions
* Efficient GPU and CPU kernels
* Symbolic and/or automatic differentiation
* Computation graph optimization for performance and numerical stability
* Device (backend) agnostic interface
* Neural-network training tools
* Multi-GPU and multi-node computation

We have witnessed an explosion in the number of deep learning frameworks: Caffe, Chainer, CNTK, Theano, Torch, MXNet, and TensorFlow are perhaps the most common ones in the deep learning community, but there are many more out there: Deeplearning4j, Leaf, Brainstorm, DSSTNE, Neon, Veles, Paddle, etc.

These frameworks differ in their emphasis on production or research, the backends they support, the languages they employ, the balance they strike between user-friendliness and flexibility, etc. In the progress, they duplicate large amounts of work. Many frameworks re-implement array data types, kernels, memory allocators, adjoint operators, etc.

Vision
======

The features of a deep learning framework effectively comprise a compiler pipeline. From this perspective we can take Theano as an example: the arrays are a type system; the Python interface is a language; the computation graph is the intermediate representation; the performance and stability optimizations are the optimizations; replacing nodes with GPU-specific nodes is lowering the IR; the final conversion to CUDA kernels is code generation; etc.

Theano has traditionally taken this view, and refers to itself as a framework that combines aspects from a computer algebra system (CAS) and an optimizing compiler. TensorFlow holds a similar view, whereas frameworks such as Torch tend to eschew these abstractions in favor of a more minimalist approach.

Toolchain
---------

In keeping with the compiler analogy, our vision for Myia is a *domain-specific compiler toolchain*, similar in spirit to LLVM (which defines itself as a "collection of modular and reusable compiler and toolchain technologies"). It would take the form of a set of tools, specifications, and clearly defined APIs targeted at array programming on a variety of backends (CPU and GPU).

Applying this Unix philosophy of modular software development to deep learning frameworks has many benefits. Having an intermediate representation for array programming will allow for retargetability, where a variety of different frontends and backends can benefit from the same set of e.g. gradient operators and numerical optimizations, and each framework will be able to benefit from new features implemented as part of the toolchain.

The second target of Myia is to fully embrace the fact that deep learning frameworks such as Theano/TensorFlow are effectively trying to be domain-specific languages (DSLs) in all but name. However, several common practices in language and compiler design such as a coherent type system, control flow, SSA representation, different IR levels, etc. were never fully exploited.

Components
----------

Components of Myia could include:

* Intermediate representation

  * Language-independent representation of computation graphs (with support for control flow e.g. branching and loops)
  * Array type system
  * Specification of mathematical operators and interface to add more
  * Hint system for device-specific instructions
* Automatic differentiation

  * Specification of adjoint operators
  * A gradient operator for computation graphs
* Optimization

  * Compiler optimizations such as common subexpression elimination, graph simplification (i.e. symbolic simplification), constant propagation, etc.
  * Numerical optimizations for speed/stability
* Backend

  * Backend-specific primitives
  * Abstractions for memory management
  * Implementation of array types
  * Dynamic kernel compilation
* Front-end

  * Python API
  
Many different components have been implemented as part of different frameworks e.g. most frameworks have array data types, a variety of kernels, Theano and TensorFlow have computation graph representations, cuDNN and cuBLAS provide primitives, etc. Ideally these components can be separated out, refactored and re-used as part of the Myia toolchain.

Features
--------

Myia's conception also stems from the desire to implement new features which are currently offered by few of the mainstream frameworks. The hope is that a new architecture will enable these features to be implemented more easily. Two examples are *checkpointing* and a *hybrid interpreter/compiled execution of code*.

Checkpointing
~~~~~~~~~~~~~

Checkpointing is an old technique in automatic differentiation. It involves the dropping of intermediate values during the forward propagation which are recalculated during the backward propagation, trading in performance for memory. It would allow experimenting with models which would otherwise not fit on a single GPU. Having a frontend/backend agnostic implementation would allow a variety of frameworks to benefit from this advanced functionality.

Hybrid interpreter/compiled execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frameworks like Theano and TensorFlow construct a computation graph which is compiled to a single callable function. This approach allows for the optimization of the graph and is more efficient (no operator overloading, no interfacing with the host language). On the other hand it forces the computation graph to include control flow primitives (such as Theano's `scan` function), makes debugging more complicated (since it can no longer be done in the host language), and slows down the development cycle because of the need to compile code.

Frameworks like torch-autograd and Chainer operate entirely in the host language (Lua and Python respectively) and use operator overloading with a tape to implement automatic differentiation. The advantage of this approach is that the control flow can be entirely determined in the host language, and debugging is very easy (e.g. one can simply use `pdb` and inspect the values returned by each operator), but it comes at the cost of performance and memory when naively implemented.

Just as LLVM's IR can be compiled statically or run using just-in-time (JIT) compilation, Myia's IR too should be able to be compiled statically (similarly to Theano) or just-in-time (similar to Chainer), or both (similar to the way Lisp programs can be partially compiled and interpreted). This would allow a user to switch seamlessly between quick development and optimized production code.

Implementation
==============

The implementation details of the entire toolchain are very much up in the air, but initial discussions lead to some ideas.

Intermediate representation
---------------------------

The intermediate representation should have the same requirements as any other medium-level intermediate representation i.e. *accurate* in the sense that it must fully describe the mathematical operations, and *independent* of the source language that generated it (Python, Lua, etc.) as well as the target language (CUDA, OpenCL, CPU, FPGA, etc.).

The computation graphs in Theano and TensorFlow have several of these properties already, although their support for control flow is awkward. In the compiler literature their approach is most similar to Click's `sea of nodes`_ representation (as used in the `FIRM compiler`_ and the JavaScript V8 TurboFan engine). Following those implementations it should be easy to see how we can properly introduce control flow in the IR (i.e. with region and phi nodes). Automatic differentiation can be applied directly on this intermediate representation i.e. the gradient of a Phi node is a conditional and vice versa.

Other ideas include generalizing several tensor operations (inner and outer products, traces, etc.) to Einstein summation nodes, as was done in `Diderot's compiler`_.

Given that the IR can be used to construct and execute graphs at runtime, it should be represented and processed in an efficient way (e.g. using C/C++, a custom binary format, or an in-memory serialization format such as Flatbuffers_). Tools should be provided for serialization and to convert graphs to a human-readable text based representation.

DMLC's `NNVM project`_ introduces a C++ based intermediate representation, but without support for control flow (i.e. it assumes the graph is acyclical).

.. _sea of nodes: http://grothoff.org/christian/teaching/2007/3353/papers/click95simple.pdf
.. _FIRM compiler: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.716.5826&rep=rep1&type=pdf
.. _NNVM project: https://github.com/dmlc/nnvm
.. _Flatbuffers: https://github.com/google/flatbuffers
.. _Diderot's compiler: https://cpc2016.infor.uva.es/wp-content/uploads/2016/06/CPC2016_paper_21-compressed.pdf

Type system
-----------

A type system reduces the chances of bugs appearing and allows for more optimization. At the very least a type system expressing floating-point precision is required for dispatching. However, a type system for multi-dimensional arrays could be made much more expressive using `dependent types`_, `intersection types`_ and subtypes_. For example, one could define:::

  ssymv: array(dim=2, dtype=float32, symmetric=true, shape=(n, n)) x array(dim=1, dtype=float32, shape=(n,)) -> array(dim=1, dtype=float32, shape=(n,))
  
Such a system would allow for dispatching based on e.g. symmetry, perform shape inference, detect type errors based on e.g. data type, and perform optimizations using e.g. the fact that a matrix with one column is equivalent to a vector in some cases.

.. _dependent types: https://en.wikipedia.org/wiki/Dependent_type
.. _intersection types: https://en.wikipedia.org/wiki/Type_system#Intersection_types
.. _subtypes: https://en.wikipedia.org/wiki/Subtyping

Built-ins
---------

The intermediate representation should allow for platform-specific built-ins. This would allow a user to e.g. use a specific convolution kernel, trading off performance for memory. Alternatively this system could take the form of annotating nodes. The backend can then use these annotations during the code generation phase.

Perhaps this system could also be used to label operations as being allowed to operate in-place when executing in interpreter mode (in compiled mode the compiler can determine this by itself).

Modules
-------

Let's use the word *module* to refer to a small computation graph i.e. a set of inputs, a series of operators, and a set of outputs. A large number of tools will operate on these modules. For example, a gradient operator can take a module as input and output an adjoint module that calculates the gradient, or it can output a single module which performs both the forward and backward prop. The optimizer will take a module and output an optimized module that performs the equivalent computation, etc.

Compiled and interpreted mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In compiled mode, we could imagine the entire computation graph being represented using a single module. We can optimize this module, then calculate the gradient, and then once more optimize the module, resulting in a single module which calculates both forward and backward prop as efficiently as possible.

During interpreter mode, we can imagine each operation (or a small set of operations) to produce a single module. This module is differentiated, producing an adjoint module which calculates the gradient. This adjoint module is added to the tape, in the same way that AD is implemented using operatore overloading. During the forward prop we are sure to save all the inputs to the adjoint module in memory, so that we can simply walk through the tape in reverse in order to calculate the gradients.

With this approach one can support both compiled and interpreted mode while using the same set of tools.

Device interface
----------------

The final result of the pipeline is a module that has been optimized and differentiated, but is still device independent. The precise process by which we transform a module into executable code is not clearly defined yet.

In a general compilation pipeline the intermediate representation is often lowered into a device-specific representation before being converted into executable code. In deep learning it is common for computation graphs to be executed on a mix of devices e.g. partially on the GPU and partially on the CPU. Hence, a device-specific representation is problematic. Theano's approach is to keep the same representation, assume the operators are executed on CPU by default, and replace operators with GPU equivalents where possible, inserting memory transfers where needed. Frameworks such as Torch allow the user more fine-grained control, allowing them to specify for each operator whether it should be run on the GPU or CPU. A good solution for Myia would perhaps be to use a hints system to express preferences for devices, which allows the user control while not sacrificing the independece of the IR.

Significant work might be needed to ensure that e.g. loops (which are expressed using conditionals and jumps only) use memory efficiently. This engineering effort is the price of simplifying the control flow in the IR to conditionals and phi nodes only.

Other considerations are e.g. memory allocators, device transfers, maintaing a state (e.g. for streams, cuBLAS handles, etc.).These should be abstracted in a way similar to e.g. Torch and Collenchyma.
