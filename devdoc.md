
The Myia Pipeline
=================

When a function is decorated with `@myia` and called with a list of arguments, it roughly goes through the following steps:

* **Parse:** The function is parsed into Myia's untyped IR. Myia's IR is in graph form, with calls pointing directly to their arguments.
* **Infer:** Myia runs an abstract interpreter in order to figure out the possible types for each node. It starts from the types of the arguments provided by the user: if these types are different in the next call, the pipeline will be run anew with the new types.
  * Additional functions may be parsed in this phase, when Myia finds out that it needs them.
  * *Macro expansion* runs in this phase. A macro is a function that takes graph nodes and/or the inferred types for these nodes, and constructs a new subgraph that should replace the original call.
* **Monomorphize:** This step transforms the untyped IR into the typed IR. Monomorphizing is the process of making a version of each function for each possible type signature it may have. Thus, at the end of this step each graph has a unique type signature (monomorphic, as opposed to the original polymorphic Python functions).
* **Simplify types:** During inference, Myia recognizes many structural types: tuples, ADTs, dicts, etc. This step replaces them all with tuples so that the following steps can be simplified. It also transforms untagged union types into tagged union types, and a few other things.
* **Optimize:** Graph transformations and simplifications are applied to the monomorphized graph. Most of these transformations reduce the size of the graph. For example, `x * 1 => x`, or `x + 0 => x`, or if a tuple is built and then immediately indexed, we can replace that by a direct reference to the element (in other words, `(a, b, c)[1] => b`).
  * *Gradient expansion* is performed in this step. The optimizer, upon seeing the expression `J(f)`, will generate a graph `jf` that computes the gradient of graph `f`.
  * *Inlining* is also performed in this step.
* **Compile:** The optimized graph is fed to the backend. Multiple backends are supported and the user is free to choose the one they prefer.
* **Wrap:** This step wraps the compiled function. Mirroring the conversion that occurs in the simplify_types step, it converts the arguments from rich representations such as classes to tuples, and converts the result output by the backend from tuples to the original rich representations, to make it appear seamless to the user.

The pipeline outputs a callable function. By default, when calling a Myia function, the arguments will be converted into the backend's representation (which might involve transferring the data to GPU memory). The return value will also be in the backend's reprensentation. The user can call a method to convert it to Python/Numpy, or they can feed it back to the Myia function, in which case they will not need to be converted (this allows iterative training a model entirely on the GPU).

The standard pipeline is defined in `myia/pipeline/standard.py`.


Parse
-----

The **parse** step takes a normal Python function and converts it to a `Graph`. Not all of Python is supported, but the important parts are.

A Myia `Graph` is a pure functional representation, which means in place operations on Python data, such as setting an element of an array, will not work.

At the moment, all function calls must be connected to the output in some way, which means that e.g. a `print(x)` statement in the middle of the function won't work, because it does not return a value that's needed to compute the output. An experimental feature is being developed which will fix this issue, allowing `print` statements to work as normal, as well as some imperative-style code.


Infer
-----

The **infer** step performs an elaborate feat of type and shape inference.

As a starting point, we take the types of the arguments given to the root function and create a `Context` for the root function that represents a call of that function with these argument types. Each possible input type signature will create a new `Context` so that we can properly support polymorphism.

Each `Primitive`, `Graph`, `Macro`, etc. is associated to an **`Inferrer`**. When the inference engine encounters the call `f(x, y)` it will proceed as follows:

1. Compute the type of `f`. The type of `f` will typically contain all the possible graphs, primitives, macros, etc. that could end up in call position at that location.
2. For each possible operation, we get the inferrer for that operation.
3. We create a tuple of `Reference` object, one for each argument. A `Reference` contains a graph `Node` and a `Context`. It is the sort of object for which we can infer a type.
4. We run each `Inferrer` on the references and collect the results.
5. We merge all the results, to make sure that all calls return compatible results. Usually there is only one possible function, therefore only one result, but in the case of `if` statements and the like there can be more than one.

Now, an `Inferrer` can handle a call in one of two ways:

* **`reroute`**: The inferrer can choose to replace the call with a different one, and then resume inference on the new call. For example, the inferrer for `f(x)` could decide to simply replace `f(x)` by `g(x + 1)`, and then infer the type of the replacement. The inferrer for a `Macro` defines `reroute` so that it calls the macro and returns what the macro returns.
* **`run`** or **`infer`**: The inferrer may also return the type of the output of the call given the types of the inputs. The inferrer does not have to request the types of all the inputs. The default implementation of `run` fetches all the input types and then calls `infer`.

If multiple inference results must be merged, and multiple inferrers decide to reroute the call to different nodes, we consider that they are not compatible and we raise an inference error. This can only happen if a macro is chosen in a value-dependent way, and it is an unlikely situation.

### AbstractValue

`myia/abstract/data.py` defines `AbstractValue` and a whole lot of subclasses. These objects are Myia types. Each `Reference` is associated to one of such object. The `run` function of an inferrer must return an `AbstractValue`.

### async/await

The inferrer repurposes Python's `async` and `await` to perform a kind of data flow based inference.

* `reroute` and `run` are `async` functions.
* To get the type of a `Reference`, one must call `await ref.get()`.

The reason behind the design is that by representing inferrers as coroutines, we can more easily compute the type of a recursive function. In a recursive function, there is typically one branch for the terminating condition which can be inferred immediately, and a second one that has the recursive call and will therefore cause a deadlock (you need to infer the type of the recursive call to infer the type of the branch, but you need the type of the branch to infer the type of the recursive call). If each branch is mapped to a coroutine, though, we can run them in parallel, set the type to the first one to terminate, which will unblock inference for the second. Then we can check that both branches agree.

Monomorphize
------------

After inference, we get a collection of `Context`s for each `Graph`. In one context, the output node might be an integer. In a different context, it might be a float, or a tuple. In one context, we may want to replace a graph node by another, in a different context we may want to replace that same node with something else entirely. The inference engine contains all the information, although it does not replace any nodes, nor does it store the type information in the nodes.

The monomorphize step untangles this mess. At the end of this step, each graph has a unique context, all the nodes are the right ones for that context, and each node now contains its type in the `abstract` attribute.

Simplify types
--------------

This step replaces composite types by tuples, and corresponding operations. For example, `AbstractClass` becomes `AbstractTuple`, `make_record` becomes `make_tuple` and `record_getitem` becomes `tuple_getitem`. This simplifies further processing.

Optimize
--------

The optimizer performs a variety of optimizations such as:

* Simple arithmetic simplifications such as `x * 0 => 0` or `0 + x => x`.
* Structural optimizations such as `(a, b, c)[1] => b`.
* Expanding partial calls: `partial(f, x)(y, z) => f(x, y, z)`.
* Inlining a function that's only used once.
* Inlining a function that only contains a single function call.

Most optimizations are defined using a simple pattern language. Some define an input pattern and a replacement pattern, the more complex ones define an input pattern and a function to compute the replacement.

### Gradient expansion

The expression `J(f)` represents a function `jf` such that, roughly speaking:

```python
jf(J(x), J(y)) == (J(f(x, y)), lambda dout: (df/denv, df/dx, df/dy))
```

The transform of a graph `f` is a new graph, and it can introduce a lot of boilerplate. Therefore we optimize `f` as much as we can before computing the transform. That's why the expansion is here (in theory it could also be done during inference).

### Renormalize

This is currently a pain point in the pipeline because it is a massive waste of time and bug-prone, so it is being rewritten. Basically, optimizations, and especially gradient expansion, cause many nodes to lack a type. Thus we run the infer and monomorphize steps of the pipeline a few times during optimizations.

We are working on replacing this with lightweight incremental inference.

Compile
-------

Once the graph is optimized it is given to the compiler for the user-chosen backend. There is currently a Relay backend and a PyTorch backend.

Operations
==========

The central hub for Myia's operations is `myia/operations/__init__.py`, which can be regenerated (e.g. when adding new operations) with `scripts/regen.py`. Some of these operations map to primitive, others map to macros, and others map to functions that are written in Python and meant to be parsed by Myia. Here's a lexicon to clarify things:

* **`Operation`**: This is a specification for an operation. It has a name and a set of defaults, including a default mapping (implementation), but it does not commit to a particular implementation. For example, `myia.operations.add` is the `+` operator, which by default will mirror Python's `+` including checking the `__add__` and `__radd__` attributes. If you wanted to, though, you could decide to override this and map it directly to the `scalar_add` primitive.
  * The list of operations is in `myia/operations/__init__.py`. Note that the defaults for each entry is the name of a submodule in `myia/operations`, where the defaults are actually defined. They are loaded lazily.

The following are things an `Operation` can map to:

* **`Primitive`**: This represents an atomic operation, usually one that should be implemented by all backends. There are also non-backend Primitives, which should be eliminated by pipeline steps before the compilation step.
  * **`BackendPrimitive`**: Usually defines a *Python implementation*; an *inferrer*, which takes the input types and returns the output type; and a *backpropagator* which returns the gradients of the inputs given the gradient of the output.
  * **`InferencePrimitive`**: Defines an inferrer only. These primitives typically handle types that disappear in the simplify_types step, thus they are replaced during that step by corresponding operations on tuples or other types, or removed altogether.
  * **`PlaceholderPrimitive`**: Currently only `J` and `Jinv`. They are eliminated during optimization, more precisely during gradient expansion.
  * The list of primitive is in `myia/operations/primitives.py`. As for operations, the defaults are implemented in submodules (it is possible to override them and e.g. use a different backpropagator for an existing primitive, but it is not recommended).
  * Note that there is a specific `Operation` for each primitive, which maps to the `Primitive`, so for example `myia.operations.scalar_add` maps to `myia.primitives.scalar_add`. They are not exactly the same thing: the latter will always give you the primitive, but it is possible (but generally not recommended) to remap the former to something else.
* **`Graph`**: A Graph is generally the result of parsing a Python function with Myia. So e.g. the `array_add` operation is simply a Graph that calls the `array_map` primitive on `scalar_add` and two arrays.
* **`MetaGraph`**: A MetaGraph is essentially a Graph generator. Given a list of input types, a MetaGraph can generate a Graph that's specialized to these types. For example, if it is given a tuple of n elements as input, it could directly generate a Graph that has n inputs and adds them together (so there is a distinct graph for each value of n).
* **`Macro`**: A Macro is a bit similar to a MetaGraph in the sense that it can generate code, but it is more powerful. A Macro takes as its inputs the inferrer's Reference objects: it can see the graph nodes corresponding to its inputs, as well as their inferred types, and it must return a graph node to replaces the macro call.
  * **`@myia_static`** is a decorator that creates a function that must be called at compile time. It is a particular (limited) kind of `Macro`. A `@myia_static` function requires the values of all of its arguments to be inferrable statically. If so, it is run with these values, and its return value is wrapped in a constant node and substituted for the myia_static call.

Evolution of operations
-----------------------

The kind of operations a `Graph` may call varies at each stage of the pipeline. Unless specified otherwise, we list the kinds of operations that are found in the graphs at the *end* of each step:

* **Parse:** **`Operation`, `Graph`**. The parser does not introduce primitives directly.
  * Note that if you use a function or global variable `f` in a function parsed by Myia, the parser will generate the call `myia.operations.resolve(global_ns, "f")`. It will not directly insert the value of `f`. So even if you define a function or a macro and call it, what you actually get after this step is a call to the `resolve` operation.
* **Infer**
  * *During* inference: **`Operation`, `Graph`, `Primitive`, `MetaGraph`, `Macro`**. The `resolve` operation mentioned above is handled in this step, so this is where Myia actually discovers all the functions and macros that are being called. MetaGraphs and Macros only exist transiently during this process, because they are first discovered and then immediately expanded.
  * *After* inference: **`Graph`, `InferencePrimitive`, `BackendPrimitive`, `PlaceholderPrimitive`**. Operations are resolved to their mapping, MetaGraphs and Macros are expanded.
* **Monomorphize:** **`Graph`, `InferencePrimitive`, `BackendPrimitive`, `PlaceholderPrimitive`**.
* **Simplify types:** **`Graph`, `BackendPrimitive`, `PlaceholderPrimitive`**. Inference primitives are replaced by corresponding backend primitives in this step.
* **Optimize:** **`Graph`, `BackendPrimitive`**. Placeholder primitives (namely `J` and `Jinv`) are expanded and eliminated in this step.

For the **compile** step, the backend thus receives a root `Graph` that calls `BackendPrimitive`s as well as other `Graph`s.


Defining a new operation
------------------------

`scripts/new_operation.py` generates the boilerplate for a new operation. Use it  like this:

```bash
python scripts/new_operation.py template OPERATION=name ARGUMENTS="arg1, arg2, ..."
```

There are four templates so far:

* `prim`: Create a new BackendPrimitive.
* `infprim`: Create a new InferencePrimitive.
* `macro`: Create a new Macro.
* `op`: Create a new operation (generic).

This will create a file for the operation with some boilerplate, a test file, and will also regenerate the master list of operations and primitives. Note that for backend primitives, code will also need to be added to the backends for them to actually work.
