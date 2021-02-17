
# Myia rewrite

This is an attempt to rewrite Myia using essentially the same methods, but in a more principled way and learning from previous mistakes.


## ir

### parent

I would vote to add a `parent` field to `Graph` which is set by the parser using the nesting information in the source code. This will ease manipulations and may also help identify mistakes.

### edges

One thing I had wanted for the IR at the beginning, and I still think it would be a good idea, would be to have actual Edge objects. Part of the reason is that it allows for extra debug information: consider the code `x = f(a); g(x); h(x)`. Without Edge, there is only one location for `x`, its definition corresponding to `f(a)`. With Edge, we can also store location information for each use of `x`. So in this case `x` would have one Node and two Edges. So if e.g. there is a type error for the first argument of `g(x, y, z)` we can point to the use of `x` rather than just its definition.

An Edge could also contain extra information, such as a label for keyword arguments. The Universe argument could have a special label, e.g. `$UNIVERSE` rather than be positional, which would add a layer of safety when we manipulate these graphs (the universe is less likely to end up where it shouldn't if it's a specially crafted keyword argument). A `vararg` flag on an Edge could make it an `*argument`.

With this change, the `uses` set would just be a set of Edges instead of a set of (index, node) (or (node, index)?) I would also move to make `inputs` dict of `{edge.label: edge}`, where the function would be `node.inputs["$CALL"]` instead of `node.inputs[0]`, and the rest would be positional or keyword arguments.

Eventually we could simplify the graph by eliminating all non-numeric, non-$CALL labels, as well as all vararg flags.

Lastly, we could simplify the Universe transform in the parser by replacing it with a `$SEQUENCE` edge on each node which simply points to the previous node in the normal order of evaluation. Edges with that label would be ignored by the inferrer or backends, but a straightforward transform could use it to thread `$UNIVERSE` edges through the execution.

### manager

I would try to implement as much as possible without resorting to the manager.

## parser

If the IR is modified, the parser would have to be ported to the refreshed version. I would try to include the Universe transform right off the bat.

I would argue for one simplification: instead of generating `resolve(operator_ns, "add")` for the addition operator we can really just generate `Constant(operator.add)`. Special Myia operations like `myia_next` would point directly to stdlib Myia functions that implement the behavior in pure Python. `resolve` should remain to resolve global variables, which could change, or could be defined after the parse.


## Python backend

The Python backend, which generates Python code, should probably work on any graph at any stage. A few adjustments from the existing one would enable this:

* Evaluate calls to `resolve` immediately.
* Translate built-ins like `operator.add(x, y)` to `x + y`, `getattr(x, "name")` to `x.name`, for prettiness.
* When there is a reference to an external function, use the `__module__` and `__qualname__` fields to generate the proper import.
* Any value that cannot be imported that way can be given a mangled name. The backend can return Python code along with a dictionary of globals, and the code can be evaluated with e.g. `exec(code, glb, lcl)` where `glb` is the dictionary returned by the backend.
* It would be good to have a "source map" from the original code to the generated code. For this reason it might be advisable to generate AST nodes instead of textual source code, which would let us associate the original source code locations to the generated nodes.
* Use `myia.utils.info.Labeler` with the proper options set to generate valid Python names.


## abstract

`abstract` will contain the data types and manipulation methods to implement the type system for Myia. They should organized better than in the original version to maximize code reuse. Here's an idea:

* `Tracks` is a flexible mapping from names like "value" or "shape" to values that are **NOT** abstract types. The following tracks may be defined:
  * `value` for a value that's known at compile time
  * `shape` for the shape of an array and whatnot
  * `interface` for the Python type that represents the interface. This track will be eliminated prior to optimization
* `AbstractAtom(tracks)` represents leaf types (integers, strings, etc.)
* `AbstractParameterized(elem1, ..., tracks)` represents a type that's parameterized by a finite number of other types
    * `AbstractStructure(elem1, ..., tracks)` represents a product type (tuple, struct, etc.) If some fields are named, the interface track will contain the necessary information.
    * `AbstractUniform(elem, tracks)` represents an array, list or dict where all elements have the same type.
    * `AbstractArrow(args, ret, tracks)` represents a function from argument types to a return type.
* `AbstractUnion(opt1, ..., tracks)` represents a sum type (a union, the set of functions that may be called at a call site). It will be eliminated in favor of AbstractTaggedUnion prior to optimization.
* `AbstractTaggedUnion(opt1, ..., tracks)` represents a sum type, but where each possible element is tagged. I'm not entirely sure if it can share a parent with `AbstractUnion`
* `AbstractInferrer(func, tracks)` implements an arbitrary function from argument types to a return type. We aim to eliminate all of these in favor of `AbstractArrow` after monomorphization.


## inferrer

It seems that the async/await approach is too much of a pain to debug, understand and extend, and a better one may be needed. I have two ideas:

1. Generator-based: instead of `typ = await ref.get()`, use something like `typ = yield RequestType(ref)`. It is not super different from async/await in principle, but there's no need to deal with all the stdlib asyncio code. It would be more like a real coroutine where we know exactly what we are yielding execution to.

   With this strategy, we can also more easily keep track of what requests what and perhaps generate a dependency graph that we can visualize with snektalk or whatnot.


2. Code-transform-based: in the same way that we have a generic transform to thread Universe, we could have a generic transform to create an inferrer from a graph. Then we can use the Python backend to run it. For example:

    ```python
    a = f(x)
    b = g(y)
    return h(a, b)

    =>

    T_a = inferrer(f)(T_x)
    T_b = inferrer(g)(T_y)
    return inferrer(h)(T_a, T_b)
    ```

    One advantage is that it has a similar structure to the program and is easier to trace. Calling it on a higher order function would naturally produce an inferrer for the closure. Another advantage is that it would be possible to compile it and produce fast specialized inferrers.

    The disadvantages are that it's not clear how to handle recursion properly, and a priori it can't do macros or associate types to nodes, unless nodes are added as arguments, which I suppose they could be.
