
# Developer help

## Project structure

To help grok the structure, project files are tagged in the following listing:

* `!` means this file is **required reading** to understand Myia.
* `S` means this file contains code related to **syntax**.
* `I` means this file contains code related to **the interpreter**.
* `G` means this file contains code related to **gradients**.
* `F` means this file contains code related to **inference**.
* `O` means this file contains code related to **optimization**.
* `T` means this file contains code related to **testing**.

```
myia/                            # Source code for Myia
    __main__                 T   # Entry point for myia's CLI
    front              !         # @myia decorator
    impl/                        # Implementations of primitives
        flow_all           F     # Implementations for inference/dfa
        impl_abstract      F     # Implementations for inference/avm
        impl_bprop       IG      # Backpropagators for interpret/vm
        impl_interp      IG      # Implementations for interpret/vm
        main           ! I       # Utilities for implementations
        proj               F     # Utilities for inferrers
        proj_shape         F     # Shape inference for inference/avm
        proj_type          F     # Type inference for inference/avm
    inference/                   # Inference (values, types, shapes)
        avm                F T   # Abstract virtual machine
        dfa                F T   # Dataflow analysis
        types              F     # Type representations
    interpret/                   # Interpreter
        vmutil           I F     # Translate AST to VM instructions
        vm               I       # Stack-based virtual machine
    ir/                          # Graph IR and opts
        convert         S        # Convert from LambdaNode to IRGraph
        graph          !S        # Definition of IRNode and IRGraph
        graph.css                # Stylesheet for display of IRGraph
        opt                 O    # Closure (un)conversion
        pattern             O    # Implements pattern optimizations
    legacy_interpret/...         # Legacy vesion of interpret/ for inference/
    lib                ! I       # Impl of Record, Closure, StructuralMap
    parse               S        # Parse Python code, produce IR
    stx/                         # IR
        about          !S        # Track what nodes are about what other nodes
        env            !S        # Environments and symbol generation
        nodes          !S        # Definitions of the IR's nodes
        transform                # Boilerplate for transforms
    symbols            !SIG      # Symbols for builtins
    transform/                   # Code transformations
        a_normal        S G      # Convert IR to A-normal form
        grad           !  G      # Gradient transform
    util/                        # Utilities
        buche                T   # Helpers for the Buche logger
        debug                T   # Buche-based debugger
        event              F T   # Event framework (listeners, emit events)
        misc                     # Various helper functions
        myia.css                 # Stylesheet to display the IR with Buche
    validate             IG  T   # Miscellaneous validators
```

## Pipeline

The current pipeline for Myia is defined in `myia/front.py` as an instance of `UniversePipelineGenerator` (defined in `myia/lib.py`). The `@myia` decorator of a function `f` is roughly equivalent to the statement `f = myia.front.standard_universe[f]`.

A pipeline is a stack of `Universe`. What a `Universe` does is that it transforms a value into a suitable representation at this stage in the pipeline, and caches the result. For example, the `IRUniverse` transforms a function into an instance of `IRGraph`, and caches the graph to avoid generating it more than once.

A `Universe` can have another universe as a parent, which means that it first transforms a value through its parent, and then transforms the result. The current pipeline is as follows, with each Universe having the previous one as a parent:

* `PythonUniverse` (no parent): Resolves a global `Symbol` to the corresponding value. For example, the global symbol with label `f` and namespace `global::file.py` resolves to the value of the global variable `f` in the module defined by `file.py`. It does not resolve symbols with namespace `global::builtin`.
* `SymbolicUniverse`: Transforms functions into `LambdaNode` instances, which is the old IR. This is a temporary stopgap.
* `IRUniverse`: Transforms `LambdaNode` into `IRGraph`, which is the new IR.
* `OptimizedUniverse`: Optimizes `IRGraph` through various passes. More than one `OptimizedUniverse` can be stacked, since they operate on the same representation.
* `VMUniverse`: Makes `VMFunction` from `IRGraph`, where operations are linearized and expressed in a way that can be run with a `VM`. While this is not the case at the moment, `VMUniverse` is also intended to transform values such as scalars or numpy arrays into the representation understood by the primitives.
* `EvaluationUniverse`: Makes `CallableVMFunction`, which is the interface meant for the end user. When applicable, values from the `VMUniverse` are converted to Python scalars, numpy ndarrays, etc. as expected by the user.

Each universe type may have options, although most don't at the moment. It is possible (although untested) to have multiple independent pipelines running at the same time, and they may share the first few stages, so it is possible to test e.g. multiple optimization schemes on the same code without reparsing.


## Old representation

(See next section for new representation)

Myia's representation is defined in `myia/stx/nodes.py` and consists of eight node types:

* `Apply` is a function application.
* `Begin` is a sequence of expressions, the last of which is returned.
* `Closure` is a partial application of a function on a list of arguments.
* `Lambda` is a function, with a list of arguments and a body.
* `Let` defines a list of variable/value bindings and a body.
* `Symbol` is a symbolic variable (but generate them using `GenSym` instances).
* `Tuple` is a literal tuple.
* `Value` is a literal.

This should probably be simplified (`Begin` is sort of redundant with `Let`).

The `Grad` transform is defined on this representation. This representation is transformed into the new representation prior to being run in the VM.


## New representation

Myia's representation is defined in `myia/ir/graph.py`. It consists of one node type (`IRNode`) and one graph class (`IRGraph`).

### IRNode

`IRNode` has the following fields:

* `graph: IRGraph` points to the graph to which this node belongs.
* `tag: Symbol` is the name of this node, e.g. the name of a corresponding input or intermediate variable in the original source code.
* `fn: IRNode` points to a node returning the function to call to compute this node's value.
* `inputs: List[IRNode]` points to the arguments of the call.
* `value: Any` is the (static) value that this node always returns, or the special token `NO_VALUE`.
* `users: Set[(Role, IRNode)]` is a set of nodes that depend on this one. `Role` is one of `FN` (this node returns a callable for another node) or `IN(i)` (this node is the ith input of another node).

An `IRNode` can be:

* A computation (`node.is_computation()`)
  - fn != None, inputs == list of inputs, value == NO_VALUE
* A numeric/string/etc. constant (`node.is_constant()`)
  - fn == None, value == the constant
* A builtin function like add or multiply (`node.is_builtin()`)
  - fn == None, value == the Symbol for the builtin
* A pointer to another Myia function (`node.is_graph()`)
  - fn == None, value == an IRGraph
* An input (`node.is_input()`)
  - fn == None, value == NO_VALUE

Note that `node.is_constant()` is true if the node is a builtin or a graph, so it is a superset of those conditions.

### IRGraph

`IRGraph` has the following fields:

* `parent: IRGraph` points to the graph to which this graph belongs (if this graph is a closure), or `None` (if this graph is a top-level function).
* `tag: Symbol` is the name of this graph (i.e. the name of the function it represents).
* `inputs: List[IRNode]` is a list of inputs for this graph. `node.is_input()` must be true for all of them.
* `output: IRNode` is the output of this graph. To return multiple outputs, they must be wrapped in a tuple.


## Symbols and implementations

Myia's primitives are defined in `myia.symbols`. They are not classes or objects (like Theano ops, for example), which leaves their semantics more open. A `Symbol` is a simple structure that has a `label`, a `namespace` (which is `global` or `builtin` for the primitives), a `version` and optionally a `relation` to another symbol (e.g. gradient-of, a-normal-form-of, and so on). Two symbols are equal if all of these attributes are equal.

Implementations are defined in `myia.impl` and are typically defined using decorators that associate a symbol to a function under certain semantics. The most straightforward implementations are in `myia.impl.impl_interp`. Other files implement backpropagators and inferrers.

Notably, `myia.impl.impl_abstract` defines "abstract" implementations that can operate on the unknown value `ANY` and return multiple results in case of uncertainty. These are used by `myia.inference.avm.AVM`, an abstract virtual machine, to perform inference and backtracking.

In essence, this design choice makes it easier to experiment with new inferrers and new ways to interpret operations, because they can be isolated and fully defined at a single location. On the other hand, code related to any particular primitive has to be split into many files.


## Interpreter

`myia.interpret.vm` defines a stack-based virtual machine for Myia, which is not intended to be high performance, but performs tail call optimization and thus does not suffer from Python's small stack.


## Gradients

`myia.grad` defines the `Grad` transformer. The method is based on [this paper](http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf) ([1]).

### Gradient functions

For each function transformed by Grad, several auxiliary functions will be created. These are the ones you need to know about:

* `↑f` is the "tagged" version of `f`. In [1] this is `f` with a top-left harpoon, i.e. f⃐. `↑f` returns two values: first, it returns the "tagged" version of `f`'s normal output. Second, it returns the backpropagator closure `♢f` (see below).

* `♢f` is the backpropagator for `f`. In [1] this is `f` with a bar on top, i.e. f̄. Its input is the gradient with respect to its output (let's call it `∇f`). Its output is the tuple `(closure_grads, *argument_grads)`. The latter part is what typically interests us, but the former part is required to make everything run smoothly (see Partial Application section below).

  Note that `♢f` is not a top-level function, but a closure over the real top-level function `♦f` (see below).

* `♦f` is the closure-converted backpropagator function for `f`. Being closure-converted means it has no free variables. This is an implementation detail.

### Gradient variables

In the generated code you will see the following variables:

* `↑x` is the "tagged" version of `x`. If `x` is a scalar, that's the identity, if `x` is a data structure this applies the tag on every member, and if `x` is a function, see above.

* `♢x` is the "backpropagator" for `x`.

* `∇x` is the "sensitivity" with respect to `x`. In other words, this is where we accumulate the gradient for `x`.

In a nutshell, when there is an assignment like `x = f(y, z)` in the code:

* In the forward pass, we generate:
  ```python
  ↑x, ♢x = ↑f(↑y, ↑z)   # forward computation
  ```

* In the backward pass, we generate:
  ```python
  ∇x = zeros_like(x)    # initialization
  ...
  ∇f, ∇y, ∇z += ♢x(∇x)  # propagation
  ```

Note that the statements for the propagation are generated in reverse order, so if `x` is an input to other function calls later in the code, the gradient will accumulate into `∇x` *before* it is used to accumulate into `∇f, ∇y, ∇z`, starting from `∇out`, the input to the backpropagator function. That's why it's called *back*prop.

### Gradient example

In a nutshell, supposing we have the following function:

```python
z = 10  # free variable
def f(x, y):
    a = g(x, y, z)
    b = h(a, x)
    return b
```

Then we will get something like this:

```python
↑z = 10  # free variable

def ♦f(♢a, ♢b, ∇b):
    # The zeros should be the same "shape" as g, x, y, ...
    zero_init(∇g, ∇x, ∇y, ∇z, ∇h, ∇a)
    # Backpropagation, operates in reverse order: propagate through h, then
    # through g. Notice:
    # * We have gradient terms for g and h, because they could be closures
    #   or partial applications, and we must track the contributions.
    # * The left-hand side looks just like the function application.
    #   h(a, x) becomes ∇h, ∇a, ∇x += ...
    # * The right-hand side is also very easy to remember, it's bprop(grad)
    #   for each variable you set in the original function, in reverse order.
    ∇h, ∇a, ∇x += ♢b(∇b)
    ∇g, ∇x, ∇y, ∇z += ♢a(∇a)
    # Note that ∇z is stashed in the first return value. Gradients for all
    # of f's free variables must go there.
    return ((∇z,), ∇x, ∇y)

def ↑f(↑x, ↑y):
    # The "tagged" functions ↑g and ↑h give us both tagged forward results
    # and backpropagators.
    ↑a, ♢a = ↑g(↑x, ↑y, ↑z)
    ↑b, ♢b = ↑h(↑a, ↑x)
    def ♢f(∇f):
        # Closure on ♦f
        return ♦f(♢a, ♢b, ∇f)
    # We return the tagged original return value and a backpropagator.
    return ↑b, ♢f
```

The reality is a bit more complicated, but not by much. Take note that we transform functions to a-normal form before we run Grad. In a-normal form, all statements look like `variable1 = fn(variable2, ...)`. No nested expressions.

We accumulate gradients for functions as well, because they may be closures. If there is a closure `f` in the code, and `x` and `y` are its free variables, then we will simply generate something like this:

```python
∇x, ∇y = ∇f
```

This allows us to recuperate contributions made by calls to `f`.

### Partial application

Closures in Myia are compiled to partial applications, and we allow partial applications to primitives (`while` generates a partial application to `identity`). This creates a subtlety in the interaction with backpropagators.

The backpropagator for `f` should return `(closure_grads, *argument_grads)`. Now, if `f` has no free variables then `closure_grads` is, quite naturally, the empty tuple `()`. However, note that this is the case of all the functions Myia compiles, because the free variables are prepended to the list of arguments.

When we make a partial application of `f` on its first n arguments, we basically state that these n arguments are "free variables". Concretely, that means we need the first n elements of `argument_grads` to *move* to the end of `closure_grads` in the backpropagator for the partial.

We could do this by taking the return value of a backpropagator and fudging it appropriately (we did at first), but that creates a lot of crud in the graph and it's cleaner to do it directly. What this means is:

* The Grad class takes a `nargs_closure` argument stating how many arguments at the beginning are free variables.
* Gradients of primitives *also* require an `nargs_closure` parameter, because we can--and do--take partials of them, and the same logic must apply. This is implemented using the `GRAD` "macro", which generates the right return value depending on an internal parameter.
* Thus the `grad` field of both `PrimitiveImpl` and `FunctionImpl` is actually a function of one argument, `nargs_closure` (integer). Its output is cached to avoid needless recomputation.


## Inference

Two modules in Myia perform inference, although they do it a bit differently.

**`myia.inference.dfa`** implements a dataflow/control flow analysis (I think it's essentially 0-CFA). In a nutshell, it tracks which "values" flow to which places using simple rules such as "if a value flows to the body of a function, then it flows to all applications of that function". These rules are run until equilibrium (we know there is an equilibrium because there is a finite number of possible values to propagate).

What we get from this:

* We can track which functions might be called at a given call site.
* We can propagate types.
* Given information we want to know about a certain node, we can propagate what information we need about other nodes in order to compute it.

That analysis is not very precise, however, because the information propagated to a node is context-independent: for example, we can know that `x` and `y` might be `Float32` or an `Int64`, but we can't infer whether their types are correlated. It shouldn't be too difficult to upgrade the inferrer to keep that information (aka 1-CFA -- k-CFA means k layers of context), although it seems that this blows up complexity to EXPTIME (but perhaps not -- [this paper](http://matt.might.net/papers/might2010mcfa.pdf) suggests that 1-CFA for OOP languages is tractable because OO languages collapse problematic contexts by creating closures explicitly (i.e. "new" operations), but Myia's representation does create closures explicitly. I have to read more carefully to see whether this applies to Myia or not, but in any case this seems like a relevant paper.)

Anyway, there is a second inference algorithm, which depends on the DFA. The first pass, performed by the DFA, is that given something we want to know, e.g. the `shape` of a node, we first propagate what we need to know about every other node. For example, the shape of `a + b` depends on the shapes of `a` and of `b`, whereas the shape of `a if x else b` depends on the shapes of `a` and of `b`, but on the *value* of `x`.

Once that information is propagated we can use the second algorithm:

**`myia.inference.avm`** (Abstract VM) is a modified version of the interpreter which can save checkpoints and do backtracking. Where the normal intepreter would work with some value `x`, the abstract interpreter works with an `AbstractValue({type: <type(x)>, shape: <shape(x)>, ...})`. The abstract value can contain the real value, if there is one available, but otherwise it can simply contain information *about* the value, namely what the result of certain operations on it would be. For every node, the abstract interpreter then tries to compute the information it needs. For that purpose it has access to "projector" functions, for example a function that can output the shape of `a + b` given the shapes of `a` and `b`.

The AVM can do backtracking: whenever it pops a `Fork(value1, value2, ...)` instance from the stack, it saves the current state, executes as normal using the first value, and once it is done, it will come back and execute with the second value, and so on. A `switch` (note: `if` generates `switch`) on an unknown (`ANY`) conditional pushes a `Fork`, which is currently the only situation where that happens.

The AVM memoizes the results of function calls, with a caveat: since it may operate on unknown values, which might cause exploring both branches of a conditional, it might end up recursively calling `f(ANY)` from `f(ANY)`. To avoid an infinite loop, when the AVM encounters such a call, it will `Fork` on all the results the call could have had so far, and it will save a conditional checkpoint for any new values discovered in the future. The procedure terminates if the set of possible return values is finite, even if there is a potential infinite loop.

Next improvement (partly done as of 2017/09/06 -- there are some complications to making it more general) would be value decay: when the values of `a` and `b` are known, the abstract interpreter will compute the value of `a + b`, but it should keep track of how many operations were executed to obtain a value, and widen the result to `ANY` after a certain number of steps. That would bound the possible exploration depth by making sure that all of the arguments to all functions either stagnate or become `ANY`.

The current implementation still has issues, e.g. if `f(x)` may either return `1` or `2`, it will yield `3` as a possible value for `f(x) + f(x)`, although that's not possible, because it doesn't keep track of inconsistent instantiations. Unclear whether that's a big deal.


## Optimization

Optimizations operate on the graph IR (new representation). Multiple optimization passes can be run by `OptimizedUniverse`. By default, no passes are run, but `myia/ir/pattern.py` contains a few optimizations that are known to work, such as copy elimination, inlining, `(x, y, ...)[0] => x` or multiplication by one.

Most optimizations are pattern optimizations, which require a pattern in s-expression form and are run by an `EquilibriumPass`. For example, this is the code for the pattern that does `(x, y, ...)[0] => x`:

```python
@pattern_opt(builtins.index, (builtins.mktuple, X, ...), V)
def index_into_tuple(univ, node, X, V):
    return X[int(V.value)]
```

`X` and `V` are pre-defined variables. `X` matches anything, and when followed by `...` it matches an arbitrary number of arguments. `V` matches a constant. They are defined as follows:

```python
from ..inference.types import var

X = var('X')
V = var('V', lambda node: node.is_constant())
```

The variables used in a pattern become arguments with the same name in the decorated function. They are always either `IRNode` instances, or a `list` of zero or more `IRNode` if they are followed by `...` in the pattern.

`EquilibriumPass` is meant to take a set of `PatternOpt` (functions as decorated above) and apply them over and over in some arbitrary order until none can be applied. It is therefore important to make sure the set of optimizations is strongly normalizing (invariant to the order in which they are applied). This being said, `EquilibriumPass` is not very well tested and may still be buggy.


## Buche

The [Buche](https://github.com/breuleux/buche) logger is used to show Myia's internal representations. Assuming you have a file named `pow10.py` that contains the following code:

```python
def test_pow10(x):
    v = x
    i = 0
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v
```

Then you can test its gradient and also view all the intermediate functions created by Myia as well as their transforms with the following command:

```bash
buche python -m myia inspect mytest.py --args 4 --mode grad --decls
```

Alternatively, you can test functions from the test suite directly, if you are in the project's root directory:

```bash
buche python -m myia inspect tests.test_grad:test_pow10 --args 4 --mode grad --decls
```

An object's representation in Buche is returned by its `__hrepr__(H, hrepr)` method. Documentation on how to implement that method can be found [here](https://github.com/breuleux/hrepr).

You can use buche in your code to print pretty much any Python object. See [here](https://github.com/breuleux/pybuche) for some sample code.

The following classes have special display methods for Buche:

* `buche(myia.stx.nodes.Lambda(...))` shows Myia's functional IR.
* `buche(myia.ir.graph.IRGraph(...))` displays Myia's graph IR using the JavaScript library `cytoscape`.
