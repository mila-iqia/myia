# Myia

I would like to implement a prototype. More specifically, a vertical prototype (i.e. the entire pipeline with as little code as possible) that can execute the `sum(tanh(x))` example from the roadmap. We can then build on this prototype and base our discussions of it.

As a first step we do the forward prop only (both compiled and interpreted), after that we add the backward prop.

## Forward prop

### API

* Minimum Python package with `sum` and `tanh` functions and a tensor type
* This API outputs the IR representation

### IR

* Define a minimal schema using FlatBuffers (or possibly Cap'n Proto)
* Ensure that the IR can be transfered incrementally, since the interpreted mode should receive one operator at a time

### Compiled mode

#### Optimizer

* Receives the entire IR
* Should add a hint that `tanh` can be done in-place (or should there be a second IR with memory allocations?)

#### Execution engine

* Takes the IR with the hint and returns a single function that performs `tanh` in-place and performs the reduction
  * Use e.g. Torch kernels, NumPy, whatever for computation
* Somehow needs an interface with the host language so that it can be called

### Interpreter mode

#### Execution engine

* Receives one operator in IR form at a time
* Receives input data from the host language
* Executes each operator immediately, holds the necessary intermediate data in memory
* Returns the data requested and clears memory

*Note: How and where the Python interface is between these parts is unclear, but the execution engine, optimizer, etc. should all operate independently from the actual API.*
