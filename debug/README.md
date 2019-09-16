
# debug

Utility to debug Myia.

You can call this with `<buche> python -m debug <args>`, or alias `python -m debug` to a shortcut like `mdbg`.

```
Debug Myia

Usage:
  dm <command>
     [-f FUNCTION...] [-a ARG...] [-g...]
     [-O OPT...]
     [--config FILE...]
     [--pipeline PIP...]
     [--scalar]
     [--no-beautify]
     [--function-nodes]
     [<rest>...]

Options:
  -f --fn FUNCTION...   The function to run.
  -a --args ARG...      Arguments to feed to the function.
  -g                    Apply gradient once for each occurrence of the flag.
  -c --config FILE...   Use given configuration.
  -O --opt OPT...       Run given optimizations.
  -p --pipeline PIP...  The pipeline to use.
  --scalar              Use the scalar pipeline.
  --no-beautify         Don't beautify graphs.
  --function-nodes      Show individual nodes for functions called.
```

This requires `buche` to be installed: https://github.com/breuleux/buche/releases

The value for most arguments and options can be of the form `module:variable`,
in which case the corresponding module is imported and the variable is read
from it. Multiple values for arguments or types are separated by either `,` or
`;`.

The `-g` flag doesn't do anything right now.


## Examples

Suppose `x.py` contains the following code:

```python

import numpy

x1 = 4
x2 = 8.5
x3 = numpy.random.random((4, 6))

def hibou(x, y):
    return x + y
```

Showing the graph.

```
mdbg show x:hibou
mdbg show x:hibou --pipeline parse,resolve,infer,specialize,opt --types i64,i64
mdbg show x:hibou --pipeline '!opt' --types i64,i64
mdbg show x:hibou --pipeline '!opt' --args x:x2,x:x3
mdbg show x:hibou --pipeline '!opt' --types i64,i64 -O inline
```

The `!opt` pipeline is the same as `parse,resolve,infer,specialize,opt`.


## Custom operations

Suppose `y.py` contains the following code:

```python

from debug import steps

def inferred(o):
    res = o.run(default=[steps.parse,
                         steps.infer,
                         steps.specialize])
    ibuche(res)
```

Then you can run this, and it will print the pipeline results:

```
mdbg y:inferred x:hibou --args x:x2,x:x3
```

You can also run this to get an interactive prompt, and you can click on parts of the results to put them in variables:

```
mdbg -i y:inferred x:hibou --args x:x2,x:x3
```
