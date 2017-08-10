"""
Validation and testing functionality.

* Sanity checks, e.g. ``missing_source`` or ``unbound``
* Comparing results of Python and Myia implementations.
* Estimate gradients with finite differences for comparison
  purposes.
"""


from typing import Iterable, Set, Tuple as TupleT, \
    Callable, Dict, List, Any, Union

from .ast import MyiaASTNode, Symbol, Lambda, Let, ParseEnv
from .compile import a_normal
from .front import parse_source, parse_function
from .grad import Grad, one
from .interpret import evaluate, root_globals


def missing_source(node: MyiaASTNode) -> Iterable[MyiaASTNode]:
    """
    Yield all nodes that don't have a location set.
    """
    if not node.location:
        yield node
    for child in node.children():
        yield from missing_source(child)


def unbound(node: MyiaASTNode,
            avail: Set[Symbol] = None) -> Iterable[Symbol]:
    """
    Yield all symbols that are not bound by their enclosing
    Lambda (excluding globals/builtins).
    """
    if avail is None:
        avail = set()
    if isinstance(node, Symbol):
        if node.namespace not in {'global', 'builtin'} \
                and node not in avail:
            yield node
    elif isinstance(node, Lambda):
        yield from unbound(node.body, set(node.args))
    elif isinstance(node, Let):
        avail = set(avail)
        for s, v in node.bindings:
            yield from unbound(v, avail)
            avail.add(s)
        yield from unbound(node.body, avail)
    else:
        for child in node.children():
            yield from unbound(child, avail)


# The variation applied on an input in either direction to estimate
# the gradient.
eps = 1e-10

# The tolerance for the difference between the estimation and the
# computed gradient.
rel_error = 1e-03


class GradTester:
    """
    Test a computed gradient against a finite differences estimate
    of the gradient.

    Arguments:
        fn: The function to test against.
        gfn: The function to compute the gradient.
        args: The point in the function's domain where we want
            to estimate the gradient.
        argnames: The names of the arguments.
        outnames: The names of the outputs.
    """
    def __init__(self,
                 fn: Callable,
                 gfn: Callable,
                 args: List[Any],
                 argnames: List[str],
                 outnames: List[str] = None) -> None:
        self.fn = fn
        self.gfn = gfn
        self.args = args
        self.argnames = argnames
        out = fn(*args)
        outname = fn.__name__
        if isinstance(out, tuple):
            self.outnames = list(f'{outname}_{i+1}' for i in range(len(out)))
            self.out = out
            self.wrap = lambda x: x
            self.unwrap = lambda x: x
        else:
            if outnames is None:
                self.outnames = [outname]
            else:
                self.outnames = outnames
            self.out = (out,)
            self.wrap = lambda x: (x,)
            self.unwrap = lambda x: x[0]
        if any(not isinstance(x, (int, float)) for x in self.out):
            raise TypeError('Can only compute gradient'
                            ' for a tuple of outputs.')
        self.nin = len(self.argnames)
        self.nout = len(self.outnames)

    def compute_exact(self) -> Dict[str, float]:
        """
        Compute the exact gradient.

        Returns:
            A dictionary that maps d<outname>/d<argname> to the
            gradient computed by gfn on args.
        """
        results = {}
        for i, outname in enumerate(self.outnames):
            out_sen = tuple((1 if k == i else 0) for k in range(self.nout))
            grads = self.gfn(self.unwrap(out_sen))[1:]
            for j, argname in enumerate(self.argnames):
                results[f'd{outname}/d{argname}'] = grads[j]
        self.exact = results
        return results

    def compute_finite_diff(self) -> Dict[str, float]:
        """
        Compute the finite differences gradient.

        Returns:
            A dictionary that maps d<outname>/d<argname> to the
            gradient computed by finite difference with fn on args.
        """
        results = {}
        for i, argname in enumerate(self.argnames):
            argsl = list(self.args)
            argsl[i] -= eps
            tup_r1 = self.wrap(self.fn(*argsl))
            argsl[i] += 2 * eps
            tup_r2 = self.wrap(self.fn(*argsl))
            d = tuple((r2 - r1) / (2 * eps) for r1, r2 in zip(tup_r1, tup_r2))
            for j, outname in enumerate(self.outnames):
                results[f'd{outname}/d{argname}'] = d[j]
        self.finite_diff = results
        return results

    def compare(self) -> Dict[str, Dict]:
        """
        Compare the exact gradients to the estimated ones.

        Returns:
            A dictionary that maps d<outname>/d<argname> to a dictionary
            that contains both gradients and a boolean 'match' field.
        """
        exact = self.compute_exact()
        fin = self.compute_finite_diff()
        results = {}
        for k in exact:
            e = exact[k]
            f = fin[k]
            threshold = max(abs(rel_error * e), abs(rel_error * f))
            results[k] = dict(
                exact = e,
                difference = f,
                match = abs(e - f) <= threshold
            )
        return results


def get_functions(data) -> TupleT[Callable, Symbol, ParseEnv]:
    """
    Arguments:
        data: Either a Python function or an (url, lineno, source_code)
            tuple.

    Returns:
        (pyfn, sym, bindings) where pyfn is a Python function,
        sym is the symbol for the Myia version of the function,
        and bindings contains all symbol/Lambda mappings it needs.
    """
    if isinstance(data, tuple):
        url, line_offset, code = data
        _globals: Dict = {}
        exec(code, _globals)
        sym, genv = parse_source(url, line_offset, code)
        pyfn = _globals[sym.label]
    else:
        pyfn = data
        sym, genv = parse_function(pyfn)

    pyfn.__globals__.update({str(k): v for k, v in root_globals.items()})
    return pyfn, sym, genv


def analysis(mode: str, spec, args=None) -> Union[Callable, Dict]:
    """
    Arguments:
        mode:
            * 'eval' to test the evaluation of a function
            * 'grad' to test the gradient
            * 'grad2' to test the second order gradient
        spec:
            * either a ``(url, lineno, source_code)`` tuple,
            * or a function
        args (optional):
            List of arguments to analyze the function at,
            or None.

    Returns:
        The results of the test on the provided arguments,
        or a function that takes a list of arguments and
        returns the results.
    """
    pyfn, sym, bindings = get_functions(spec)
    method = globals()[f'analysis_{mode}']
    test = method(pyfn, sym, bindings)
    if args:
        return test(args)
    else:
        return test


def compare_calls(funcs: Dict[str, Callable],
                  args: List[Any]) -> Dict:
    """
    Compare the results of multiple functions on the same list
    of arguments. If a function raises an exception, that
    exception will be used in place of its return value.

    Arguments:
        funcs: A mapping from function names to functions to
            test.
        args: A list of arguments that will be passed to each
            function.

    Returns:
        A dictionary from function names to the results of the
        calls (or the exception(s) raised), as well as the
        boolean 'match' key, which is True iff all results are
        the same.
    """
    rval = {}
    for k, fn in funcs.items():
        try:
            r = fn(*args)
        except Exception as exc:
            r = exc
        rval[f'{k}_result'] = r
    vals = list(rval.values())
    rval['match'] = all(x == y for x, y in zip(vals[:-1], vals[1:]))
    return rval


def analysis_eval(pyfn: Callable,
                  sym: Symbol,
                  bindings: ParseEnv) -> Callable:
    """
    Return a function that takes a list of arguments ``args`` and
    compares the result of the pure function ``pyfn`` to its Myia
    implementation in ``bindings[sym]``.
    """
    func = evaluate(bindings[sym])

    def test(args):
        return compare_calls(dict(python = pyfn, myia = func), args)

    return test


def analysis_grad(pyfn: Callable,
                  sym: Symbol,
                  bindings: ParseEnv) -> Callable:
    """
    Return a function that takes a list of arguments ``args`` and
    compares:

    * The results of the pure Python function ``pyfn`` to its Myia
      implementation and to the Grad-transformed Myia function.
    * The finite-differences estimation of the derivative of
      ``pyfn`` and the Myia-computed gradient.
    """
    func = evaluate(bindings[sym])

    lbda = bindings[sym]
    albda = a_normal(lbda)
    assert isinstance(albda, Lambda)
    G = Grad(sym, albda)
    g = G.transform()
    assert G.global_env is bindings

    gfunc = evaluate(bindings[g])

    def test(args):
        myiag, bprop = gfunc(*args)
        comparison = compare_calls(dict(
            python = pyfn,
            myia = func,
            myiag = lambda *args: myiag
        ), args)
        gt = GradTester(pyfn, bprop, args, func.argnames, None)
        comparison.update(dict(
            derivatives = gt.compare()
        ))
        return comparison

    return test


# TODO: The following is not fully solid yet.

def grad2_transform(rsym, bindings):
    rlbda = bindings[rsym]
    genv = rlbda.global_env
    gen = rlbda.gen
    rrsym = genv.gen('GG')

    from .ast import Lambda, Apply, Value, Let
    from .symbols import builtins

    sym_arg = gen('ARG')
    sym_values = (gen("values"), Apply(rsym, sym_arg))
    sym_bprop = (gen("bprop"), Apply(builtins.index, sym_values[0], Value(1)))
    sym_grads = (gen("grads"), Apply(sym_bprop[0], Value(1)))
    sym_grad = (gen("grad"), Apply(builtins.index, sym_grads[0], Value(1)))
    ret = Let((sym_values, sym_bprop, sym_grads, sym_grad), sym_grad[0])
    glbda = Lambda([sym_arg], ret, gen)
    genv[rrsym] = glbda
    glbda.ref = rrsym
    glbda.global_env = rlbda.global_env

    return rrsym

    # return Lambda(
    #     ARGS,
    #     Let((gen('X'), Apply()))
    #     Apply(builtins.index, Apply())
    # )
    # glbda.global_env[]
    # return glbda


def analysis_grad2(pyfn, sym, bindings):
    lbda = bindings[sym]
    func = evaluate(lbda)

    G = Grad(sym, a_normal(lbda))
    g = G.transform()
    assert G.global_env is bindings

    gfunc = evaluate(bindings[g])

    # from .buche import buche
    # buche(g)

    g2_sym = grad2_transform(g, bindings)
    G = Grad(g2_sym, a_normal(bindings[g2_sym]))
    g2 = G.transform()
    gfunc2 = evaluate(bindings[g2])

    def gradients(*args):
        return gfunc(*args)[1](1)[1]

    def test(args):
        # return gfunc2(*args)[1](1)

        myiag2, bprop = gfunc2(*args)
        gt = GradTester(gradients, bprop, args,
                        func.argnames, ('(df/dx)',))
        return gt.compare()

    return test
