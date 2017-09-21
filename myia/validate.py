"""
Validation and testing functionality.

* Sanity checks, e.g. ``missing_source`` or ``unbound``
* Comparing results of Python and Myia implementations.
* Estimate gradients with finite differences for comparison
  purposes.
"""


from typing import Iterable, Set, Tuple as TupleT, \
    Callable, Dict, List, Any, Union
import numpy
import itertools
from .stx import MyiaASTNode, Symbol, LambdaNode, LetNode, \
    maptup, create_lambda, is_global
from .transform import a_normal, Grad, ggen
from .parse import parse_source, parse_function
from .interpret import evaluate
from .lib import Record, structural_map
from .impl import impl_interp as M


class NoTestGrad:
    def __init__(self, value):
        self.value = value


def missing_source(node: MyiaASTNode) -> Iterable[MyiaASTNode]:
    """
    Yield all nodes that don't have a location set.
    """
    if not node.find_location():
        yield node
    for child in node.children():
        yield from missing_source(child)


def unbound(node: MyiaASTNode,
            avail: Set[Symbol] = None) -> Iterable[Symbol]:
    """
    Yield all symbols that are not bound by their enclosing
    LambdaNode (excluding globals/builtins).
    """
    if avail is None:
        avail = set()
    if isinstance(node, Symbol):
        if not is_global(node) and node not in avail:
            yield node
    elif isinstance(node, LambdaNode):
        yield from unbound(node.body, set(node.args))
    elif isinstance(node, LetNode):
        avail = set(avail)
        for s, v in node.bindings:
            yield from unbound(v, avail)
            maptup((lambda x: avail.add(x)), s)
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


def clean_args(args):
    return tuple(a.value if isinstance(a, NoTestGrad) else a for a in args)


def gen_paths(obj, path):
    if isinstance(obj, NoTestGrad):
        pass
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            yield from gen_paths(x, path + (i,))
    elif isinstance(obj, Record):
        for k, v in obj.__dict__.items():
            yield from gen_paths(v, path + (k,))
    elif isinstance(obj, numpy.ndarray):
        for coord in itertools.product(*[range(d) for d in obj.shape]):
            yield path + (coord,)
    elif isinstance(obj, (int, float)):
        yield path
    else:
        pass


def resolve_path(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def gen_variants(obj, gen, path):
    """
    For each scalar element in obj, generate a list of copies obj where that
    element has been modified by gen, and the path to that element.
    Basically:

    >>> res = gen_variants((10, 20, 30), lambda x: (x-1, x+1), ())
    >>> for x in res: print(x)
    ([(9, 20, 30), (11, 20, 30)], (0,))
    ([(10, 19, 30), (10, 21, 30)], (1,))
    ([(10, 20, 29), (10, 20, 31)], (2,))

    This is used to generate modified inputs to estimate the gradient wrt each
    element, and to generate output sensitivities to do backprop of the
    gradient of each output.
    """
    if isinstance(obj, NoTestGrad):
        pass
    elif isinstance(obj, (list, tuple)):
        T = type(obj)
        for i, x in enumerate(obj):
            for variants, p in gen_variants(x, gen, path + (i,)):
                yield ([T(variant if i == j else y for j, y in enumerate(obj))
                        for variant in variants], p)
    elif isinstance(obj, Record):
        d = obj.__dict__
        for k, v in d.items():
            for variants, p in gen_variants(v, gen, path + (k,)):
                yield ([Record(**{kk: (variant if k == kk else vv)
                                  for kk, vv in d.items()})
                        for variant in variants], p)
    elif isinstance(obj, numpy.ndarray):
        for coord in itertools.product(*[range(d) for d in obj.shape]):
            for variants, p in gen_variants(obj[coord], gen, path + (coord,)):
                res = []
                for variant in variants:
                    obj2 = obj.copy()
                    obj2[coord] = variant
                    res.append(obj2)
                yield (res, p)
    elif isinstance(obj, (int, float)):
        yield (gen(obj), path)
    else:
        pass


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
        out = fn(*clean_args(args))
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
        self.nin = len(self.argnames)
        self.nout = len(self.outnames)

    def set_result(self, results, opath, ipath, value):
        opath = (self.outnames[opath[0]],) + opath[1:]
        ipath = (self.argnames[ipath[0]],) + ipath[1:]
        outname = '.'.join(map(str, opath))
        argname = '.'.join(map(str, ipath))
        results[f'd{outname}/d{argname}'] = value

    def compute_exact(self) -> Dict[str, float]:
        """
        Compute the exact gradient.

        Returns:
            A dictionary that maps d<outname>/d<argname> to the
            gradient computed by gfn on args.
        """
        results: Dict[str, float] = {}
        z = M.zeros_like(self.out)
        for (out_sen,), opath in gen_variants(z, lambda x: [1], ()):
            grads = self.gfn(self.unwrap(out_sen))[1:]
            for ipath in gen_paths(grads, ()):
                if isinstance(resolve_path(self.args, ipath), NoTestGrad):
                    continue
                self.set_result(results, opath, ipath,
                                resolve_path(grads, ipath))
        self.exact = results
        return results

    def wiggle(self, x):
        return x - eps, x + eps

    def compute_finite_diff(self) -> Dict[str, float]:
        """
        Compute the finite differences gradient.

        Returns:
            A dictionary that maps d<outname>/d<argname> to the
            gradient computed by finite difference with fn on args.
        """
        results: Dict[str, float] = {}
        for (under, over), ipath in gen_variants(self.args, self.wiggle, ()):
            under = clean_args(under)
            over = clean_args(over)

            under_res = self.wrap(self.fn(*under))
            over_res = self.wrap(self.fn(*over))

            def mkdiff(a, b):
                return (b - a) / (2 * eps)

            diff = structural_map(mkdiff, under_res, over_res)
            for opath in gen_paths(diff, ()):
                self.set_result(results, opath, ipath,
                                resolve_path(diff, opath))

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


def get_functions(data) -> TupleT[Callable, LambdaNode]:
    """
    Arguments:
        data: Either a Python function or an (url, lineno, source_code)
            tuple.

    Returns:
        (pyfn, lbda) where pyfn is a Python function,
        and lbda is the LambdaNode compiled from it.
    """
    if isinstance(data, tuple):
        url, line_offset, code = data
        _globals: Dict = {}
        exec(code, _globals)
        lbda = parse_source(url, line_offset, code)
        pyfn = _globals[lbda.ref.label]
    else:
        pyfn = data
        lbda = parse_function(pyfn)

    return pyfn, lbda


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
    pyfn, lbda = get_functions(spec)
    method = globals()[f'analysis_{mode}']
    test = method(pyfn, lbda)
    rval = {
        'test': test,
        'lbda': lbda
    }
    if args:
        rval['result'] = test(args)
    return rval


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

    def same(a, b):
        def ass(x, y):
            assert x == y
        try:
            structural_map(ass, a, b)
        except AssertionError:
            return False
        else:
            return True

    rval['match'] = all(same(x, y) for x, y in zip(vals[:-1], vals[1:]))
    return rval


def analysis_eval(pyfn: Callable,
                  lbda) -> Callable:
    """
    Return a function that takes a list of arguments ``args`` and
    compares the result of the pure function ``pyfn`` to its Myia
    implementation in ``bindings[sym]``.
    """
    func = evaluate(lbda)

    def test(args):
        return compare_calls(dict(python = pyfn, myia = func), args)

    return test


def analysis_grad(pyfn: Callable,
                  lbda) -> Callable:
    """
    Return a function that takes a list of arguments ``args`` and
    compares:

    * The results of the pure Python function ``pyfn`` to its Myia
      implementation and to the Grad-transformed Myia function.
    * The finite-differences estimation of the derivative of
      ``pyfn`` and the Myia-computed gradient.
    """
    func = evaluate(lbda)
    albda = a_normal(lbda)
    assert isinstance(albda, LambdaNode)
    G = Grad(lbda.ref, albda)
    glbda = G.transform()

    gfunc = evaluate(glbda)

    def test(args):
        args2 = clean_args(args)
        myiag, bprop = gfunc(*args2)
        comparison = compare_calls(dict(
            python = pyfn,
            myia = func,
            myiag = lambda *args: myiag
        ), args2)
        gt = GradTester(pyfn, bprop, args, func.argnames, None)
        comparison.update(dict(
            derivatives = gt.compare()
        ))
        return comparison

    return test


# TODO: The following is not fully solid yet.

def grad2_transform(rlbda):
    rsym = rlbda.ref
    gen = rlbda.gen
    rrsym = ggen('GG')

    from .stx import LambdaNode as Lambda, ApplyNode as Apply, \
        ValueNode as Value, LetNode as Let
    from .symbols import builtins

    sym_arg = gen('ARG')
    sym_values = (gen("values"), Apply(rsym, sym_arg))
    sym_bprop = (gen("bprop"), Apply(builtins.index, sym_values[0], Value(1)))
    sym_grads = (gen("grads"), Apply(sym_bprop[0], Value(1)))
    sym_grad = (gen("grad"), Apply(builtins.index, sym_grads[0], Value(1)))
    ret = Let((sym_values, sym_bprop, sym_grads, sym_grad), sym_grad[0])
    glbda = create_lambda(rrsym, [sym_arg], ret, gen)
    return glbda


def analysis_grad2(pyfn, lbda):
    sym = lbda.ref
    func = evaluate(lbda)

    G = Grad(sym, a_normal(lbda))
    glbda = G.transform()

    gfunc = evaluate(glbda)

    gwrap = grad2_transform(glbda)
    G = Grad(gwrap.ref, a_normal(gwrap))
    g2lbda = G.transform()
    gfunc2 = evaluate(g2lbda)

    def gradients(*args):
        return gfunc(*args)[1](1)[1]

    def test(args):
        myiag2, bprop = gfunc2(*args)
        gt = GradTester(gradients, bprop, args,
                        func.argnames, ('(df/dx)',))
        return gt.compare()

    return test
