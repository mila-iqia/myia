"""Estimate gradients with finite differences."""


from typing import Iterable, Set, Tuple as TupleT, \
    Callable, Dict, List, Any, Union
import numpy
import itertools

from myia.utils import smap
from myia.py_implementations import zeros_like


class NoTestGrad:
    def __init__(self, value):
        self.value = value


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
    # elif isinstance(obj, Record):
    #     for k, v in obj.__dict__.items():
    #         yield from gen_paths(v, path + (k,))
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
    # elif isinstance(obj, Record):
    #     for k, v in obj:
    #         for variants, p in gen_variants(v, gen, path + (k,)):
    #             yield ([obj.__variant__(k, variant)
    #                     for variant in variants], p)
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
        z = zeros_like(self.out)
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

            diff = smap(mkdiff, under_res, over_res)
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
                match = bool(abs(e - f) <= threshold)
            )
        return results
