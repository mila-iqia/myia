"""Estimate gradients with finite differences."""


import itertools
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, List

import numpy
from ovld import ovld

from ..utils import dataclass_fields, smap


@smap.variant
def _zeros_like(self, x: object):
    if x is None:
        return None
    else:
        return 0


class NoTestGrad:
    """Class to signify that a value's gradient shouldn't be tested.

    I don't fully remember how this is supposed to work.
    """

    def __init__(self, value):
        """Initialize NoTestGrad."""
        self.value = value


# The variation applied on an input in either direction to estimate
# the gradient.
eps = 1e-10

# The tolerance for the difference between the estimation and the
# computed gradient.
rel_error = 1e-03


def clean_args(args):
    """Remove instances of NoTestGrad in the given arguments."""
    return tuple(a.value if isinstance(a, NoTestGrad) else a for a in args)


@ovld
def gen_paths(self, obj: NoTestGrad, path):
    """Generate all paths to a scalar through an object.

    For example, ((a, b), {'x': c}) would generate the paths
    (0, 0), (0, 1) and (1, 'x') for a, b and c.
    """
    yield from []


@ovld  # noqa: F811
def gen_paths(self, obj: (list, tuple), path):
    for i, x in enumerate(obj):
        yield from gen_paths(x, path + (i,))


@ovld  # noqa: F811
def gen_paths(self, obj: numpy.ndarray, path):
    for coord in itertools.product(*[range(d) for d in obj.shape]):
        yield path + (coord,)


@ovld  # noqa: F811
def gen_paths(self, obj: object, path):
    if is_dataclass(obj):
        for name, value in dataclass_fields(obj).items():
            yield from gen_paths(value, path + (name,))
    else:
        yield path


def resolve_path(obj, path):
    """Follow the given path on the given object."""
    for p in path:
        if is_dataclass(obj):
            obj = getattr(obj, p)
        else:
            obj = obj[p]
    return obj


@ovld
def gen_variants(self, obj: NoTestGrad, gen, path):
    """
    Generate perturbated variants of the given object.

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
    yield from []


@ovld  # noqa: F811
def gen_variants(self, obj: object, gen, path):
    if is_dataclass(obj):
        fields = list(obj.__dataclass_fields__.keys())
        for field in fields:
            x = getattr(obj, field)
            for variants, p in self(x, gen, path + (field,)):
                new_variants = []
                for variant in variants:
                    d = {
                        f2: (variant if f2 == field else getattr(obj, f2))
                        for f2 in fields
                    }
                    new_variants.append(type(obj)(**d))
                yield (new_variants, p)
    else:
        yield (gen(obj), path)


@ovld  # noqa: F811
def gen_variants(self, obj: (list, tuple), gen, path):
    for i, x in enumerate(obj):
        for variants, p in self(x, gen, path + (i,)):
            yield (
                [
                    type(obj)(
                        variant if i == j else y for j, y in enumerate(obj)
                    )
                    for variant in variants
                ],
                p,
            )


@ovld  # noqa: F811
def gen_variants(self, obj: numpy.ndarray, gen, path):
    for coord in itertools.product(*[range(d) for d in obj.shape]):
        for variants, p in self(obj[coord], gen, path + (coord,)):
            res = []
            for variant in variants:
                obj2 = obj.copy()
                obj2[coord] = variant
                res.append(obj2)
            yield (res, p)


class GradTester:
    """Test computed gradient against finite differences estimate.

    Arguments:
        fn: The function to test against.
        gfn: The function to compute the gradient.
        args: The point in the function's domain where we want
            to estimate the gradient.
        argnames: The names of the arguments.
        outnames: The names of the outputs.

    """

    def __init__(
        self,
        fn: Callable,
        gfn: Callable,
        args: List[Any],
        argnames: List[str],
        outnames: List[str] = None,
        epsilon: float = eps,
        rel_error: float = rel_error,
    ) -> None:
        """Initialize a GradTester."""
        self.epsilon = epsilon
        self.rel_error = rel_error
        self.fn = fn
        self.gfn = gfn
        self.args = args
        self.clean_args = clean_args(args)
        self.argnames = argnames
        out = fn(*self.clean_args)
        outname = fn.__name__
        if isinstance(out, tuple):
            self.outnames = list(f"{outname}_{i+1}" for i in range(len(out)))
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

    def _set_result(self, results, opath, ipath, value):
        opath = (self.outnames[opath[0]],) + opath[1:]
        ipath = (self.argnames[ipath[0]],) + ipath[1:]
        outname = ".".join(map(str, opath))
        argname = ".".join(map(str, ipath))
        results[f"d{outname}/d{argname}"] = value

    def compute_exact(self) -> Dict[str, float]:
        """Compute the exact gradient.

        Returns:
            A dictionary that maps d<outname>/d<argname> to the
            gradient computed by gfn on args.

        """
        results: Dict[str, float] = {}
        z = _zeros_like(self.out)
        for (out_sen,), opath in gen_variants(z, lambda x: [1.0], ()):
            grads = self.gfn(*self.clean_args, self.unwrap(out_sen))
            for ipath in gen_paths(grads, ()):
                if isinstance(resolve_path(self.args, ipath), NoTestGrad):
                    continue
                self._set_result(
                    results, opath, ipath, resolve_path(grads, ipath)
                )
        self.exact = results
        return results

    def wiggle(self, x):
        """Return x +- some epsilon."""
        if x is None:
            return None, None
        else:
            return x - self.epsilon, x + self.epsilon

    def compute_finite_diff(self) -> Dict[str, float]:
        """Compute the finite differences gradient.

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

            eps = self.epsilon

            @smap.variant
            def mkdiff(self, a: object, b):
                return (b - a) / (2 * eps)

            diff = mkdiff(under_res, over_res)
            for opath in gen_paths(diff, ()):
                self._set_result(
                    results, opath, ipath, resolve_path(diff, opath)
                )

        self.finite_diff = results
        return results

    def compare(self) -> Dict[str, Dict]:
        """Compare the exact gradients to the estimated ones.

        Returns:
            A dictionary that maps d<outname>/d<argname> to a dictionary
            that contains both gradients and a boolean 'match' field.

        """
        exact = self.compute_exact()
        fin = self.compute_finite_diff()
        results = {}
        rel = self.rel_error
        for k in exact:
            e = exact[k]
            f = fin[k]
            if e is None:
                match = f == 0
            elif e == f:
                match = True
            else:
                threshold = max(abs(rel * e), abs(rel * f))
                match = bool(abs(e - f) <= threshold)
            results[k] = dict(exact=e, difference=f, match=match)
        return results

    def assert_match(self):
        """Assert that the exact gradients match the estimated ones."""
        results = self.compare()
        failed = False
        argspec = [
            f"{name}={arg}" for name, arg in zip(self.argnames, self.args)
        ]
        print(f"In:  {', '.join(argspec)}")
        outspec = [
            f"{name}={arg}" for name, arg in zip(self.outnames, self.out)
        ]
        print(f"Out: {', '.join(outspec)}")
        for path, data in results.items():
            if data["match"]:
                print(f"{path} OK: == {data['exact']}")
            else:
                failed = True
                print(
                    f"{path} MISMATCH:"
                    f" {data['exact']} != {data['difference']}"
                    f" (exact / finite diff)"
                )

        if failed:
            raise Exception("Gradients do not match.")


__all__ = [
    "GradTester",
    "NoTestGrad",
    "clean_args",
    "gen_paths",
    "gen_variants",
    "resolve_path",
]
