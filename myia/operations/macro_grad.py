"""Implementation of the 'grad' operation."""

from collections import defaultdict
from functools import reduce

from .. import lib, operations
from ..lib import (
    ALIASID,
    About,
    Constant,
    Graph,
    MetaGraph,
    MultitypeGraph,
    MyiaTypeError,
    Named,
    NamedDebugInfo,
    Parameter,
    core,
    generate_getters,
    macro,
    setter_from_getter,
    sexp_to_node,
)
from ..xtype import Number
from . import primitives as P


@macro
async def grad(info, fn, *args):
    """Create a function for the gradient of another function.

    `grad` must be called on a function, and returns a function, so to
    get df/dx one must write `grad(f)(x, y, ...)`.

    Usage:
        grad(f)(x, y)              == df/dx
        grad(f, 'y')(x, y)         == df/dy
        grad(f, 'x', 'y')(x, y)    == (df/dx, df/dy)
        grad(f, return_value=True) == (f(x, y), df/dx)
        grad(f, dout=z)            == z * df/dx, if f(x, y) is a scalar
    """
    fn, *argtypes = await info.abstracts()
    wrt = []

    flags = {"return_value": False}

    for arg in argtypes:
        if isinstance(arg, lib.AbstractKeywordArgument):
            if arg.key in flags:
                flags[arg.key] = lib.build_value(
                    arg.argument, default=lib.ANYTHING
                )
            else:
                raise MyiaTypeError(f"grad takes no argument named {arg.key}")
        else:
            val = lib.build_value(arg, default=lib.ANYTHING)
            if isinstance(val, (int, str)):
                wrt.append(val)
            else:
                raise MyiaTypeError(f"Invalid argument to grad, {arg}")

    if not isinstance(fn, lib.AbstractFunction):
        raise MyiaTypeError(
            f"'grad' takes a function as its argument, not {fn}."
        )

    fn = fn.get_unique()
    if isinstance(fn, lib.GraphFunction):
        arg = fn.graph
        if arg.parent is not None:
            raise MyiaTypeError(
                f"'grad' does not work on closures ('grad' was given argument"
                f" '{arg}', which is a closure with parent '{arg.parent}'.)"
            )
    elif isinstance(fn, lib.MetaGraphFunction):
        arg = fn.metagraph
    elif isinstance(fn, lib.PrimitiveFunction):
        arg = fn.prim
    else:
        raise MyiaTypeError(f"'grad' cannot handle {fn}")

    return Constant(GradOperation(arg, wrt, **flags))


_cast_helper = MultitypeGraph("cast_helper")


@_cast_helper.register(Number, Number)
@core
def _scalar_cast_helper(x, model):
    t = operations.typeof(model)
    return P.scalar_cast(x, t)


@_cast_helper.register(Number, lib.AbstractArray)
@core
def _scalar_to_array_cast_helper(x, model):
    t = operations.typeof(model)
    cast = P.scalar_cast(x, t.element)
    return P.scalar_to_array(cast, t)


ROOT = Named("ROOT")


class GradOperation(MetaGraph):
    """Implements the grad(f) operation.

    This MetaGraph is returned by a call to `grad`.
    """

    def __init__(
        self,
        fn,
        wrt,
        *,
        return_value=False,
        always_return_tuple=False,
        dout_parameter=False,
        sum_aliases=True,
    ):
        """Initialize GradOperation."""
        super().__init__("grad")
        self.fn = fn
        self.wrt = wrt
        self.return_value = return_value
        self.always_return_tuple = always_return_tuple
        self.dout_parameter = dout_parameter
        self.sum_aliases = sum_aliases

    def make_signature(self, args):
        """Make the signature.

        The signature is a pair with the first element being the signature
        generated from self.fn and the second a boolean saying whether there is
        a dout argument or not.
        """
        aliases = defaultdict(list)
        if self.sum_aliases:
            for i, arg in enumerate(args):
                for elem, getter in generate_getters(arg, ROOT):
                    aid = elem.values.get(ALIASID, None)
                    if aid is not None:
                        assert aid is not lib.ANYTHING
                        aliases[aid].append((i, getter))
        aliases = tuple(sorted((k, tuple(v)) for k, v in aliases.items()))

        if (
            len(args) > 0
            and isinstance(args[-1], lib.AbstractKeywordArgument)
            and args[-1].key == "dout"
        ):
            dout = "kw"
        else:
            dout = self.dout_parameter
        if dout:
            args = args[:-1]
        if any(isinstance(arg, lib.AbstractKeywordArgument) for arg in args):
            raise MyiaTypeError(
                f"Only 'dout' is valid as a keyword argument in a"
                " grad-transformed function."
            )
        if isinstance(self.fn, (Graph, MetaGraph)):
            sig = self.fn.make_signature(args)
        else:
            sig = (len(args),)
        return sig, dout, aliases

    def generate_graph(self, sig):
        """Make the graph for the grad.

        If wrt is an integer, the wrt-th gradient will be returned directly.
        If it is a tuple of integers, then a tuple of the specified gradients
        will be returned in the same order (duplicates are allowed).

        If self.return_value is True, a tuple will always be returned and the
        first element will be the return value of the function. The other
        elements will be the gradients.
        """
        gsig, dout, aliases = sig
        if isinstance(self.fn, (Graph, MetaGraph)):
            g = self.fn.generate_graph(gsig)
            dbg = g.debug
            nargs = len(g.parameters)
            orig_parameters = g.parameters
            orig_parameter_names = g.parameter_names
        else:
            g = self.fn
            dbg = NamedDebugInfo()
            (nargs,) = gsig
            orig_parameters = [Parameter(None) for _ in range(nargs)]
            orig_parameter_names = None

        def _getindex(wrt):
            if wrt == "*":
                raise MyiaTypeError(f"'*' in grad must be the only parameter")
            elif isinstance(wrt, str):
                try:
                    return orig_parameter_names.index(wrt)
                except ValueError:
                    raise MyiaTypeError(f"{g} has no argument named '{wrt}'")
            elif 0 <= wrt < nargs:
                return wrt
            else:
                raise MyiaTypeError(
                    f"Cannot get gradient with respect to argument {wrt}"
                    f" for {g} because it is out of range."
                )

        if self.wrt == ["*"]:
            wrt = list(range(nargs))
        else:
            wrt = list(map(_getindex, self.wrt))
            if len(wrt) == 1:
                wrt = wrt[0]
            elif wrt == []:
                wrt = 0

        with About(dbg, "grad"):
            df = Graph()
            df.set_flags("core", "reference")

        jf = df.apply(P.J, g)

        params = []
        for orig_p in orig_parameters:
            with About(orig_p.debug, "grad"):
                params.append(df.add_parameter())

        jparams = [df.apply(P.J, p) for p in params]
        app = df.apply(jf, *jparams)
        out = df.apply(P.Jinv, df.apply(P.tuple_getitem, app, 0))
        bprop = df.apply(P.tuple_getitem, app, 1)

        if dout:
            bprop_arg = df.add_parameter()
            bprop_arg.debug.name = "dout"
            if dout == "kw":
                bprop_arg = df.apply(P.extract_kwarg, "dout", bprop_arg)
        else:
            bprop_arg = df.apply(_cast_helper, 1, out)

        if isinstance(wrt, int):
            direct_return = not self.always_return_tuple
            wrt = [wrt]
        else:
            direct_return = False

        bapp = df.apply(bprop, bprop_arg)
        all_results = [
            df.apply(P.tuple_getitem, bapp, idx + 1) for idx in range(nargs)
        ]

        adjusted = {i: all_results[i] for i in range(nargs)}
        for aid, equivs in aliases:
            contribs = []
            for i, entry in equivs:
                node = sexp_to_node(entry, df, sub={ROOT: all_results[i]})
                contribs.append(node)
            combined = reduce(
                lambda x, y: df.apply(operations.gadd, x, y), contribs
            )

            for i, entry in equivs:
                setter = setter_from_getter(entry, combined)
                node = sexp_to_node(setter, df, sub={ROOT: adjusted[i]})
                adjusted[i] = node

        elems = [out] if self.return_value else []
        elems += [adjusted[idx] for idx in wrt]

        if len(elems) == 1 and direct_return:
            df.output = elems[0]
        else:
            df.output = df.apply(P.make_tuple, *elems)

        return df


__operation_defaults__ = {
    "name": "grad",
    "registered_name": "grad",
    "mapping": grad,
    "python_implementation": None,
}
