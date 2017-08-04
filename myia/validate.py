from .ast import Symbol, Lambda, Let
from .compile import a_normal
from .front import parse_source, parse_function
from .grad import Grad, one
from .interpret import evaluate, root_globals


def missing_source(node):
    if not node.location:
        yield node
    for child in node.children():
        yield from missing_source(child)


def unbound(node, avail=None):
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


eps = 1e-10
rel_error = 1e-03


def finite_diff(fn, args, eps=eps):
    rval = []
    for i in range(len(args)):
        argsl = list(args)
        argsl[i] -= eps
        r1 = fn(*argsl)
        argsl[i] += 2 * eps
        r2 = fn(*argsl)
        d = (r2 - r1) / (2 * eps)
        rval.append(d)
    return rval


class GradTester:

    def __init__(self, fn, gfn, args, out, argnames, outnames=None):
        self.fn = fn
        self.gfn = gfn
        self.args = args
        self.argnames = argnames
        outname = fn.__name__
        if isinstance(out, tuple):
            self.outnames = tuple(f'{outname}_{i+1}' for i in range(len(out)))
            self.out = out
            self.wrap = lambda x: x
            self.unwrap = lambda x: x
        else:
            self.outnames = (outname,)
            self.out = (out,)
            self.wrap = lambda x: (x,)
            self.unwrap = lambda x: x[0]
        if any(not isinstance(x, (int, float)) for x in self.out):
            raise TypeError('Can only compute gradient'
                            ' for a tuple of outputs.')
        self.nin = len(self.argnames)
        self.nout = len(self.outnames)

    def compute_exact(self):
        results = {}
        for i, outname in enumerate(self.outnames):
            out_sen = tuple((1 if k == i else 0) for k in range(self.nout))
            grads = self.gfn(self.unwrap(out_sen))[1:]
            for j, argname in enumerate(self.argnames):
                results[f'd{outname}/d{argname}'] = grads[j]
        self.exact = results
        return results

    def compute_finite_diff(self):
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

    def compare(self):
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


def get_functions(data):
    if isinstance(data, tuple):
        url, line_offset, code = data
        _globals = {}
        exec(code, _globals)
        sym, genv = parse_source(url, line_offset, code)
        pyfn = _globals[sym.label]
    else:
        pyfn = data
        sym, genv = parse_function(pyfn)

    bindings = genv.bindings
    pyfn.__globals__.update({str(k): v for k, v in root_globals.items()})
    return pyfn, sym, bindings


def analysis(mode, spec, args=None):
    pyfn, sym, bindings = get_functions(spec)
    method = globals()[f'analysis_{mode}']
    test = method(pyfn, sym, bindings)
    if args:
        return test(args)
    else:
        return test


def compare_calls(funcs, args):
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


def analysis_eval(pyfn, sym, bindings):
    func = evaluate(bindings[sym])

    def test(args):
        return compare_calls(dict(python = pyfn, myia = func), args)

    return test


def analysis_grad(pyfn, sym, bindings):
    func = evaluate(bindings[sym])

    lbda = bindings[sym]
    G = Grad(sym, a_normal(lbda))
    g = G.transform()
    assert G.global_env.bindings is bindings

    gfunc = evaluate(bindings[g])

    def test(args):
        myiag, bprop = gfunc(*args)
        comparison = compare_calls(dict(
            python = pyfn,
            myia = func,
            myiag = lambda *args: myiag
        ), args)
        gt = GradTester(pyfn, bprop, args, myiag, func.argnames, None)
        comparison.update(dict(
            derivatives = gt.compare()
        ))
        return comparison

    return test


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
    assert G.global_env.bindings is bindings

    gfunc = evaluate(bindings[g])

    # from .buche import buche
    # buche(g)

    g2_sym = grad2_transform(g, bindings)
    G = Grad(g2_sym, a_normal(bindings[g2_sym]))
    g2 = G.transform()
    gfunc2 = evaluate(bindings[g2])

    def test(args):
        return gfunc2(*args)[1](1)

    return test
