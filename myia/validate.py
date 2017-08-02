from .ast import Symbol, Lambda, Let
from .compile import a_normal
from .front import parse_source, parse_function0, get_global_env
from .grad import Grad, one
from .interpret import evaluate, global_env


def missing_source(node):
    if not node.location:
        node.annotations = node.annotations | {'missing-source'}
        print('Missing source location: {}'.format(node))
        if node.trace:
            t = node.trace[-1]
            print('  Definition at:')
            print('    ' + t.filename + ' line ' + str(t.lineno))
            print('    ' + t.line)
    for child in node.children():
        missing_source(child)


def unbound(node, avail=None):
    if avail is None:
        avail = set()
    if isinstance(node, Symbol):
        if node.namespace not in {'global', 'builtin'} \
                and node not in avail:
            node.annotations = node.annotations | {'unbound'}
    elif isinstance(node, Lambda):
        unbound(node.body, set(node.args))
    elif isinstance(node, Let):
        avail = set(avail)
        for s, v in node.bindings:
            unbound(v, avail)
            avail.add(s)
        unbound(node.body, avail)
    else:
        for child in node.children():
            unbound(child, avail)


def guard(fn, args):
    try:
        return fn(*args)
    except Exception as exc:
        return exc


def get_functions(data):
    if isinstance(data, tuple):
        url, line_offset, code = data
        _globals = {}
        exec(code, _globals)
        r, bindings = parse_source(url, line_offset, code)
        pyfn = _globals[r.label]
    else:
        pyfn = data
        r, bindings = parse_function0(pyfn)

    lbda = bindings[r]
    if not isinstance(lbda, Lambda):
        print('grad can only operate on a function.', file=sys.stderr)

    func = evaluate(r, bindings)

    pyfn.__globals__.update({str(k): v for k, v in global_env.items()})

    return pyfn, lbda, func, r, bindings


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


def grad_test(data):

    pyfn, lbda, func, r, bindings = get_functions(data)

    transformed = {}
    gbindings = {}

    G = Grad(r, a_normal(lbda))
    g = G.transform()
    transformed[r] = g
    gbindings.update(G.global_env.bindings)

    gfunc = evaluate(g, {**gbindings, **bindings})

    def test(args):
        python = guard(pyfn, args)
        myia = guard(func, args)
        myiag, bprop = gfunc(*args)
        gt = GradTester(pyfn, bprop, args, myiag, func.argnames, None)
        return dict(
            python = python,
            myia = myia,
            myiag = myiag,
            match = python == myia == myiag,
            derivatives = gt.compare()
        )

    return dict(
        func = func,
        func_py = pyfn,
        func_sym = r,
        func_bindings = bindings,
        grad = gfunc,
        grad_sym = g,
        grad_bindings = gbindings,
        test = test
    )
