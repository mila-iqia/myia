from .ast import Symbol, Lambda, Let
from .compile import a_normal
from .front import parse_source
from .grad import Grad
from .interpret import evaluate


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


def grad_test(url, line_offset, code):
    _globals = {}
    exec(code, _globals)
    r, bindings = parse_source(url, line_offset, code)
    pyfn = _globals[r.label]
    lbda = bindings[r]
    if not isinstance(lbda, Lambda):
        print('grad can only operate on a function.', file=sys.stderr)
    G = Grad(r, a_normal(lbda))
    g = G.transform()
    gbindings = G.global_env.bindings

    func = evaluate(r, bindings)
    gfunc = evaluate(g, gbindings)

    eps = 1e-10

    def test(args):
        python = pyfn(*args)
        myia = func(*args)
        myiag, bprop = gfunc(*args)
        grads = bprop(1)[1:]
        derivatives = {}
        for i, arg in enumerate(func.argnames):
            argsl = list(args)
            argsl[i] -= eps
            r1 = pyfn(*argsl)
            argsl[i] += 2 * eps
            r2 = pyfn(*argsl)
            d = (r2 - r1) / (2 * eps)
            g = grads[i]
            gg = 1e-10 if g == 0 else g
            derivatives[arg] = dict(
                difference = d,
                computed = g,
                match = abs(d / gg - 1) < 1e-04
            )

        return dict(
            python = python,
            myia = myia,
            myiag = myiag,
            match = python == myia == myiag,
            derivatives = derivatives
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
