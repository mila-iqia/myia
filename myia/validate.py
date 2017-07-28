from .ast import Symbol, Lambda, Let
from .compile import a_normal
from .front import parse_source, parse_function0
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


def grad_test(data):

    pyfn, lbda, func, r, bindings = get_functions(data)

    transformed = {}
    gbindings = {}
    for k, v in bindings.items():
        G = Grad(k, a_normal(v))
        g = G.transform()
        transformed[k] = g
        gbindings.update(G.global_env.bindings)

    g = transformed[r]

    gfunc = evaluate(g, {**gbindings, **bindings})

    def test(args):
        python = guard(pyfn, args)
        assert isinstance(python, (int, float))
        myia = guard(func, args)
        myiag, bprop = gfunc(*args)
        grads = bprop(one(myiag))[1:]
        diffs = finite_diff(pyfn, args)
        derivatives = {}
        for arg, g, d in zip(func.argnames, grads, diffs):
            threshold = max(abs(rel_error * d), abs(rel_error * g))
            derivatives[arg] = dict(
                difference = d,
                computed = g,
                match = abs(d - g) <= threshold
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
