from .ast import Symbol, Lambda, Let


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

