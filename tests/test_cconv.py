
from myia.api import parse
from myia.cconv import NestingAnalyzer


def check_nest(rels, fvspecs):
    expected_deps = {}
    for rel in rels.split(','):
        rel = rel.split('->')
        if len(rel) == 1:
            g, = rel
            expected_deps[g] = set()
        else:
            g1, g2 = rel
            expected_deps[g1] = {g2}

    expected_fvs = {}
    for fvspec in fvspecs.split('; '):
        if not fvspec:
            continue
        gname, fvs = fvspec.split(':')
        expected_fvs[gname] = set(fvs.split(','))

    def check(fn):
        def test():
            gfn = parse(fn)
            analysis = NestingAnalyzer().run(gfn)
            def name(g):
                from myia.anf_ir import Constant, Graph
                if isinstance(g, Constant) and isinstance(g.value, Graph):
                    g = g.value
                gname = g.debug.name
                if gname == fn.__name__:
                    gname = 'X'
                return gname

            deps = {}
            for g, gs in analysis.deps.items():
                deps[name(g)] = {name(g2) for g2 in gs}
            assert deps == expected_deps
            for g1, g2 in analysis.parents.items():
                if g2:
                   assert analysis.nested_in(g1, g2)
                   assert not analysis.nested_in(g2, g1)

            fvs = {}
            for g, vs in analysis.fvs.items():
                fvs[name(g)] = {name(v) for v in vs}
            assert fvs == expected_fvs

        return test

    return check


# 'X' stands for the top level function, e.g. in test_nested, 'X' is
# equivalent to 'test_nested'. 'g->X' means that g is nested in
# test_nested.


@check_nest('X', '')
def test_flat(x):
    """Sanity check."""
    return x


@check_nest('X,g->X', 'g:x')
def test_nested(x):
    """g is nested in X."""
    def g():
        return x
    return g


@check_nest('X,g', '')
def test_fake_nested(x):
    """g is not really nested in X because it does not have free variables."""
    def g(x):
        return x
    return g


@check_nest('X,g->X', 'g:x,g')
def test_recurse(x):
    """Test that recursion doesn't break the algorithm."""
    def g():
        return g() + x
    return g


@check_nest('X,g', '')
def test_recurse2(x):
    """Recursion with top level, no free variables."""
    def g():
        return test_recurse2()
    return g


@check_nest('X,g->X', 'g:x')
def test_recurse3(x):
    """Recursion with top level, free variables."""
    def g():
        return test_recurse3() + x
    return g


@check_nest('X,g->X,h->g,i->h', 'g:x; h:x,y; i:x,y,z')
def test_deep_nest(x):
    """Chain of closures."""
    def g(y):
        def h(z):
            def i(w):
                return x + y + z + w
            return i
        return h
    return g(2)


@check_nest('X,g->X,h->X,i->X', 'g:h; h:i; i:x')
def test_fake_deep_nest(x):
    """i is not nested in g,h because it does not use fvs from them."""
    def g():
        def h():
            def i():
                return x
            return i
        return h
    return g()


@check_nest('X,g->X,h->X', 'h:a; g:h')
def test_calls(x):
    """g has the same nesting as h, h nested in X"""
    a = x + x
    def h():
        return a
    def g():
        return h()
    return g()


@check_nest('X,g,h', '')
def test_calls2(x):
    """g has the same nesting as h, h not nested in X"""
    def h(x):
        return x
    def g():
        return h(3)
    return g()


@check_nest('X,f->X,g->X,h->X,i->X,j->i',
            'f:a,b; g:b; h:b,c,f,g; i:a; j:a,w')
def test_fvs(x):
    """Test listing of free variables."""
    a = x + 1
    b = x + 2
    c = a + b
    d = c + 4
    def f(x):
        return a + b + x
    def g(x):
        return b
    def h(x):
        return f(c) + g(b)
    def i(w):
        def j(y):
            return w + y + a
        return j
    return h(3) + i(4)


def test_multiple_analysis():
    """Test reusing the same NestingAnalyzer."""
    def f(x):
        def g():
            return x
        return g

    def f2(x, y):
        def g2():
            return x + y
        return g2

    a = NestingAnalyzer()
    gf = parse(f)
    gf2 = parse(f2)

    a.run(gf)
    a.run(gf2)
    a.run(gf)

    parents = {g.debug.debug_name: pg and pg.debug.debug_name
               for g, pg in a.parents.items()}
    assert parents == {'f': None, 'f2': None, 'g': 'f', 'g2': 'f2'}
