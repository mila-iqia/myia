from myia.api import parse
from myia.debug.label import short_labeler
from myia.cconv import NestingAnalyzer
from myia.ir.anf import Constant, Graph


def _parse_fvspec(fvspec):
    expected_fvs = {}
    for spec in fvspec.split('; '):
        if not spec:
            continue
        gname, fvs = spec.split(':')
        expected_fvs[gname] = set(fvs.split(','))
    return expected_fvs


def check_nest(rels, fv_direct, fv_total):
    expected_deps = {}
    for rel in rels.split(','):
        rel = rel.split('->')
        if len(rel) == 1:
            g, = rel
            expected_deps[g] = set()
        else:
            g1, g2 = rel
            expected_deps[g1] = {g2}

    expected_fvs_direct = _parse_fvspec(fv_direct)
    expected_fvs_total = _parse_fvspec(fv_total)

    def check(fn):
        def test():
            gfn = parse(fn)

            def name(g):
                if isinstance(g, Constant) and isinstance(g.value, Graph):
                    g = g.value
                gname = short_labeler.name(g)
                if gname == fn.__name__:
                    gname = 'X'
                return gname

            analysis = NestingAnalyzer(gfn)
            for g1, g2 in analysis.parents().items():
                if g2:
                    assert analysis.nested_in(g1, g2)
                    assert not analysis.nested_in(g2, g1)

            for g1, children in analysis.children().items():
                for child in children:
                    assert analysis.nested_in(child, g1)
                    assert not analysis.nested_in(g1, child)

            nonlocal expected_deps
            expected_deps = {x: y for x, y in expected_deps.items() if y}
            parents = {}
            for g, parent in analysis.parents().items():
                if parent:
                    parents[name(g)] = {name(parent)}
            assert parents == expected_deps

            fvs = {}
            for g, vs in analysis.free_variables_total().items():
                if vs:
                    fvs[name(g)] = {name(v) for v in vs}
            assert fvs == expected_fvs_total

            fvs = {}
            for g, vs in analysis.free_variables_direct().items():
                if vs:
                    fvs[name(g)] = {name(v) for v in vs}
            assert fvs == expected_fvs_direct

        return test

    return check


# 'X' stands for the top level function, e.g. in test_nested, 'X' is
# equivalent to 'test_nested'. 'g->X' means that g is nested in
# test_nested.


@check_nest('X', '', '')
def test_flat(x):
    """Sanity check."""
    return x


@check_nest('X,g->X', 'g:x', 'g:x')
def test_nested(x):
    """g is nested in X."""
    def g():
        return x
    return g


@check_nest('X,g', '', '')
def test_fake_nested(x):
    """g is not really nested in X because it does not have free variables."""
    def g(x):
        return x
    return g


@check_nest('X,g->X', 'g:x', 'g:x,g')
def test_recurse(x):
    """Test that recursion doesn't break the algorithm."""
    def g():
        return g() + x
    return g


@check_nest('X,g', '', '')
def test_recurse2(x):
    """Recursion with top level, no free variables."""
    def g():
        return test_recurse2()
    return g


@check_nest('X,g->X', 'g:x', 'g:x')
def test_recurse3(x):
    """Recursion with top level, free variables."""
    def g():
        return test_recurse3() + x
    return g


@check_nest('X,g->X,h->g,i->h', 'i:x,y,z', 'g:x; h:x,y; i:x,y,z')
def test_deep_nest(x):
    """Chain of closures."""
    def g(y):
        def h(z):
            def i(w):
                return x + y + z + w
            return i
        return h
    return g(2)


@check_nest('X,g->X,h->X,i->X', 'i:x', 'g:h; h:i; i:x')
def test_fake_deep_nest(x):
    """i is not nested in g,h because it does not use fvs from them."""
    def g():
        def h():
            def i():
                return x
            return i
        return h
    return g()


@check_nest('X,g->X,h->X', 'h:a', 'h:a; g:h')
def test_calls(x):
    """g has the same nesting as h, h nested in X"""
    a = x + x

    def h():
        return a

    def g():
        return h()
    return g()


@check_nest('X,g,h', '', '')
def test_calls2(x):
    """g has the same nesting as h, h not nested in X"""

    def h(x):
        return x

    def g():
        return h(3)
    return g()


@check_nest('X,f->X,g->X,h->X,i->X,j->i',
            'f:a,b; g:b; h:b,c; j:a,w',
            'f:a,b; g:b; h:b,c,f,g; i:a; j:a,w')
def test_fvs(x):
    """Test listing of free variables."""
    a = x + 1
    b = x + 2
    c = a + b

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


@check_nest('X,f->X,g->f,h->g', 'f:x; h:y,z', 'f:x; g:y; h:y,z')
def test_deep2(x):
    def f(y):
        def g(z):
            def h():
                return y + z
            return h
        return g(x)
    return f(x + 1)


@check_nest('X,g->f,h->g', 'h:y; g:x', 'h:y; g:x')
def test_nested_double_reference(_x):
    def f(x):
        def g():
            y = x - 1

            def h():
                return f(y)
            return h()
        return g()
    return f(_x)


@check_nest('X,f->X,g->f,h->f', 'f:a; g:a,b; h:b', 'f:a; g:a,b; h:b')
def test_previously_mishandled(x):
    # This covers a case that the old NestingAnalyzer was mishandling
    a = x * x

    def f():
        b = a * a

        def g():
            c = a * b
            return c

        def h():
            d = b * b
            return d
        return g() + h() + b

    return f()


@check_nest('X,f1->X,f2->f1,f3->f2,f4->f3,f5->f4',
            'f1:a; f2:a,b; f3:a,b,c; f4:a,b,c,d; f5:d,e',
            'f1:a; f2:a,b; f3:a,b,c; f4:a,b,c,d; f5:d,e')
def test_deepest(x):
    a = x * x

    def f1():
        b = a * a

        def f2():
            c = a * b

            def f3():
                d = a * b * c

                def f4():
                    e = a * b * c * d

                    def f5():
                        f = d * e
                        return f

                    return f5()
                return f4()
            return f3()
        return f2()
    return f1()
