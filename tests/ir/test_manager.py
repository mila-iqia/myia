
import pytest

from collections import Counter

from myia.api import parse
from myia.debug.label import short_labeler
from myia.ir import is_constant, GraphManager, GraphCloner
from myia.prim import Primitive


swap1 = Primitive('swap1')
swap2 = Primitive('swap2')
swap3 = Primitive('swap3')
swap4 = Primitive('swap4')
swap5 = Primitive('swap5')
swap = swap1


swaps = [swap1, swap2, swap3, swap4, swap5]


class NestingSpecs:

    def __init__(self, stage, specs):
        self.expected = self._parse_specs(specs)
        self.stage = stage

    def _parse_specs(self, specs):
        if specs is None:
            return None
        expected = {}
        for spec in specs.split(';'):
            spec = spec.strip()
            if not spec:
                continue
            if '->' in spec:
                key, value = spec.split('->')
                if value:
                    expected[key] = value
            elif ':' in spec:
                key, values = spec.split(':')
                values = set(values.split(',')) - {''}
                if values:
                    expected[key] = values
            else:
                expected[spec] = True
        return expected

    def name(self, node):
        if isinstance(node, bool):
            return node
        name = short_labeler.name(node)
        if name in self.stage.subs:
            return self.stage.subs[name]
        else:
            return name

    def check(self, results):
        if self.expected is None:
            return
        clean_results = {}
        for k, v in results.items():
            if k is None:
                continue
            k = self.name(k)
            if isinstance(v, (Counter, set, dict)):
                v = {self.name(x) for x in v}
                v = {x for x in v if x}
            elif v:
                v = self.name(v)
            if v:
                clean_results[k] = v
        assert clean_results == self.expected


class Stage:
    def __init__(self, **specs):
        def _lengthen(k):
            k = k.replace('fv', 'free_variable')
            k = k.replace('deps', 'dependencies')
            return k
        self.specs = {_lengthen(k): NestingSpecs(self, v)
                      for k, v in specs.items()}
        self.subs = None

    def check(self, manager):
        for key, specs in self.specs.items():
            print(f'Verifying field: {key}')
            specs.check(getattr(manager, key))
            print(f'OK for: {key}')


def clone(g):
    return GraphCloner(g, total=True)[g]


def _check_uses(manager):
    for node in manager.all_nodes:
        for i, inp in enumerate(node.inputs):
            assert inp in manager.all_nodes
            assert (node, i) in manager.uses[inp]
        for node2, key in manager.uses[node]:
            assert node2 in manager.all_nodes
            assert node2.inputs[key] is node


def check_manager(*stages, **specs):
    if specs and stages:
        raise Exception('Bad call to check_manager')

    if specs:
        stages = [Stage(**specs)]

    def check(fn):
        subs = {fn.__name__: 'X'}
        for stage in stages:
            stage.subs = subs

        def test():
            def _replace(uses, value):
                for node, key in uses:
                    mng.push_set_edge(node, key, value)

            gfn = clone(parse(fn))

            todo = [[] for stage in stages]

            mng = GraphManager(gfn)

            for node in mng.all_nodes:
                if is_constant(node) and node.value in swaps:
                    j = swaps.index(node.value)
                    for swap_node, i in mng.uses[node]:
                        assert i == 0
                        uses = set(mng.uses[swap_node])
                        _, first, second = swap_node.inputs
                        todo[j + 1].append((uses, second))
                        mng.push_replace(swap_node, first)

            mng.commit()
            mng.reset()

            for stage, operations in zip(stages, todo):
                for uses, new_node in operations:
                    _replace(uses, new_node)
                mng.commit()
                _check_uses(mng)
                stage.check(mng)

        return test

    return check


################################
# Tests not involving mutation #
################################


@check_manager(nodes='X:x', parents='', fvs_direct='')
def test_flat(x):
    """Sanity check."""
    return x


@check_manager(nodes='X:x', parents='g->X', fvs_direct='g:x')
def test_nested(x):
    """g is nested in X."""
    def g():
        return x
    return g


@check_manager(parents='')
def test_fake_nested(x):
    """g is not really nested in X because it does not have free variables."""
    def g(x):
        return x
    return g


@check_manager(parents='g->X',
               fvs_direct='g:x',
               fvs_total='g:x,g',
               graphs_used='X:g; g:g',
               graphs_reachable='X:g; g:g',
               recursive='g')
def test_recurse(x):
    """Test that recursion doesn't break the algorithm."""
    def g():
        return g() + x
    return g


@check_manager(parents='')
def test_recurse2(x):
    """Recursion with top level, no free variables."""
    def g():
        return test_recurse2()
    return g


@check_manager(parents='g->X', fvs_direct='g:x')
def test_recurse3(x):
    """Recursion with top level, free variables."""
    def g():
        return test_recurse3() + x
    return g


@check_manager(parents='g->X; h->g; i->h',
               fvs_direct='i:x,y,z',
               fvs_total='g:x; h:x,y; i:x,y,z')
def test_deep_nest(x):
    """Chain of closures."""
    def g(y):
        def h(z):
            def i(w):
                return x + y + z + w
            return i
        return h
    return g(2)


@check_manager(parents='g->X; h->X; i->X',
               fvs_direct='i:x',
               fvs_total='g:h; h:i; i:x')
def test_fake_deep_nest(x):
    """i is not nested in g,h because it does not use fvs from them."""
    def g():
        def h():
            def i():
                return x
            return i
        return h
    return g()


@check_manager(parents='g->X; h->X',
               children='X:g,h',
               scopes='X:X,g,h; g:g; h:h',
               fvs_direct='h:a',
               fvs_total='h:a; g:h')
def test_calls(x):
    """g has the same nesting as h, h nested in X"""
    a = x + x

    def h():
        return a

    def g():
        return h()
    return g()


@check_manager(parents='')
def test_calls2(x):
    """g has the same nesting as h, h not nested in X"""

    def h(x):
        return x

    def g():
        return h(3)
    return g()


@check_manager(parents='f->X; g->X; h->X; i->X; j->i',
               children='X:f,g,h,i; i:j',
               scopes='X:X,f,g,h,i,j; f:f; g:g; h:h; i:i,j; j:j',
               graphs_used='X:h,i; h:f,g; i:j',
               graph_users='f:h; g:h; h:X; i:X; j:i',
               fvs_direct='f:a,b; g:b; h:b,c; j:a,w',
               fvs_total='f:a,b; g:b; h:b,c,f,g; i:a; j:a,w')
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


@check_manager(parents='f->X; g->f; h->g',
               fvs_direct='f:x; h:y,z',
               fvs_total='f:x; g:y; h:y,z')
def test_deep2(x):
    def f(y):
        def g(z):
            def h():
                return y + z
            return h
        return g(x)
    return f(x + 1)


@check_manager(parents='g->f; h->g',
               nodes='X:_x; f:x; g:y',
               fvs_direct='h:y; g:x',
               fvs_total='h:y; g:x')
def test_nested_double_reference(_x):
    def f(x):
        def g():
            y = x - 1

            def h():
                return f(y)
            return h()
        return g()
    return f(_x)


@check_manager(parents='f->X; g->f; h->f',
               fvs_direct='f:a; g:a,b; h:b',
               fvs_total='f:a; g:a,b; h:b')
def test_previously_mishandled(x):
    """This covers a case that the old NestingAnalyzer was mishandling."""
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


@check_manager(parents='f1->X; f2->f1; f3->f2; f4->f3; f5->f4',
               fvs_direct='f1:a; f2:a,b; f3:a,b,c; f4:a,b,c,d; f5:d,e',
               fvs_total='f1:a; f2:a,b; f3:a,b,c; f4:a,b,c,d; f5:d,e')
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


@check_manager(nodes='X:x,y')
def test_unused_parameter(x, y):
    return x * x


@check_manager(graphs_used='X:j; g:f; h:g; j:h,i',
               graphs_reachable='X:f,g,h,i,j; g:f; h:f,g; j:f,g,h,i',
               recursive='')
def test_reachable(x, y):
    def f(x):
        return x * x

    def g(x):
        return f(x) + f(y)

    def h():
        return g(13)

    def i():
        return 6

    def j(x, y):
        return i() + h(x)

    return j(x, y)


############################
# Tests involving mutation #
############################


# These tests verify that parents, free variables, etc. are properly
# tracked when mutating the graph through the manager.

# Each test_mut contains one or more calls to swap[1,2,3,4,5].
# All calls to swap/swap1 are done first, then all calls to swap2, etc.
# We check that the listed properties are correct for each stage.

# For example:

# Expression                  => Stage 1 => Stage 2 => Stage 3
# `swap(a, b) + swap(c, d)`   => a + c   => b + d
# `swap1(a, b) + swap2(c, d)` => a + c   => b + c   => b + d
# `swap2(a, b) + swap1(c, d)` => a + c   => a + d   => b + d


@check_manager(
    Stage(parents='f->X', fvs_direct='f:x'),
    Stage(parents='',     fvs_direct=''),
)
def test_mut_nested_to_global(x, y):
    def f(z):
        return swap(x, 1) * z
    return f(3)


@check_manager(
    Stage(parents='',     fvs_direct=''),
    Stage(parents='f->X', fvs_direct='f:x'),
)
def test_mut_global_to_nested(x, y):
    def f(z):
        return swap(1, x) * z
    return f(3)


@check_manager(
    Stage(parents='f->X',
          children='X:f',
          scopes='X:X,f; f:f',
          graph_deps_total='f:X',
          fvs_direct='f:x',
          fvs_total='f:x'),
    Stage(parents='f->X',
          children='X:f',
          scopes='X:X,f; f:f',
          graph_deps_total='f:X',
          fvs_direct='f:x',
          fvs_total='f:x'),
)
def test_mut_multiple_uses(x, y):
    """Test removing one occurrence of a fv when a second remains."""
    def f():
        return swap(x, 1) * x
    return f()


@check_manager(
    Stage(parents='g->X',
          children='X:g',
          scopes='X:X,g; f:f; g:g; h:h',
          graphs_used='X:f,g,h; h:f',
          graph_users='f:X,h; g:X; h:X',
          graph_deps_direct='g:X',
          graph_deps_total='g:X',
          fvs_direct='g:x',
          fvs_total='g:x'),

    Stage(parents='g->X; h->X',
          children='X:g,h',
          scopes='X:X,g,h; f:f; g:g; h:h',
          graphs_used='X:f,g,h; h:g',
          graph_users='f:X; g:X,h; h:X',
          graph_deps_direct='g:X',
          graph_deps_total='g:X; h:X',
          fvs_direct='g:x',
          fvs_total='g:x; h:g'),

    Stage(parents='',
          children='',
          scopes='X:X; f:f; g:g; h:h',
          graphs_used='X:f,g,h; h:g',
          graph_users='f:X; g:X,h; h:X',
          graph_deps_direct='',
          graph_deps_total='',
          fvs_direct='',
          fvs_total=''),
)
def test_mut_closure(x, y):
    def f(q):
        return q * q

    def g(z):
        return swap2(x, 1) * z

    def h(w):
        return swap1(f, g)(w)
    return f(1) + g(2) + h(3)


@check_manager(
    Stage(scopes='X:X; c:c; f1:f1; f2:f2'),
    Stage(scopes='X:X; c:c; g:g')
)
def test_remove_unused_graphs(x, y):
    def c(p):
        return p

    def f1(q):
        return c(q)

    def f2(r):
        return f1(r * r)

    def g(z):
        return c(z)

    return swap(f2, g)(3)


@check_manager(
    Stage(parents='f->X', fvs_direct='f:x',   fvs_total='f:x'),
    Stage(parents='f->X', fvs_direct='f:x,y', fvs_total='f:x,y'),
    Stage(parents='f->X', fvs_direct='f:y',   fvs_total='f:y'),
)
def test_mut_update_total(x, y):
    def f():
        return swap1(1, y) + swap2(x, 2)
    return f()


@check_manager(
    Stage(parents='f->X; g->f; h->g',
          fvs_direct='h:a,b,x',
          fvs_total='f:x; g:a,x; h:a,b,x'),

    Stage(parents='f->X; g->f; h->g',
          fvs_direct='h:a,b,x,y',
          fvs_total='f:x,y; g:a,x,y; h:a,b,x,y'),

    Stage(parents='f->X; g->f; h->g',
          fvs_direct='h:a,b,y',
          fvs_total='f:y; g:a,y; h:a,b,y'),
)
def test_mut_update_total_nest(x, y):
    def f(a):
        def g(b):
            def h():
                return a + b + swap1(1, y) + swap2(x, 2)
            return h
        return g
    return f(123)


@check_manager(
    Stage(parents='f->X; g->f; h->g',
          children='X:f; f:g; g:h',
          scopes='X:X,f,g,h; f:f,g,h; g:g,h; h:h',
          graph_deps_direct='h:X,f,g; f:X',
          graph_deps_total='h:X,f,g; g:X,f; f:X',
          fvs_direct='f:x; h:a,b,x',
          fvs_total='f:x; g:a,x; h:a,b,x'),

    Stage(parents='f->X; g->f; h->g',
          children='X:f; f:g; g:h',
          scopes='X:X,f,g,h; f:f,g,h; g:g,h; h:h',
          graph_deps_direct='h:X,f,g; f:X',
          graph_deps_total='h:X,f,g; g:X,f; f:X',
          fvs_direct='f:x; h:a,b,x,y',
          fvs_total='f:x,y; g:a,x,y; h:a,b,x,y'),

    Stage(parents='f->X; g->f; h->g',
          children='X:f; f:g; g:h',
          scopes='X:X,f,g,h; f:f,g,h; g:g,h; h:h',
          graph_deps_direct='h:X,f,g; f:X',
          graph_deps_total='h:X,f,g; g:X,f; f:X',
          fvs_direct='f:x; h:a,b,y',
          fvs_total='f:x,y; g:a,y; h:a,b,y'),
)
def test_mut_multiple_uses_deep(x, y):
    def f(a):
        def g(b):
            def h():
                return a + b + swap1(1, y) + swap2(x, 2)
            return h
        return g(x)
    return f(123)


@check_manager(
    Stage(parents='f->X; g->X', fvs_direct='f:x,y', fvs_total='f:x,y; g:f'),
    Stage(parents='f->X; g->X', fvs_direct='f:x,y', fvs_total='f:x,y; g:f'),
)
def test_mut_multiple_uses_closure(x, y):
    def f(a):
        return x * y

    def g():
        return f(1) + swap(f(2), 3)

    return g()


@check_manager(
    Stage(parents='f->X', fvs_direct='f:a', fvs_total='f:a'),
    Stage(parents='f->X', fvs_direct='f:b', fvs_total='f:b'),
)
def test_mut_uses(x, y):
    a = x * x
    b = y * y

    def f():
        return swap(a, b)
    return f()


@check_manager(
    Stage(parents=''),
    Stage(parents=''),
)
def test_mut_use_global(x, y):
    def f():
        return 1

    def g():
        return 2
    return swap(f, g)


#################
# Miscellaneous #
#################


def test_cannot_replace_return():
    @parse
    def f(x):
        return x * x

    mng = GraphManager(f)

    with pytest.raises(Exception):
        mng.push_replace(f.return_, f.parameters[0])


def test_manager_exclusivity():
    @parse
    def f(x):
        return x * x

    mng = GraphManager(f)
    assert f._manager is mng

    with pytest.raises(Exception):
        GraphManager(f)


def test_weak_manager():

    @parse
    def f(x, y):
        return x * y

    mng1 = GraphManager(f, manage=False)
    assert f._manager is None
    with pytest.raises(Exception):
        mng1.commit()
    mng2 = GraphManager(f, manage=True)
    assert f._manager is mng2
    GraphManager(f, manage=False)
    assert f._manager is mng2


def test_drop_root():

    @parse
    def f(x, y):
        return x * y

    mng = GraphManager(f)
    assert f in mng.nodes
    mng._maybe_drop_graphs({f})
    assert f in mng.nodes


def test_graph_properties():

    @parse
    def test(x):
        def f(y):
            def g(z):
                def h():
                    return y + z
                return h
            return g(x)
        return f(x + 1)

    with pytest.raises(Exception):
        test.manager

    mng = GraphManager(test)

    for g in mng.graphs:
        assert g.manager is mng
        assert g.nodes is mng.nodes[g]
        assert g.constants is mng.constants[g]
        assert g.free_variables_direct is mng.free_variables_direct[g]
        assert g.free_variables_total is mng.free_variables_total[g]
        assert g.graphs_used is mng.graphs_used[g]
        assert g.graph_users is mng.graph_users[g]
        assert g.graphs_reachable is mng.graphs_reachable[g]
        assert g.graph_dependencies_direct is mng.graph_dependencies_direct[g]
        assert g.graph_dependencies_total is mng.graph_dependencies_total[g]
        assert g.parent is mng.parents[g]
        assert g.children is mng.children[g]
        assert g.scope is mng.scopes[g]
        assert g.recursive is mng.recursive[g]
