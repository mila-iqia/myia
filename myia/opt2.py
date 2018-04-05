
from myia.unify import Var, SVar, Unification, expandlist
from myia.utils import Named
from myia.graph_utils import toposort
from myia.anf_ir import ANFNode, Graph, Apply, Constant, Parameter, Special
from myia.anf_ir_utils import \
    succ_incoming, freevars_boundary, replace, \
    is_apply, is_constant, is_parameter, is_special


class VarNode(Special):
    def __var__(self):
        return self.special


def sexp_to_node(sexp, graph):
    if isinstance(sexp, tuple):
        return Apply([sexp_to_node(x, graph) for x in sexp], graph)
    elif isinstance(sexp, Var):
        return VarNode(sexp, graph)
    elif isinstance(sexp, ANFNode):
        return sexp
    else:
        return Constant(sexp)


def sexp_to_graph(sexp):
    g = Graph()
    g.output = sexp_to_node(sexp, g)
    return g


class GraphUnification(Unification):
    def visit(self, fn, node):
        if is_apply(node):
            return Apply(expandlist(map(fn, node.inputs)), fn(node.graph))
        elif is_parameter(node):
            g = fn(node.graph)
            if isinstance(g, Graph):
                return node
            else:
                return Parameter(g)
        elif is_constant(node):
            return Constant(fn(node.value))
        elif isinstance(node, VarNode):
            return fn(node.special)
        elif is_special(node):
            return Special(fn(node.special), fn(node.graph))
        else:
            raise self.VisitError


class PatternSubstitutionOptimization:
    def __init__(self, pattern, replacement):
        g = Var('GRAPH')
        self.pattern = sexp_to_node(pattern, g)
        self.replacement = sexp_to_node(replacement, g)
        self.unif = GraphUnification()

    def __call__(self, node):
        equiv = self.unif.unify(node, self.pattern)
        if equiv:
            return self.unif.reify(self.replacement, equiv)
        else:
            return None


class PatternOptimizerSinglePass:
    def __init__(self, patterns):
        self.patterns = patterns

    def iterate(self, graph):
        incl = freevars_boundary(graph, False)
        return toposort(graph.output, succ_incoming, incl)

    def replace(self, old, new):
        return replace(old, new)

    def __call__(self, graph):
        changes = False
        for node in self.iterate(graph):
            for pattern in self.patterns:
                new = pattern(node)
                if new and new is not node:
                    self.replace(node, new)
                    changes = True
                    continue
        return changes


class PatternOptimizerEquilibrium:
    def __init__(self, single_pass):
        self.single_pass = single_pass

    def __call__(self, *graphs):
        any_changes = False

        changes = True
        while changes:
            changes = False
            for graph in graphs:
                changes |= self.single_pass(graph)
                any_changes |= changes

        return any_changes
