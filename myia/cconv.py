
from collections import defaultdict
from .anf_ir import Apply, Constant, Parameter, Graph
from .anf_ir_utils import dfs


def is_graph(x):
    return isinstance(x, Constant) and isinstance(x.value, Graph)


class NestingAnalyzer:

    class ParentProxy:
        def __init__(self, graph):
            self.graph = graph

    def __init__(self):
        self.prox = {}
        self.graphs = set()
        self.nodes = set()
        self.seeked = set()
        self.deps = defaultdict(set)
        self.parents = {}
        self.fvs = defaultdict(set)

    def run(self, root):
        self.nodes = set(dfs(Constant(root), True))
        self.graphs = {node.value for node in self.nodes if is_graph(node)}
        self.compute_deps()
        self.deprox()
        self.simplify()
        self.parents = {g: None if len(gs) == 0 else list(gs)[0]
                        for g, gs in self.deps.items()}
        self.free_variables()
        return self

    def compute_deps(self):
        for g in self.graphs:
            prox = self.ParentProxy(g)
            self.prox[g] = prox
        for node in self.nodes:
            orig = node.graph
            for inp in node.inputs:
                if is_graph(inp):
                    self.deps[orig].add(self.prox[inp.value])
                elif inp.graph and inp.graph is not orig:
                    self.deps[orig].add(inp.graph)

    def seek_parents(self, g, path=None):
        if path is None:
            path = set()
        if g in path:
            return set()
        deps = self.deps[g]
        parents = set()
        for dep in deps:
            if isinstance(dep, self.ParentProxy):
                parents |= self.seek_parents(dep.graph, path|{g})
            else:
                parents.add(dep)
        return parents - {g}

    def deprox(self):
        newdeps = {}
        for g in self.graphs:
            deps = {d for d in self.deps[g]
                    if not isinstance(d, self.ParentProxy)}
            print(g.debug.name, {p.debug.name for p in self.seek_parents(g)})
            newdeps[g] = self.seek_parents(g)
        self.deps = newdeps

    def simplify(self):
        to_cull = {list(deps)[0]
                   for g, deps in self.deps.items()
                   if len(deps) == 1}

        done = True
        for g, deps in self.deps.items():
            if len(deps) > 1:
                deps.difference_update(to_cull)
                done = False

        if not done:
            self.simplify()

    def free_variables(self):
        for node in self.nodes:
            for inp in node.inputs:
                if is_graph(inp):
                    owner = self.parents[inp.value]
                else:
                    owner = inp.graph
                if owner is None:
                    continue
                g = node.graph
                while g is not owner:
                    self.fvs[g].add(inp)
                    g = self.parents[g]
