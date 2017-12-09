
from ..lib import BackedUniverse, is_struct, StructuralMap, Primitive
from .graph import IRGraph, IRNode
from ..symbols import builtins
from ..stx import GenSym, is_builtin


ogen = GenSym('::opt')


class OptimizedUniverse(BackedUniverse):
    def __init__(self, parent, passes, duplicate=False):
        super().__init__(parent)
        self.passes = passes
        self.duplicate = duplicate

    def acquire(self, orig_x):
        x = self.parent[orig_x]
        if isinstance(x, IRGraph):
            if self.duplicate:
                g, _, _ = x.dup(no_mangle=True)
                g.lbda = x.lbda
                x = g
            self.cache[orig_x] = x
            self.optimize(x)
            return x
        elif is_struct(x):
            return StructuralMap(self.acquire)(x)
        else:
            return x

    def optimize(self, graph):
        for passs in self.passes:
            passs(self, graph)


class ClosureUnconversionPass:

    def __call__(self, universe, graph):
        cpru = universe.universes['const_prop']
        pool = {graph}
        while pool:
            g = pool.pop()
            for node in g.iternodes():
                if node.fn and node.fn.value is builtins.partial:
                    fnn = node.inputs[0]
                    fn = fnn.value
                    if isinstance(fn, IRGraph):
                        g2 = fn
                        g.replace(node, node.inputs[0])
                        for i, g2i in zip(node.inputs[1:], g2.inputs):
                            g2.replace(g2i, i)
                        g2.inputs = g2.inputs[len(node.inputs) - 1:]
                        pool.add(g2)
                    elif is_builtin(fn):
                        sym = fn
                        fn = cpru[sym]
                        if isinstance(fn, Primitive):
                            g2 = IRGraph(None, ogen(sym, '~'), g.gen)
                            clins = node.inputs[1:]
                            argins = [IRNode(g2, ogen(argn))
                                      for argn in fn.argnames[len(clins):]]
                            g2.inputs = clins + argins
                            o = IRNode(g2, ogen('/out'))
                            g2.output = o
                            o.set_app(fnn, g2.inputs)
                            g2n = IRNode(None, g2.tag)
                            g2n.value = g2
                            g.replace(node, g2n)
                elif isinstance(node.value, IRGraph):
                    pool.add(node.value)
                elif node.fn and isinstance(node.fn.value, IRGraph):
                    pool.add(node.fn.value)
