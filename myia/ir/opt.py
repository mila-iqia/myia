
from ..lib import BackedUniverse, is_struct, StructuralMap
from .graph import IRGraph


class OptimizedUniverse(BackedUniverse):
    def __init__(self, parent, passes):
        super().__init__(parent)
        self.passes = passes

    def acquire(self, x):
        x = self.parent[x]
        if isinstance(x, IRGraph):
            self.cache[x] = x
            self.optimize(x)
            return x
        elif is_struct(x):
            return StructuralMap(self.acquire)(x)
        else:
            return x

    def optimize(self, graph):
        for passs in self.passes:
            passs(self, graph)
