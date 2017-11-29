
from ..lib import BackedUniverse, is_struct, StructuralMap
from .graph import IRGraph


class OptimizedUniverse(BackedUniverse):
    def __init__(self, parent, passes):
        super().__init__(parent)
        self.passes = passes

    def acquire(self, orig_x):
        x = self.parent[orig_x]
        if isinstance(x, IRGraph):
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
