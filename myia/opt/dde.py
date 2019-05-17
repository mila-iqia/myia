"""Dead data elimination."""


from collections import defaultdict

from ..prim import ops as P
from ..utils import newenv, Partializable
from ..abstract import AbstractFunction, GraphFunction, PartialApplication, \
    DEAD, PrimitiveFunction, TypedPrimitive
from ..graph_utils import dfs
from ..ir import Constant, succ_incoming, Graph


def _flatten_call(fn):

    if isinstance(fn, PartialApplication):
        return (*_flatten_call(fn.fn), *fn.args)

    elif isinstance(fn, GraphFunction):
        return (fn.graph,)

    elif isinstance(fn, (PrimitiveFunction, TypedPrimitive)):
        return (fn.prim,)

    else:
        raise AssertionError(f'Unsupported: {fn}')


def _graphs_from(calls):
    return [x[0] for x in calls]


def _subslices(seq, keep=0):
    for i in range(len(seq) - keep + 1):
        yield seq[:-i] if i else seq


class DeadDataElimination(Partializable):
    """Eliminate expressions that compute unretrieved data."""

    def __init__(self, optimizer=None):
        """Initialize a DeadDataElimination."""
        self.optimizer = optimizer

    def output_structure(self, root):
        mng = root.manager
        mng.keep_roots(root)

        structure = {}

        def collect_from(node):
            if node.is_apply(P.make_tuple):
                return {i: ((node, i + 1), collect_from(inp))
                        for i, inp in enumerate(node.inputs[1:])}
            elif node.is_apply(P.env_setitem):
                _, env, key, value = node.inputs
                envd = collect_from(env)
                # It could be a non-dict in user code that uses env_setitem on
                # something other than newenv
                assert isinstance(envd, dict)
                return {**envd, key.value: ((node, 3), collect_from(value))}
            elif node.is_constant() and node.value == newenv:
                return {}
            else:
                return node

        for g in mng.graphs:
            structure[g] = collect_from(g.output)

        return structure

    def structural_dependencies(self, root):
        mng = root.manager
        mng.keep_roots(root)

        finished = defaultdict(list)
        cache = {}

        def collect_deps(node):

            if node in cache:
                return cache[node]

            if node.is_apply(P.tuple_getitem):
                _, x, key = node.inputs
                rval = (*collect_deps(x), key.value)

            elif node.is_apply(P.env_getitem):
                _, x, key, _ = node.inputs
                rval = (*collect_deps(x), key.value)

            elif node.is_apply():
                for inp in node.inputs:
                    finished[inp].append(collect_deps(inp))

                fn, *args = node.inputs
                args = [a.abstract for a in args]
                fna = fn.abstract
                assert isinstance(fna, AbstractFunction)

                calls = [_flatten_call(f) for f in fna.get_sync()]

                rval = []

                for f, *args1 in calls:
                    if f is P.array_map:
                        f = (*args1, *args)[0]
                        calls = [_flatten_call(f2) for f2 in f.get_sync()]
                        finished[node].append((_graphs_from(calls),))
                    elif isinstance(f, Graph):
                        rval.append(f)

                rval = tuple(rval) if rval else (None,)
                rval = rval,

            else:
                rval = (None,)

            cache[node] = rval
            return rval

        for node in mng.all_nodes:
            collect_deps(node)

        return finished

    def full_structural_dependencies(self, root):

        structure = self.output_structure(root)
        paths = self.structural_dependencies(root)

        results = {}

        def helper(x):
            if isinstance(x, dict):
                rval = {}
                ttotal = set()
                for k, (node, v) in x.items():
                    path = helper(v)
                    if isinstance(path, dict):
                        total = set()
                        for k2, (node2, v2) in path.items():
                            rval[(k, *k2)] = (node2, v2)
                            total.update(v2)
                            ttotal.update(v2)
                        rval[(k,)] = (node, total)
                    else:
                        rval[(k,)] = (node, path)
                return rval
            else:
                all_paths = set()
                for node in dfs(x, succ_incoming):
                    possibilities = paths.get(node, [])
                    for (start, *path) in possibilities:
                        for fn in start or (None,):
                            if fn is not None:
                                all_paths.add((fn, *path))
                return all_paths

        for g, s in structure.items():
            res = helper(s)
            if isinstance(res, dict):
                total = set()
                for k, (_, contrib) in res.items():
                    assert isinstance(contrib, set)
                    total.update(contrib)
                res[()] = ((g.return_, 1), total)
                results[g] = res
            else:
                results[g] = {(): ((g.return_, 1), res)}

        return results

    def find_dead_paths(self, root):
        fulldeps = self.full_structural_dependencies(root)

        seen = set()
        keep = set()

        def succ(path):
            if path in seen:
                return
            g, *path = path
            if g not in fulldeps:
                return
            path = tuple(path)
            sg = fulldeps[g]
            while path and path not in sg:
                path = path[:-1]
            seen.add((g, *path))
            keep.update(_subslices((g, *path), keep=1))
            node, paths = sg[path]
            for p in paths:
                yield p

        for path in fulldeps.get(root, []):
            for _ in dfs((root, *path), succ):
                pass

        missing = {}
        for g, paths in fulldeps.items():
            missing[g] = set((node, p) for (p, (node, _)) in paths.items()
                             if (g, *p) not in keep
                             and all(subp not in seen
                                     for subp in _subslices((g, *p), keep=1)))
        return missing

    def __call__(self, root):
        """Apply dead data elimination."""
        missing = self.find_dead_paths(root)
        mng = root.manager
        for g, dead in missing.items():
            if g not in mng.graphs:
                continue
            for (node, idx), _ in dead:
                repl = Constant(DEAD)
                # repl.abstract = AbstractError(DEAD)
                repl.abstract = node.inputs[idx].abstract
                mng.set_edge(node, idx, repl)
        return False  # Pretend there are no changes, for now
