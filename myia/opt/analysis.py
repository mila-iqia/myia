
from collections import defaultdict

from ..prim import ops as P
from ..utils import newenv
from ..abstract import AbstractFunction, GraphFunction, PartialApplication, \
    DEAD, AbstractError
from ..graph_utils import dfs
from ..ir import Constant, succ_incoming


def analyze_structure(root):
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
            if isinstance(envd, dict):
                # return {**envd, key.value: (value, collect_from(value))}
                return {**envd, key.value: ((node, 3), collect_from(value))}
            else:
                return node
        elif node.is_constant() and node.value == newenv:
            return {}
        else:
            return node

    for g in mng.graphs:
        structure[g] = collect_from(g.output)

    return structure


def analyze_structural_deps(root):
    mng = root.manager
    mng.keep_roots(root)

    finished = {}
    cache = {}

    def collect_fngraph(fn):
        if isinstance(fn, GraphFunction):
            return fn.graph
        elif isinstance(fn, PartialApplication):
            return collect_fngraph(fn.fn)
        else:
            return None

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
                finished[inp] = collect_deps(inp)
            fn, *_ = node.inputs
            fna = fn.abstract
            assert fna is None or isinstance(fna, AbstractFunction)
            rval = (fna and tuple(collect_fngraph(f)
                                  for f in fna.get_sync()),)

        else:
            rval = (None,)

        cache[node] = rval
        return rval

    for node in mng.all_nodes:
        collect_deps(node)

    # from collections import defaultdict
    # paths = defaultdict(set)
    # for node, seq in finished.items():
    #     start, *path = seq
    #     if start is None:
    #         continue
    #     for fn in start:
    #         paths[fn].add(tuple(path))

    # return paths

    return finished


def analyze_final(root):

    structure = analyze_structure(root)
    paths = analyze_structural_deps(root)

    results = {}

    def helper(x):
        if isinstance(x, dict):
            # return {k: (node, helper(v)) for k, (node, v) in x.items()}
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
                # for path in helper(v):
                #     if isinstance(path, dict):
                #         for k2, v2 in path.items():
                #             rval[(k, *k2)] = v2
                #     else:
                #         rval[(k,)] = path
            # rval[()] = ()
            return rval
        else:
            all_paths = set()
            for node in dfs(x, succ_incoming):
                start, *path = paths.get(node, (None,))
                for fn in start or (None,):
                    if fn is not None:
                        all_paths.add((fn, *path))
            return all_paths

    for g, s in structure.items():
        res = helper(s)
        if isinstance(res, dict):
            results[g] = res
        else:
            results[g] = {(): ((g.return_, 1), res)}

    return results


def _subslices(seq, keep=0):
    for i in range(len(seq) - keep):
        yield seq[:-i] if i else seq


def find_dead_paths(root):
    structure = analyze_final(root)

    seen = set()
    keep = set()

    def succ(path):
        if path in seen:
            return
        # seen.add(path)
        # for i in range(len(path) - 1):
        #     seen.add(path[:-(i + 1)])

        # seen.update(_subslices(path, keep=1))

        # seen.add(path)

        g, *path = path
        if g not in structure:
            return
        path = tuple(path)
        sg = structure[g]
        while path and path not in sg:
            path = path[:-1]
        seen.add((g, *path))
        keep.update(_subslices((g, *path), keep=1))
        node, paths = sg[path]
        for p in paths:
            yield p

    for path in structure.get(root, []):
        for _ in dfs((root, *path), succ):
            pass

    view = defaultdict(set)
    for g, *p in seen:
        view[g].add(tuple(p))

    view2 = defaultdict(set)
    for g, paths in structure.items():
        if not isinstance(paths, dict):
            continue
        view2[g] = set(paths)

    view3 = defaultdict(set)
    missing = {}
    for g, paths in structure.items():
        if not isinstance(paths, dict):
            continue
        # missing[g] = set((node, p) for (p, (node, _)) in paths.items()
        #                  if all(subp not in seen
        #                         for subp in _subslices((g, *p), keep=1)))

        # view3[g] = set(p for (p, (node, _)) in paths.items()
        #                if (g, *p) not in keep
        #                and all(subp not in seen
        #                        for subp in _subslices((g, *p), keep=1)))

        missing[g] = set((node, p) for (p, (node, _)) in paths.items()
                         if (g, *p) not in keep
                         and all(subp not in seen
                                 for subp in _subslices((g, *p), keep=1)))

    # buche.dict(seen=view, all=view2, minus=view3)

    return missing


def dead_data_elimination(root):
    missing = find_dead_paths(root)
    # ibuche(missing)
    mng = root.manager
    for g, dead in missing.items():
        if g not in mng.graphs:
            continue
        for (node, idx), _ in dead:
            repl = Constant(DEAD)
            repl.abstract = AbstractError(DEAD)
            mng.set_edge(node, idx, repl)
        # for node, _ in dead:
        #     repl = Constant(DEAD)
        #     repl.abstract = AbstractError(DEAD)
        #     mng.replace(node, repl)
