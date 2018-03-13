"""Graph analysis framework."""
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple as TupleT, \
    Optional, List

from myia.ir.anf import Graph, Apply, Constant, ANFNode, Parameter
from myia.ir.utils import is_constant_graph
from myia.cconv import NestingAnalyzer
from myia.graph_utils import toposort

from myia.prim import implementations as py_implementations, Primitive, MetaVar
from myia.prim.py_implementations import typeof
from myia.prim.ops import if_
from myia.unify import Unification, var
from myia.utils import HierDict
from myia.dtype import Type, Function

from .value import ESTIMATORS, NO_VALUE


class Frame:
    def __init__(self, parent: 'Frame', nodes: Iterable[ANFNode], *,
                 types: Mapping[ANFNode, Type] = None,
                 values: Mapping[ANFNode, Any] = None) -> None:
        if types is None:
            types = {}
        if values is None:
            values = {}

        self.nodes = nodes
        self.parent = parent
        self.cur_node: Optional[ANFNode] = None
        self.aux: Dict = dict()

        if parent:
            self.types: Mapping[ANFNode, Type] = HierDict(parent.types)
            self.types.update(types)
            self.values: Mapping[ANFNode, Any] = HierDict(parent.values)
            self.values.update(values)
            self.depth: int = parent.depth + 1
        else:
            self.types = types
            self.values = values
            self.depth = 0

    @property
    def path(self):
        f = self.parent
        p = []
        while f:
            p.append(f.cur_node)
            f = f.parent
        return tuple(reversed(p))


class Call:
    def __init__(self, node, *, types=None, values=None):
        self.node = node
        self.types = types
        self.values = values


class GraphAnalyzer:
    """Perform abstract interpretation of a graph deriving types and values."""

    def __init__(self, *, implementations: Mapping[Primitive, Callable] = None,
                 estimators: Mapping[Primitive, Callable] = None,
                 max_depth=1000) -> None:
        """Create the basic attributes."""
        self.signatures: Dict[Graph, Function] = dict()
        self.eval_cache: Dict[Graph, Dict] = dict()
        self.paths: Dict[TupleT[ANFNode, ...], Frame] = dict()
        self.values: Dict[ANFNode, Any] = dict()
        self.types: Dict[ANFNode, Type] = dict()
        self._equiv: Dict = dict()
        if implementations is None:
            implementations = py_implementations
        self._implementations = implementations
        if estimators is None:
            estimators = ESTIMATORS
        self._estimators = estimators
        self.max_depth = max_depth
        self.U = Unification()

    def _handle_graph(self, graph: Graph):
        if graph not in self.eval_cache:
            self.eval_cache[graph] = dict()
        fn_t = Function((var() for p in graph.parameters), var())
        self.signatures[graph] = fn_t

    def _handle_node(self, node: ANFNode, frame):
        if isinstance(node, Constant):
            self._handle_constant(node)

        elif isinstance(node, Parameter):
            pass

        elif isinstance(node, Apply):
            return self._handle_apply(node, frame)

        else:
            raise AssertionError("Unknown node type")

    def _handle_constant(self, node):
        if is_constant_graph(node):
            self.types[node] = self.signatures[node.value]
            # TODO: handle closures
            self.values[node] = node.value
        else:
            self.types[node] = typeof(node.value)
            self.values[node] = node.value

    def _handle_apply(self, node, frame):
        fn = frame.values[node.inputs[0]]

        if isinstance(fn, Primitive):
            if fn == if_:
                return self._handle_apply_if(node, frame)
            else:
                self._handle_apply_primitive(node, frame)

        elif isinstance(fn, Graph):
            return self._handle_apply_graph(node, frame)

        else:
            return self._handle_apply_generic(node, frame)

    def _handle_apply_if(self, node, frame):
        cond = frame.values[node.inputs[1]]

        if frame.cur_node is node:
            fn_true, fn_false = frame.aux[(node, 'fn')]
        else:
            fn_true = Apply([node.inputs[2]], node.graph)
            fn_false = Apply([node.inputs[3]], node.graph)
            frame.aux[(node, 'fn')] = fn_true, fn_false

        def fake_call(node, frame, branch):

            def extract(frame, branch):
                type = frame.types[branch]
                del frame.types[branch]
                val = frame.values[branch]
                del frame.values[branch]
                return (type, val)

            if frame.cur_node is node:
                fpath = frame.path + (node,)
                bpath = frame.path + (branch,)
                self.paths[bpath] = self.paths[fpath]
                frame.cur_node = branch
                res = self._handle_apply(branch, frame)
                if res is None:
                    del self.paths[fpath]
                    self.paths[fpath + (branch,)] = self.paths[bpath]
                    del self.paths[bpath]
                    return extract(frame, branch)
                return res  # pragma: no cover
            else:
                res = self._handle_apply(branch, frame)
                if res is None:
                    return extract(frame, branch)  # pragma: no cover
                return res

        if cond is NO_VALUE:
            k = (node, 'res')
            if k not in frame.aux:
                res = fake_call(node, frame, fn_true)
                if isinstance(res, tuple):
                    frame.aux[k] = res
                    res = None
                if res is not None:
                    return res
            res = fake_call(node, frame, fn_false)
            if isinstance(res, tuple):
                ftype = res[0]
                fval = res[1]
                ttype = frame.aux[k][0]
                tval = frame.aux[k][1]
                if fval != tval:
                    frame.values[node] = NO_VALUE
                else:
                    frame.values[node] = fval
                if self.U.unify(ftype, ttype, self._equiv) is None:
                    raise TypeError("Branches of if with incompatible types")
                else:
                    frame.types[node] = self.U.reify(ftype, self._equiv)
                return None
            return res

        else:
            if cond:
                res = fake_call(node, frame, fn_true)
                if isinstance(res, tuple):
                    frame.types[node] = res[0]
                    frame.values[node] = res[1]
                    return None
                return res
            else:
                res = fake_call(node, frame, fn_false)
                if isinstance(res, tuple):
                    frame.types[node] = res[0]
                    frame.values[node] = res[1]
                    return None
                return res

    def _handle_apply_primitive(self, node, frame):
        fni = node.inputs[0]
        args = node.inputs[1:]
        fn_t = frame.types[fni]
        fn = frame.values[fni]

        if isinstance(fn_t, MetaVar):
            frame.types[node] = fn_t.infer(args, frame, self._equiv)
        else:
            args_t = tuple(frame.types[a] for a in args)
            call_t = Function(args_t, var())
            if self.U.unify(call_t, self.U.clone(fn_t), self._equiv) is None:
                raise TypeError("Incompatible apply")
            frame.types[node] = self.U.reify(call_t.retval, self._equiv)

        args_v = tuple(frame.values[a] for a in args)

        if all(a is not NO_VALUE for a in args_v):
            frame.values[node] = self._implementations[fn](*args_v)

        else:
            if fn in self._estimators:
                frame.values[node] = self._estimators[fn](*args_v)
            else:
                frame.values[node] = NO_VALUE

    def _handle_apply_graph(self, node, frame):
        fn = node.inputs[0]
        graph = frame.values[fn]
        args = node.inputs[1:]
        args_t = tuple(frame.types[a] for a in args)
        args_v = tuple(frame.values[a] for a in args)
        fn_t = self.signatures[graph]
        cache = self.eval_cache[graph]
        cpath = frame.path + (node,)
        cache_k = (args_v, frozenset((v, frame.values[v])
                                     for v in self._vars[graph]))

        if frame.cur_node is not node and cpath not in self.paths:
            need_call = False

            call_t = Function(args_t, var())
            if self.U.unify(fn_t, call_t, self._equiv) is None:
                raise TypeError("Bad call")
            frame.types[node] = self.U.reify(call_t.retval, self._equiv)

            if cache_k not in cache:
                cache[cache_k] = NO_VALUE
                need_call = True

            frame.values[node] = cache[cache_k]

            params = graph.parameters
            assert len(params) == len(args_t)

            if frame.depth < self.max_depth and need_call:
                p_types = zip(params, args_t)
                p_vals = zip(params, args_v)
                return Call(graph.return_, types=p_types, values=p_vals)
            else:
                return None  # We'll just guess NO_VALUE here.

        else:
            cframe = self.paths[cpath]
            ret_t = cframe.types[graph.return_]
            ret_v = cframe.values[graph.return_]
            call_t = Function(args_t, ret_t)
            self.U.unify(call_t, fn_t, self._equiv)
            self.signatures[graph] = self.U.reify(fn_t, self._equiv)
            cache[cache_k] = ret_v
            frame.types[node] = self.U.reify(ret_t, self._equiv)
            frame.values[node] = ret_v

    def _handle_apply_generic(self, node, frame):
        fn = node.inputs[0]
        args = node.inputs[1:]
        fn_t = frame.types[fn]
        args_t = tuple(frame.types[a] for a in args)
        call_t = Function(args_t, var())

        equiv = self.U.unify(fn_t, call_t, self._equiv)
        if equiv is None:
            raise TypeError("Invalid Apply")
        frame.types[node] = self.U.reify(call_t.retval, self._equiv)
        frame.values[node] = NO_VALUE

    def infer_type(self, graph: Graph, types: Iterable[Type]) -> Type:
        if graph not in self.signatures:
            raise ValueError("Unknown graph")
        equiv = dict(self._equiv)
        fn_t = self.signatures[graph]
        res_t = var()
        call_t = Function(tuple(types), res_t)
        if self.U.unify(fn_t, call_t, equiv):
            return self.U.reify(res_t, equiv)
        else:
            raise TypeError("Incompatible apply")

    def infer_args(self, graph: Graph, args: Iterable[Any]) -> Type:
        return self.infer_type(graph, (typeof(a) for a in args))

    def analyze(self, graph: Graph) -> None:
        """Analyze a graph.

        This will run the graph through abstract interpretation and
        evaluate types and values wherever possible.

        Parameters
        ----------
            graph: Graph

        """
        self._handle_graph(graph)
        N = NestingAnalyzer(graph)
        for g in N.coverage():
            self._handle_graph(g)

        self._vars = N.free_variables_total()

        def succ_free(node):
            """Follow node.incoming and free variables."""
            yield from node.inputs
            if is_constant_graph(node):
                yield from self._vars[node.value]

        csts: List[ANFNode] = [Constant(graph)]
        csts.extend(Constant(None) for p in graph.parameters)
        p_types = HierDict(self.types,
                           zip(csts[1:], self.signatures[graph].arguments))
        p_types[csts[0]] = var()
        p_vals = HierDict(self.values,
                          ((p, NO_VALUE) for p in csts[1:]))
        p_vals[csts[0]] = graph

        root_node = Apply(csts, None)
        # We fake a "call" to the root function
        frame = Frame(None, [root_node],
                      types=HierDict(self.types, p_types),
                      values=HierDict(self.values, p_vals))

        def _inner(node, frame):
            res = self._handle_node(node, frame)
            if res is not None:
                frame.cur_node = node
                frame = Frame(frame, toposort(res.node, succ_free),
                              types=res.types, values=res.values)
                self.paths[frame.path] = frame
                return frame

        def _outer_loop(frame):
            if frame.cur_node is not None:
                new_frame = _inner(frame.cur_node, frame)
                if new_frame is not None:
                    return new_frame
            for node in frame.nodes:
                new_frame = _inner(node, frame)
                if new_frame is not None:
                    return new_frame
            return frame.parent

        while frame:
            frame = _outer_loop(frame)

        self._merge_values()

    def _merge_values(self):
        for f in self.paths.values():
            for n, v in f.values.items():
                if n not in self.values:
                    self.values[n] = v
                else:
                    if self.values[n] != v:
                        self.values[n] = NO_VALUE
