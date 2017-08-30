
from ..util import Event, Keyword
from ..stx import Lambda, Closure, Tuple, Symbol
from collections import defaultdict
from ..impl.flow_all import default_flow, ANY
from ..impl.main import impl_bank
from ..symbols import builtins


class DFA:
    def __init__(self, genv):
        self.genv = genv
        self.flow_graph = {}
        self.values = defaultdict(set)

    def flow_to(self, a, b):
        @self.on_flow_from(a)
        def flow(value):
            self.add_value(b, value)

    def on_flow_from(self, node):
        def deco(fn):
            reg = self.flow_graph[node].register

            @reg
            def flow(_, value):
                fn(value)
            for v in self.values[node]:
                flow(None, v)
        return deco

    def on_function_flow(self, fn, node, flow_body=True, closure_args=0):
        args = node.args

        def on_new_fn_value(new_value):
            if isinstance(new_value, Lambda):
                if flow_body:
                    self.flow_to(new_value.body, node)
                lbda_args = new_value.args[closure_args:]
                for arg, lbda_arg in zip(args, lbda_args):
                    self.flow_to(arg, lbda_arg)
            elif isinstance(new_value, Closure):
                nargs = closure_args + len(new_value.args)
                self.on_function_flow(new_value.fn, node, flow_body, nargs)
            elif new_value in builtins.__dict__.values():
                if flow_body:
                    flow = impl_bank['flow'].get(new_value, default_flow)
                    flow(self, node)
            elif new_value is ANY:
                self.add_value(node, ANY)
            else:
                raise Exception(
                    f'Cannot flow a non-function here: {new_value}'
                )
        self.on_flow_from(fn)(on_new_fn_value)

    def add_value(self, node, value):
        if value not in self.values[node]:
            self.values[node].add(value)
            self.flow_graph[node](value)

    def visit(self, node):
        if node in self.flow_graph:
            return self.flow_graph[node]
        cls = node.__class__.__name__
        flow = Event(f'flow_{cls}')
        self.flow_graph[node] = flow
        method = getattr(self, f'visit_{cls}')
        return method(node)

    def visit_Apply(self, node):
        # (f a)
        # If (lambda (x) body) ~> f and v ~> a, v ~> x
        # If (lambda (x) body) ~> f and v ~> body, v ~> (f a)

        flow = self.flow_graph[node]
        self.visit(node.fn)
        for a in node.args:
            self.visit(a)

        self.on_function_flow(node.fn, node, True, 0)

    def visit_Begin(self, node):
        raise Exception('Begin not supported')

    def visit_Closure(self, node):
        self.visit(node.fn)
        for arg in node.args:
            self.visit(arg)
        self.on_function_flow(node.fn, node, False, 0)
        self.add_value(node, node)

    def visit_Lambda(self, node):
        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)
        self.add_value(node, node)

    def visit_Let(self, node):
        def _visit(v):
            if isinstance(v, Tuple):
                for _v in v.values:
                    _visit(_v)
            else:
                self.visit(v)

        def _vars(v):
            if isinstance(v, Tuple):
                rval = []
                for _v in v.values:
                    rval += _vars(_v)
                return rval
            else:
                return [v]

        def _bind(var, value):
            if isinstance(var, Tuple):
                all_vars = _vars(var)

                @self.on_flow_from(value)
                def flow_tuple(new_value):
                    if isinstance(new_value, Tuple):
                        for v, sub_value in zip(var.values, new_value.values):
                            _bind(v, sub_value)
                    else:
                        for v in all_vars:
                            self.add_value(v, ANY)
            else:
                self.flow_to(value, var)

        for v, value in node.bindings:
            self.visit(value)
            _visit(v)
            _bind(v, value)
        self.visit(node.body)
        self.flow_to(node.body, node)

    def visit_Symbol(self, node):
        if node in self.genv.bindings:
            b = self.genv[node]
            self.visit(b)
            self.add_value(node, b)
        elif node in builtins.__dict__.values():
            self.add_value(node, node)

    def visit_Tuple(self, node):
        for v in node.values:
            self.visit(v)
        self.add_value(node, node)

    def visit_Value(self, node):
        self.add_value(node, node)
