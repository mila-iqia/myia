
from ..util import Event, Keyword, buche
from ..stx import Lambda, Closure, Tuple, Symbol
from collections import defaultdict
from ..impl.flow_all import default_flow, ANY, VALUE
from ..impl.main import impl_bank
from ..symbols import builtins
from .types import Int64


class DFA:
    def __init__(self, tracks, genv):
        tracks = [track(self) for track in tracks]
        self.tracks = {track.name: track for track in tracks}
        self.value_track = self.tracks['value']
        self.genv = genv
        self.flow_graph = {}
        self.values = {track: defaultdict(set)
                       for track in self.tracks.values()}

    def propagate(self, node, track, value):
        if isinstance(track, str):
            track = self.tracks[track]
        vals = self.values[track]
        if value not in vals[node]:
            vals[node].add(value)
            self.flow_graph[node](track, value)

    def propagate_value(self, node, value):
        self.propagate(node, self.value_track, value)

    def run_flows(self, method, *args):
        for track in self.tracks.values():
            getattr(track, f'flow_{method}')(*args)

    def flow_to(self, a, b):
        @self.on_flow_from(a)
        def flow(track, value):
            if track.direction == 'forwards':
                self.propagate(b, track, value)

        @self.on_flow_from(b)
        def flow(track, value):
            if track.direction == 'backwards':
                self.propagate(a, track, value)

    def on_flow_from(self, node, require_track=None):
        if isinstance(require_track, str):
            require_track = self.tracks[require_track]

        def deco(fn):
            reg = self.flow_graph[node].register

            @reg
            def flow(_, track, value):
                if not require_track or track is require_track:
                    fn(track, value)
            for track in self.tracks.values():
                for v in self.values[track][node]:
                    flow(None, track, v)
        return deco

    def on_function_flow(self, fn, args, node, flow_body=True):

        def on_new_fn_value(track, new_value):
            nonlocal args
            assert track is self.value_track
            if isinstance(new_value, Lambda):
                if flow_body:
                    self.flow_to(new_value.body, node)
                for arg, lbda_arg in zip(args, new_value.args):
                    self.flow_to(arg, lbda_arg)
            elif isinstance(new_value, Closure):
                args = new_value.args + args
                self.on_function_flow(new_value.fn, args, node, flow_body)
            elif new_value in builtins.__dict__.values():
                if flow_body:
                    self.run_flows('prim', new_value, args, node)
            elif new_value is ANY:
                self.propagate_value(node, ANY)
            else:
                raise Exception(
                    f'Cannot flow a non-function here: {new_value}'
                )
        self.on_flow_from(fn, self.value_track)(on_new_fn_value)

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

        self.on_function_flow(node.fn, node.args, node, True)
        self.run_flows('Apply', node)

    def visit_Begin(self, node):
        raise Exception('Begin not supported')
        self.run_flows('Begin', node)

    def visit_Closure(self, node):
        self.visit(node.fn)
        for arg in node.args:
            self.visit(arg)
        self.on_function_flow(node.fn, node.args, node, False)
        self.propagate_value(node, node)
        self.run_flows('Closure', node)

    def visit_Lambda(self, node):
        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)
        self.propagate_value(node, node)
        self.run_flows('Lambda', node)

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
                @self.on_flow_from(value)
                def flow_tuple(track, new_value):
                    if isinstance(new_value, Tuple):
                        for v, sub_value in zip(var.values, new_value.values):
                            _bind(v, sub_value)
                    else:
                        for v in _vars(var):
                            self.propagate_value(v, ANY)
            else:
                self.flow_to(value, var)

        for v, value in node.bindings:
            self.visit(value)
            _visit(v)
            _bind(v, value)
        self.visit(node.body)
        self.flow_to(node.body, node)
        self.run_flows('Let', node)

    def visit_Symbol(self, node):
        if node in self.genv.bindings:
            b = self.genv[node]
            self.visit(b)
            self.propagate_value(node, b)
        elif node in builtins.__dict__.values():
            self.propagate_value(node, node)
        self.run_flows('Symbol', node)

    def visit_Tuple(self, node):
        for v in node.values:
            self.visit(v)
        self.run_flows('Tuple', node)

    def visit_Value(self, node):
        self.run_flows('Value', node)


class Track:
    def __init__(self, name, dfa, direction='forwards'):
        self.name = name
        self.dfa = dfa
        assert direction in {'forwards', 'backwards'}
        self.direction = direction

    def propagate(self, node, value):
        self.dfa.propagate(node, self, value)

    def flow_Apply(self, node):
        pass

    def flow_Begin(self, node):
        pass

    def flow_Closure(self, node):
        pass

    def flow_Lambda(self, node):
        pass

    def flow_Let(self, node):
        pass

    def flow_Tuple(self, node):
        pass

    def flow_Symbol(self, node):
        pass

    def flow_Value(self, node):
        pass

    def flow_prim(self, prim, args, node):
        impls = impl_bank['flow'][self.name]
        flow = impls.get(prim, self.flow_default)
        flow(self.dfa, args, node)

    def flow_default(self, _, args, node):
        self.propagate(node, ANY)

    def __str__(self):
        return f'<Track:{self.name}>'


class ValueTrack(Track):
    def __init__(self, dfa):
        super().__init__('value', dfa)

    def flow_Tuple(self, node):
        self.propagate(node, node)

    def flow_Value(self, node):
        self.propagate(node, node)


class TypeTrack(Track):
    def __init__(self, dfa):
        super().__init__('type', dfa)

    def flow_Value(self, node):
        if isinstance(node.value, int):
            self.propagate(node, Int64)
        elif isinstance(node.value, float):
            self.propagate(node, Float64)


_type = builtins.type
_shape = builtins.shape

needs_map = {
    builtins.add: {
        _type: (_type, _type),
        _shape: (_shape, _shape)
    },
    builtins.subtract: {
        _type: (_type, _type),
        _shape: (_shape, _shape)
    },
    builtins.dot: {
        _type: (_type, _type),
        _shape: (_shape, _shape)
    },
    builtins.less: {
        _type: (_type, _type),
        _shape: False
    },
    builtins.greater: {
        _type: (_type, _type),
        _shape: False
    },
    builtins.index: {
        _type: (VALUE, _type)
    },
    builtins.identity: {
        _type: (_type,),
        _shape: (_shape,)
    },
    builtins.switch: {
        _type: (_type, _type, _type),
        VALUE: ((), VALUE, VALUE)
    },
    'default': {
    }
}


class NeedsTrack(Track):
    def __init__(self, dfa, autoflow=[]):
        super().__init__('needs', dfa, 'backwards')
        self.autoflow = autoflow

    def flow_auto(self, node):
        assert not isinstance(node, Closure)
        for prop in self.autoflow:
            self.propagate(node, prop)

    def flow_Apply(self, node):
        self.propagate(node.fn, VALUE)
        self.flow_auto(node)

    def flow_Begin(self, node):
        self.flow_auto(node)

    def flow_Closure(self, node):
        pass

    def flow_Lambda(self, node):
        pass

    def flow_Let(self, node):
        self.flow_auto(node)

    def flow_Tuple(self, node):
        self.flow_auto(node)

    def flow_Symbol(self, node):
        self.flow_auto(node)

    def flow_Value(self, node):
        self.flow_auto(node)

    def flow_prim(self, prim, args, node):
        m = needs_map.get(prim, needs_map['default'])
        # print(prim)

        @self.dfa.on_flow_from(node, 'needs')
        def on_need(track, value):
            needs = m.get(value, None)
            if needs is None:
                needs = [VALUE for arg in node.args]
            elif needs is False:
                return

            for arg, ns in zip(node.args, needs):
                if not isinstance(ns, (list, tuple)):
                    ns = [ns]
                for n in ns:
                    self.propagate(arg, n)

    def flow_default(self, _, args, node):
        pass
