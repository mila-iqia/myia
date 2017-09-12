"""
Dataflow analysis, akin to 0-CFA.

See Inference section of DEVELOPERS.md for some more information.
"""


from ..util import Event, Keyword, buche
from ..stx import LambdaNode, ClosureNode, TupleNode, Symbol
from collections import defaultdict
from ..impl.flow_all import default_flow, ANY, VALUE
from ..impl.main import impl_bank
from ..symbols import builtins
from .types import Int64, Float64


class DFA:
    """
    Create a DFA instance.

    The DFA can propagates information on "tracks". Each track
    can define custom behavior, and can work together.
    The main track is a ValueTrack, which propagates values such
    as LambdaNodes and ClosureNodes, but there are also TypeTracks and
    NeedsTracks.

    Attributes:
        tracks: Associates track names to Track instances.
        value_track: Shortcut for ``dva.values['value']``
        genv: ParseEnv containing the global environment, in
            order to resolve constant symbols.
        flow_events: Maps MyiaASTNodes to Event instances that
            can be called when a value must be propagated to
            that node, and can be listened to.
        values: Maps each track to a map of each node to a set
            of possible values flowing to that node.
    """
    def __init__(self, tracks, genv):
        tracks = [track(self) for track in tracks]
        self.tracks = {track.name: track for track in tracks}
        self.value_track = self.tracks['value']
        self.genv = genv
        self.flow_events = {}
        self.values = {track: defaultdict(set)
                       for track in self.tracks.values()}

    def propagate(self, node, track, value):
        """
        Declare that the given node has the given value on the
        given track, and propagate that information.
        """
        if isinstance(track, str):
            track = self.tracks[track]
        vals = self.values[track]
        if value not in vals[node]:
            vals[node].add(value)
            self.flow_events[node](track, value)

    def propagate_value(self, node, value):
        """
        Shorthand for ``dfa.propagate(node, 'value', value)``
        """
        self.propagate(node, self.value_track, value)

    def run_flows(self, method, *args):
        """
        Run the ``flow_<method>`` method for each track on
        the given arguments.
        """
        for track in self.tracks.values():
            getattr(track, f'flow_{method}')(*args)

    def flow_to(self, a, b):
        """
        Declare that all values that flow to ``a`` also flow to
        ``b`` on all tracks marked as ``'forwards'``. On tracks
        marked as ``'backwards'``, values flow from ``b`` to
        ``a`` instead.
        """
        @self.on_flow_from(a)
        def flow(track, value):
            d = track.direction
            if d == 'forwards' or d == 'bidirectional':
                self.propagate(b, track, value)
            self.propagate(b, track, value)

        @self.on_flow_from(b)
        def flow(track, value):
            d = track.direction
            if d == 'backwards' or d == 'bidirectional':
                self.propagate(a, track, value)

    def on_flow_from(self, node, require_track=None):
        """
        Returns a decorator for a function that must be triggered
        every time a value is propagated to ``node``, optionally
        filtering by track. If values are already associated to
        ``node``, the function will be called immediately for each
        of them.
        """
        if isinstance(require_track, str):
            require_track = self.tracks[require_track]

        def deco(fn):
            reg = self.flow_events[node].register

            @reg
            def flow(_, track, value):
                # The first argument is the event instance, we
                # don't need it.
                if not require_track or track is require_track:
                    fn(track, value)

            # Call the function for all tracks and values
            # TODO: avoid doing this for tracks that are not
            # required!
            for track in self.tracks.values():
                for v in self.values[track][node]:
                    flow(None, track, v)
        return deco

    def function_flow(self, fn, args, node, flow_body=True):
        """
        Properly wire the flow around a function application.

        Arguments:
            fn: A node to which functions will flow.
            args: Nodes that are given to fn as arguments.
            node: The node for the whole application.
            flow_body: Whether a function's body ought to
                flow to the node (this will be False for
                closures).
        """

        def on_new_fn_value(track, new_value):
            nonlocal args
            assert track is self.value_track
            if isinstance(new_value, LambdaNode):
                if flow_body:
                    # The LambdaNode's body flows to its application's
                    # result.
                    self.flow_to(new_value.body, node)
                for arg, lbda_arg in zip(args, new_value.args):
                    # Each argument to the function flows to the
                    # LambdaNode's corresponding argument.
                    self.flow_to(arg, lbda_arg)
            elif isinstance(new_value, ClosureNode):
                # If we get a ClosureNode, we accumulate the args in front
                # and we repeat the procedure for the ClosureNode's function,
                # which will receive the totality of the arguments.
                args = new_value.args + args
                self.function_flow(new_value.fn, args, node, flow_body)
            elif new_value in builtins.__dict__.values():
                # If we get a Primitive, we dispatch to the tracks.
                if flow_body:
                    self.run_flows('prim', new_value, args, node)
            elif new_value is ANY:
                # Whatever goes.
                self.propagate_value(node, ANY)
            else:
                # TODO: This is more brutal than necessary.
                raise Exception(
                    f'Cannot flow a non-function here: {new_value}'
                )
        self.on_flow_from(fn, self.value_track)(on_new_fn_value)

    def visit(self, node):
        if node in self.flow_events:
            return self.flow_events[node]
        cls = node.__class__.__name__
        flow = Event(f'flow_{cls}')
        self.flow_events[node] = flow
        method = getattr(self, f'visit_{cls}')
        return method(node)

    def visit_ApplyNode(self, node):
        # (f a)
        # If (lambda (x) body) ~> f and v ~> a, v ~> x
        # If (lambda (x) body) ~> f and v ~> body, v ~> (f a)
        flow = self.flow_events[node]
        self.visit(node.fn)
        for a in node.args:
            self.visit(a)

        self.function_flow(node.fn, node.args, node, True)
        self.run_flows('ApplyNode', node)

    def visit_BeginNode(self, node):
        raise Exception('Begin not supported')
        self.run_flows('BeginNode', node)

    def visit_ClosureNode(self, node):
        # ClosureNodes flow to themselves.
        self.visit(node.fn)
        for arg in node.args:
            self.visit(arg)
        self.function_flow(node.fn, node.args, node, False)
        self.propagate_value(node, node)
        self.run_flows('ClosureNode', node)

    def visit_LambdaNode(self, node):
        # LambdaNodes flow to themselves.
        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)
        self.propagate_value(node, node)
        self.run_flows('LambdaNode', node)

    def visit_LetNode(self, node):
        # There is some complex behavior here, mainly because it is possible
        # to bind to tuples, and we want the analysis to deconstruct them
        # whenever possible.
        def _visit(v):
            # Visit all variables through tuples.
            if isinstance(v, TupleNode):
                for _v in v.values:
                    _visit(_v)
            else:
                self.visit(v)

        def _vars(v):
            # Return a flattened list of all variables defined here.
            if isinstance(v, TupleNode):
                rval = []
                for _v in v.values:
                    rval += _vars(_v)
                return rval
            else:
                return [v]

        def _bind(var, value):
            if isinstance(var, TupleNode):
                # If the value is bound to a TupleNode, we check what
                # flows to the value.
                @self.on_flow_from(value)
                def flow_tuple(track, new_value):
                    if isinstance(new_value, TupleNode):
                        # If a TupleNode flows, we can deconstruct it and
                        # bind each variable to the corresponding
                        # TupleNode element, gaining precision.
                        for v, sub_value in zip(var.values, new_value.values):
                            _bind(v, sub_value)
                    else:
                        # Otherwise, we flow ANY to all variables
                        # in the TupleNode.
                        for v in _vars(var):
                            self.propagate_value(v, ANY)
            else:
                # The single variable case. Quite straightforward.
                self.flow_to(value, var)

        for v, value in node.bindings:
            self.visit(value)
            _visit(v)
            _bind(v, value)
        self.visit(node.body)
        self.flow_to(node.body, node)
        self.run_flows('LetNode', node)

    def visit_Symbol(self, node):
        if node in self.genv.bindings:
            # If the symbol is associated to a global variable (which
            # are constant), we fetch the value and propagate it.
            b = self.genv[node]
            self.visit(b)
            self.propagate_value(node, b)
        elif node in builtins.__dict__.values():
            # We propagate builtin symbols, although it may be better
            # to do this another way.
            self.propagate_value(node, node)
        self.run_flows('Symbol', node)

    def visit_TupleNode(self, node):
        # TupleNodes flow to themselves, which allows us to flow individual
        # elements in deconstructing assignments or constant indexing.
        for v in node.values:
            self.visit(v)
        self.propagate_value(node, node)
        self.run_flows('TupleNode', node)

    def visit_ValueNode(self, node):
        # Values flow to themselves.
        self.propagate_value(node, node)
        self.run_flows('ValueNode', node)


class Track:
    """
    A Track supplements a DFA by defining extra flow rules
    for various node types and/or for primitives.
    """
    def __init__(self, name, dfa, direction='forwards'):
        self.name = name
        self.dfa = dfa
        assert direction in {'forwards', 'backwards', 'bidirectional'}
        self.direction = direction

    def propagate(self, node, value):
        self.dfa.propagate(node, self, value)

    def flow_ApplyNode(self, node):
        pass

    def flow_BeginNode(self, node):
        pass

    def flow_ClosureNode(self, node):
        pass

    def flow_LambdaNode(self, node):
        pass

    def flow_LetNode(self, node):
        pass

    def flow_TupleNode(self, node):
        pass

    def flow_Symbol(self, node):
        pass

    def flow_ValueNode(self, node):
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

    # def flow_TupleNode(self, node):
    #     self.propagate(node, node)

    # def flow_Value(self, node):
    #     self.propagate(node, node)


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
        _type: ((_type, VALUE), _type, _type),
        VALUE: ((), VALUE, VALUE)
    },
    'default': {
    }
}


class NeedsTrack(Track):
    def __init__(self, dfa, autoflow=[]):
        super().__init__('needs', dfa, 'bidirectional')
        self.autoflow = autoflow

    def flow_auto(self, node):
        assert not isinstance(node, ClosureNode)
        for prop in self.autoflow:
            self.propagate(node, prop)

    def flow_ApplyNode(self, node):
        self.propagate(node.fn, VALUE)
        self.flow_auto(node)

    def flow_BeginNode(self, node):
        self.flow_auto(node)

    def flow_ClosureNode(self, node):
        pass

    def flow_LambdaNode(self, node):
        pass

    def flow_LetNode(self, node):
        self.flow_auto(node)

    def flow_TupleNode(self, node):
        self.propagate(node, VALUE)
        self.flow_auto(node)

    def flow_Symbol(self, node):
        self.flow_auto(node)

    def flow_ValueNode(self, node):
        self.flow_auto(node)

    def flow_prim(self, prim, args, node):
        m = needs_map.get(prim, needs_map['default'])

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
