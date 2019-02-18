"""Transforms a graph into lower-level code."""

from tvm import relay

from ..abstract import VALUE
from ..ir import Apply, Graph, Constant, manage, print_graph
from ..graph_utils import toposort
from ..pipeline import PipelineDefinition, PipelineStep
from ..prim import Primitive, ops as P
from ..prim.ops import partial, return_, switch, make_tuple
from ..dtype import type_to_np_dtype

from .relay_helpers import optimize

to_relay_type = type_to_np_dtype

SIMPLE_MAP = {
    P.scalar_add: relay.op.add,
    P.scalar_sub: relay.op.subtract,
    P.scalar_mul: relay.op.multiply,
    P.scalar_div: relay.op.divide,
    P.scalar_mod: relay.op.mod,
    P.scalar_pow: relay.op.power,
    P.scalar_floor: relay.op.floor,
    P.scalar_uadd: lambda x: x,
    P.scalar_usub: relay.op.negative,
    P.scalar_exp: relay.op.exp,
    P.scalar_log: relay.op.log,
    # This is not right tangent vs hyperbolic tangent
    # P.scalar_tan: relay.op.tanh,

    P.scalar_eq: relay.op.equal,
    P.scalar_lt: relay.op.less,
    P.scalar_gt: relay.op.greater,
    P.scalar_ne: relay.op.not_equal,
    P.scalar_le: relay.op.less_equal,
    P.scalar_ge: relay.op.greater_equal,
    # P.bool_and: sym.logical_and,
    # P.bool_or: sym.logical_or
    P.bool_eq: relay.op.equal,

    P.make_tuple: lambda *args: relay.Tuple(args)
}


class RelayMapper:
    """Maps myia operations to relay operations."""

    def __init__(self, simple_map=None, complex_map=None):
        """Create a mapper."""
        self.mapping = {}
        if simple_map is not None:
            self.register_simple(simple_map)
        if complex_map is not None:
            self.register_complex(complex_map)

    def register(self, prim, fn):
        """Register the conversion function for a primitve."""
        assert prim not in self.mapping
        self.mapping[prim] = fn

    def register_simple(self, map):
        """Register simple conversions (1:1 map to nnvm ops)."""
        for k, v in map.items():
            self.register(k, lambda c, *args, v=v: v(*[c.ref(a)
                                                       for a in args]))

    def get(self, fn):
        return self.mapping.get(fn, None)


MAP = RelayMapper(simple_map=SIMPLE_MAP)


def ashape(node):
    """Make sure shape isn't None, that makes relay crash later."""
    shp = node.shape
    if shp is None:
        shp = ()
    return shp


class RelayRunner:
    def __init__(self, fn, dtypes):
        self.fn = fn
        self.dtypes = dtypes

    def tonumpy(self, v):
        if isinstance(v, relay.backend.interpreter.TupleValue):
            return tuple(self.tonumpy(f) for f in v.fields)
        return v.asnumpy()

    def __call__(self, *args):
        assert len(args) == len(self.dtypes)
        args = [relay.const(a, dt) for a, dt in zip(args, self.dtypes)]
        return self.tonumpy(self.fn(*args))


class CompileGraph(PipelineStep):
    """Step to convert a myia graph to a relay graph.

    Inputs:
        graph: A graph

    Outputs:
        output: a wrapped relay graph

    """
    def step(self, graph):
        """Convert the graph into a relay callable."""
        mng = manage(graph)

        function_map = {}
        self.node_map = {}

        print()

        for g in mng.graphs:
            function_map[g.debug.debug_name] = self.convert_func(g)
            print(f"FN: {g.debug.debug_name}")
            print_graph(g)
            print("=======================================")
            print(function_map[g.debug.debug_name].astext())

        module = relay.Module(function_map)
        print(module.astext())

        module.entry_func = module.global_var_map_[graph.debug.debug_name]

        optimize(module)

        print("OPTIMIZED")

        exec = relay.create_executor(mod=module)
        fn = exec.evaluate(module.entry_func)

        output = RelayRunner(
            fn, [to_relay_type(p.type) for p in graph.parameters])

        return {'output': output}

    def on_parameter(self, node):
        tt = to_relay_type(node.type)
        return relay.var(
            node.debug.debug_name,
            shape=ashape(node),
            dtype=to_relay_type(node.type))

    def on_apply(self, node):
        if node.inputs[0].is_constant(Primitive):
            fn = node.inputs[0].value
            conv = MAP.get(fn)
            if conv is not None:
                return conv(self, *node.inputs[1:])
            import pdb; pdb.set_trace()

    def on_constant(self, node):
        import pdb; pdb.set_trace()

    def ref(self, node):
        return self.node_map[node]

    def convert_func(self, graph):
        for p in graph.parameters:
            self.node_map[p] = self.on_parameter(p)

        params = [self.ref(p) for p in graph.parameters]

        # Don't visit functions (inputs[0])
        for node in toposort(graph.output, lambda n: n.inputs[1:]):
            if node.is_apply():
                self.node_map[node] = self.on_apply(node)
            elif node.is_constant():
                self.node_map[node] = self.on_constant(node)

        return relay.Function(params, self.ref(graph.output))


step_compile = CompileGraph.partial()
