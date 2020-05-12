"""Pipeline steps.

The steps are listed in roughly the same order they should be called.
"""

from itertools import count

from ..abstract import AbstractTuple, find_aliases, nobottom, type_to_abstract
from ..compile import BackendValue
from ..ir import Graph, clone
from ..opt import (
    CSE,
    DeadDataElimination,
    GraphInterfaceRewriterOpt,
    LocalPassOptimizer,
    NodeMap,
    RemoveUnusedParameters,
    lambda_lift,
    lib as optlib,
)
from ..parser import parse
from ..simplify_types import from_canonical, simplify_types, to_canonical
from ..utils import (
    InferenceError,
    MyiaInputTypeError,
    Partializable,
    new_universe,
    tracer,
)
from ..validate import ValidationError
from ..xtype import UniverseType

#############
# Optimizer #
#############


# Optimizer is used for multiple steps


class Optimizer(Partializable):
    """Pipeline step to optimize a graph.

    Inputs:
        graph: The graph to optimize.

    Outputs:
        graph: The optimized graph.
    """

    def __init__(self, phases, run_only_once=False, use_tracker=True):
        """Initialize an Optimizer."""
        self.run_only_once = run_only_once
        self.phases = phases
        self.use_tracker = use_tracker

    def __call__(self, resources, graph, argspec=None, outspec=None):
        """Optimize the graph using the given patterns."""
        if self.use_tracker:
            resources.tracker.activate()
        final_phases = []
        names = []
        for name, spec in self.phases.items():
            if isinstance(spec, list):
                nmap = NodeMap()
                for opt in spec:
                    nmap.register(getattr(opt, "interest", None), opt)
                spec = LocalPassOptimizer(nmap, resources=resources)
            else:
                spec = spec(resources=resources)
            names.append(name)
            final_phases.append(spec)

        if len(final_phases) == 1:
            run_only_once = True
        else:
            run_only_once = self.run_only_once

        counter = count(1)
        changes = True
        while changes:
            with tracer(f"lap{next(counter)}"):
                changes = False
                nn = iter(names)
                for opt in final_phases:
                    with tracer(next(nn)):
                        if opt(graph):
                            changes = True
                if run_only_once:
                    break
        res = {"graph": graph}
        return res


#########
# Parse #
#########


def step_parse(resources, input, argspec=None):
    """Assert that input is a Graph, and set it as the 'graph' key.

    Inputs:
        input: A function.
        argspec: Information about argument types.

    Outputs:
        graph: A graph.
    """
    if callable(input):
        g = parse(input, use_universe=resources.universal)
    else:
        g = resources.convert(input)
    sig = g.make_signature(argspec)
    g = g.generate_graph(sig)
    g = resources.convert(g)
    assert type(g) is Graph
    return {"graph": g}


###########
# Resolve #
###########


def step_resolve(resources, graph, argspec=None, outspec=None):
    """Resolve all global symbols in the graph.

    This will remove all resolve(...) calls in the graph.

    This step is unnecessary if the infer/specialize steps are in the pipeline.
    """
    new_graph = clone(graph, total=True)
    resources.opt_manager.add_graph(new_graph, root=True)
    resources.infer_manager = resources.opt_manager
    opt = Optimizer(
        run_only_once=True,
        phases=dict(resolve=[optlib.resolve_globals]),
        use_tracker=False,
    )
    return opt(
        resources=resources, graph=new_graph, argspec=argspec, outspec=outspec
    )


#########
# Infer #
#########


def step_infer(resources, graph, argspec):
    """Infer types, shapes, values, etc. for the graph.

    Inputs:
        graph: The graph to infer.
        argspec: Information about argument types.

    Outputs:
        outspec: Inference results for the graph's output.
        inference_context: The Context for the root graph.
    """
    orig_argspec = argspec
    if resources.universal:
        argspec = (*argspec, type_to_abstract(UniverseType))
    outspec, context = resources.inferrer(graph, argspec)
    if not nobottom(outspec):
        raise InferenceError(
            "There is no condition in which the program succeeds"
        )
    orig_outspec = outspec
    if resources.universal:
        orig_outspec = outspec.elements[1]
    return {
        "outspec": outspec,
        "argspec": argspec,
        "orig_argspec": orig_argspec,
        "orig_outspec": orig_outspec,
        "inference_context": context,
    }


##############
# Specialize #
##############


def step_specialize(resources, graph, inference_context):
    """Specialize the graph according to argument types.

    Inputs:
        graph: The graph to specialize.
        inference_context: The Context for the root graph.

    Outputs:
        graph: The specialized graph.
    """
    new_graph = resources.monomorphizer(inference_context)
    resources.opt_manager.keep_roots(new_graph)
    return {"graph": new_graph}


####################
# Erase Class type #
####################


def step_simplify_types(resources, graph, argspec, outspec):
    """Replace the Class type by Tuple type.

    This should be run on the specialized graph.

    Inputs:
        graph: The graph to prepare.

    Outputs:
        graph: The prepared graph.
    """
    resources.tracker.activate()
    mng = resources.opt_manager
    simplify_types(graph, mng)
    new_argspec = tuple(p.abstract for p in graph.parameters)
    resources.live_inferrer()
    new_outspec = graph.output.abstract
    return {
        "graph": graph,
        "argspec": new_argspec,
        "outspec": new_outspec,
        "simplify_types": True,
    }


############
# Optimize #
############


# For debugging purposes, less optimizations
step_debug_opt = Optimizer.partial(
    phases=dict(
        main=[
            # Branch culling
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            # Safe inlining
            optlib.inline_core,
            optlib.simplify_partial,
            optlib.elim_identity,
            # Miscellaneous
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
            optlib.replace_Jinv_on_graph,
        ],
        grad=[optlib.expand_J],
        cse=CSE.partial(report_changes=False),
        jelim=optlib.JElim.partial(),
    )
)


# Standard optimizations
step_opt = Optimizer.partial(
    phases=dict(
        main=[
            # Force constants
            optlib.force_constants,
            # Branch culling
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            optlib.simplify_switch1,
            optlib.simplify_switch2,
            optlib.simplify_switch_idem,
            optlib.combine_switches,
            optlib.combine_switches_array,
            # Safe inlining
            optlib.inline_trivial,
            optlib.inline_unique_uses,
            optlib.inline_inside_marked_caller,
            optlib.inline_core,
            optlib.simplify_partial,
            optlib.replace_applicator,
            # # Specialization
            # optlib.specialize_on_graph_arguments,
            # Arithmetic simplifications
            optlib.multiply_by_one_l,
            optlib.multiply_by_one_r,
            optlib.multiply_by_zero_l,
            optlib.multiply_by_zero_r,
            optlib.add_zero_l,
            optlib.add_zero_r,
            optlib.not_eq,
            optlib.multiply_by_one_l_map,
            optlib.multiply_by_one_r_map,
            optlib.multiply_by_zero_l_map,
            optlib.multiply_by_zero_r_map,
            optlib.add_zero_l_map,
            optlib.add_zero_r_map,
            optlib.usub_cancel_map,
            optlib.usub_sink_mul_l_map,
            optlib.usub_sink_mul_r_map,
            optlib.usub_sink_div_l_map,
            optlib.usub_sink_div_r_map,
            optlib.add_usub_map,
            optlib.sub_usub_map,
            optlib.not_eq_map,
            # Array simplifications
            optlib.elim_distribute,
            optlib.elim_array_reduce,
            optlib.merge_transposes,
            optlib.elim_transpose,
            # Miscellaneous
            optlib.elim_identity,
            optlib.getitem_tuple,
            optlib.getitem_constant_tuple,
            optlib.getitem_setitem_tuple,
            optlib.setitem_tuple,
            optlib.setitem_tuple_ct,
            optlib.cancel_tuple_reconstruction,
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
            optlib.replace_Jinv_on_graph,
            optlib.cancel_env_set_get,
            optlib.getitem_newenv,
            # TODO: reintegrate the getitem_env_add optimization
            # optlib.getitem_env_add,
            optlib.simplify_array_map,
            optlib.gadd_zero_l,
            optlib.gadd_zero_r,
            optlib.gadd_switch,
            optlib.incorporate_call_through_switch,
        ],
        main2=[
            # Costlier optimizations
            optlib.float_tuple_getitem_through_switch,
            optlib.float_env_getitem_through_switch,
            # We may reactivate those later, but they are slow
            # optlib.incorporate_getitem,
            # optlib.incorporate_env_getitem,
            # optlib.incorporate_call,
            # optlib.incorporate_getitem_through_switch,
            # optlib.incorporate_env_getitem_through_switch,
            # optlib.incorporate_call_through_switch,
        ],
        grad=[optlib.expand_J],
        cse=CSE.partial(report_changes=False),
        jelim=optlib.JElim.partial(),
    )
)


step_opt2 = Optimizer.partial(
    phases=dict(
        rmunused=GraphInterfaceRewriterOpt.partial(
            rewriter=RemoveUnusedParameters
        ),
        dde=DeadDataElimination.partial(),
        main=[
            optlib.force_constants,
            optlib.unfuse_composite,
            optlib.getitem_tuple,
            optlib.getitem_constant_tuple,
            optlib.getitem_setitem_tuple,
            optlib.setitem_tuple,
            optlib.setitem_tuple_ct,
            optlib.float_tuple_getitem_through_switch,
            optlib.inline_trivial,
            optlib.inline_unique_uses,
            optlib.inline_inside_marked_caller,
            optlib.inline_core,
            optlib.combine_switches_array,
            optlib.gadd_zero_l,
            optlib.gadd_zero_r,
            optlib.gadd_switch,
            optlib.setitem_dead,
            optlib.elim_stop_gradient,
        ],
        cse=CSE.partial(report_changes=False),
    )
)


##################
# Lambda lifting #
##################


def step_llift(resources, graph, outspec=None):
    """Pipeline step to lambda-lift the graph.

    Inputs:
        graph: The graph to lambda-lift.

    Outputs:
        None.
    """
    lambda_lift(graph)
    return {"graph": graph}


############
# Validate #
############


def step_validate(resources, graph, outspec=None):
    """Pipeline step to validate a graph prior to compilation.

    Inputs:
        graph: The graph to validate.

    Outputs:
        None.
    """
    if graph.output.abstract != outspec:
        raise ValidationError(
            "The output type of the graph changed during optimization"
            f" from {outspec} to {graph.output.abstract}"
        )
    resources.validator(graph)
    return {"graph": graph}


###############
# Compilation #
###############


def step_compile(resources, graph, argspec, outspec):
    """Compile the set of graphs.

    Inputs:
        graph: a graph (must be typed)
        argspec: The argument types
        outspec: The output type

    Outputs:
        output: a callable
    """
    out = resources.backend.compile(graph, argspec, outspec)
    return {"output": out}


#####################################
# Converts args while running model #
#####################################


def _to_backend(arg, backend, vt):
    if isinstance(arg, BackendValue):
        if arg.backend is not backend:
            raise ValueError("Value from wrong backend")  # pragma: no cover
        return arg.value
    else:
        return backend.to_backend_value(arg, vt)


def step_wrap(
    resources,
    graph,
    output,
    argspec,
    outspec,
    orig_argspec=None,
    orig_outspec=None,
    aliasspec=None,
    simplify_types=False,
):
    """Pipeline step to export a callable.

    Convert args to vm format, and output from vm format.

    Inputs:
        graph: The graph to wrap into a callable.
        output: callable
        argspec: types of inputs
        outspec: types of outputs
        orig_argspec: initial argspec
        orig_outspec: intial outspec
        simplify_types: boolean marker

    Outputs:
        output: wrapped callable.
    """
    if not simplify_types:
        raise AssertionError(
            "OutputWrapper step requires the simplify_types step"
        )
    fn = output
    orig_arg_t = argspec if orig_argspec is None else orig_argspec
    orig_out_t = outspec if orig_outspec is None else orig_outspec
    vm_arg_t = graph.abstract.args
    vm_out_t = graph.return_.abstract
    if resources.universal:
        vm_unv_in_t, vm_arg_t = vm_arg_t[-1], vm_arg_t[:-1]
        _, vm_out_t = vm_out_t.elements[0], vm_out_t.elements[1]

    def wrapped(*args):
        if aliasspec:
            alias_tracker, orig_aid_to_paths = aliasspec
            _, aid_to_paths = find_aliases(args, alias_tracker)
            if aid_to_paths != orig_aid_to_paths:
                raise MyiaInputTypeError("Incompatible aliasing pattern.")
        backend = resources.backend.backend
        if len(args) != len(orig_arg_t):
            raise MyiaInputTypeError("Wrong number of arguments.")
        args = tuple(
            _to_backend(to_canonical(arg, ot), backend, vt)
            for arg, ot, vt in zip(args, orig_arg_t, vm_arg_t)
        )
        if resources.universal:
            backend_universe = backend.to_backend_value(
                to_canonical(new_universe, argspec[-1]), vm_unv_in_t
            )
            _, res = fn(*args, backend_universe)
        else:
            res = fn(*args)
        if resources.return_backend:
            if isinstance(orig_out_t, AbstractTuple):
                res = tuple(
                    BackendValue(r, ot, vt, backend)
                    for r, ot, vt in zip(
                        res, orig_out_t.elements, vm_out_t.elements
                    )
                )
            else:
                res = BackendValue(res, orig_out_t, vm_out_t, backend)
        else:
            res = backend.from_backend_value(res, vm_out_t)
            res = from_canonical(res, orig_out_t)
        return res

    return {"output": wrapped}


################
# Debug export #
################


def step_debug_export(resources, graph):
    """Make a Python callable out of the graph."""
    return {"output": resources.debug_vm.vm.export(graph)}


__consolidate__ = True
__all__ = ["Optimizer"]
