"""Pipeline steps.

The steps are listed in roughly the same order they should be called.
"""


import numpy as np

from .. import dtype
from ..cconv import closure_convert
from ..ir import Graph
from ..opt import PatternEquilibriumOptimizer, lib as optlib, CSE, \
    erase_class, erase_tuple
from ..prim import vm_implementations
from ..utils import overload, flatten
from ..validate import validate, whitelist as default_whitelist, \
    validate_type as default_validate_type
from ..vm import VM

from .pipeline import pipeline_function, PipelineStep


#############
# Optimizer #
#############


# Optimizer is used for multiple steps


class Optimizer(PipelineStep):
    """Pipeline step to optimize a graph.

    Inputs:
        graph: The graph to optimize.

    Outputs:
        graph: The optimized graph.
    """

    def __init__(self,
                 pipeline_init,
                 phases,
                 run_only_once=False):
        """Initialize an Optimizer."""
        super().__init__(pipeline_init)
        self.run_only_once = run_only_once
        self.phases = []
        for name, spec in phases.items():
            if spec == 'renormalize':
                pass
            elif isinstance(spec, list):
                spec = PatternEquilibriumOptimizer(*spec, optimizer=self)
            else:
                spec = spec(optimizer=self)
            self.phases.append(spec)

    def step(self, graph, argspec=None, outspec=None):
        """Optimize the graph using the given patterns."""
        changes = True
        while changes:
            changes = False
            for opt in self.phases:
                if opt == 'renormalize':
                    assert argspec is not None
                    graph = self.resources.inferrer.renormalize(
                        graph, argspec, outspec
                    )
                elif opt(graph):
                    changes = True
            if self.run_only_once:
                break
        self.resources.manager.keep_roots(graph)
        return {'graph': graph}


#########
# Parse #
#########


@pipeline_function
def step_parse(self, input):
    """Assert that input is a Graph, and set it as the 'graph' key.

    Inputs:
        input: A function.

    Outputs:
        graph: A graph.
    """
    g = self.resources.convert(input)
    assert isinstance(g, Graph)
    return {'graph': g}


###########
# Resolve #
###########


step_resolve = Optimizer.partial(
    run_only_once=True,
    phases=dict(
        resolve=[optlib.resolve_globals]
    )
)


#########
# Infer #
#########


@pipeline_function
def step_infer(self, graph, argspec):
    """Infer types, shapes, values, etc. for the graph.

    Inputs:
        graph: The graph to infer.
        argspec: Information about argument types.

    Outputs:
        outspec: Inference results for the graph's output.
        inference_context: The Context for the root graph.
    """
    try:
        res, context = self.resources.inferrer.infer(graph, argspec)
        return {'outspec': res,
                'inference_context': context}
    except Exception as exc:
        # We still want to keep the inferrer around even
        # if an error occurred.
        return {'error': exc,
                'error_step': self}


##############
# Specialize #
##############


@pipeline_function
def step_specialize(self, graph, inference_context):
    """Specialize the graph according to argument types.

    Inputs:
        graph: The graph to specialize.
        inference_context: The Context for the root graph.

    Outputs:
        graph: The specialized graph.
    """
    new_graph = self.resources.inferrer.specialize(
        graph, inference_context
    )
    return {'graph': new_graph}


####################
# Erase Class type #
####################


@pipeline_function
def step_erase_class(self, graph, argspec, outspec):
    """Replace the Class type by Tuple type.

    This should be run on the specialized graph.

    Inputs:
        graph: The graph to prepare.

    Outputs:
        graph: The prepared graph.
    """
    mng = self.resources.manager
    erase_class(graph, mng)
    new_argspec = tuple(dict(p.inferred) for p in graph.parameters)
    graph = self.resources.inferrer.renormalize(graph, new_argspec)
    new_outspec = dict(graph.output.inferred)
    return {'graph': graph,
            'orig_argspec': argspec,
            'argspec': new_argspec,
            'orig_outspec': outspec,
            'outspec': new_outspec,
            'erase_class': True}


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

            # Miscellaneous
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
        ],
        grad=[
            optlib.expand_J,
        ],
        renormalize='renormalize',
        cse=CSE.partial(report_changes=False),
        jelim=optlib.JElim.partial(),
    )
)


# Standard optimizations
step_opt = Optimizer.partial(
    phases=dict(
        main=[
            # Branch culling
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            optlib.simplify_switch1,
            optlib.simplify_switch2,
            optlib.simplify_switch_idem,
            optlib.combine_switches,

            # Safe inlining
            optlib.inline_trivial,
            optlib.inline_unique_uses,
            optlib.inline_inside_marked_caller,
            optlib.inline_core,
            optlib.simplify_partial,
            optlib.replace_applicator,

            # Specialization
            optlib.specialize_on_graph_arguments,

            # Arithmetic simplifications
            optlib.multiply_by_one_l,
            optlib.multiply_by_one_r,
            optlib.multiply_by_zero_l,
            optlib.multiply_by_zero_r,
            optlib.add_zero_l,
            optlib.add_zero_r,

            # Array simplifications
            optlib.elim_distribute,
            optlib.elim_array_reduce,

            # Miscellaneous
            optlib.elim_identity,
            optlib.getitem_tuple,
            optlib.setitem_tuple,
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
            optlib.cancel_env_set_get,
            optlib.getitem_newenv,
            optlib.getitem_env_add,
            optlib.simplify_array_map,
        ],
        main2=[
            # Costlier optimizations
            optlib.float_tuple_getitem_through_switch,
            optlib.float_env_getitem_through_switch,
            optlib.incorporate_getitem,
            optlib.incorporate_env_getitem,
            optlib.incorporate_call,
            optlib.incorporate_getitem_through_switch,
            optlib.incorporate_env_getitem_through_switch,
            optlib.incorporate_call_through_switch,
        ],
        grad=[
            optlib.expand_J,
        ],
        renormalize='renormalize',
        cse=CSE.partial(report_changes=False),
        jelim=optlib.JElim.partial(),
    )
)

# Final optimization pass
step_opt2 = Optimizer.partial(
    phases=dict(
        unfuse=[
            optlib.unfuse_composite,
        ],
        main2=[
#            optlib.float_tuple_getitem_through_switch,
            optlib.incorporate_getitem,
            optlib.getitem_tuple,
        ],
    )
)


####################
# Erase Tuple type #
####################


@pipeline_function
def step_erase_tuple(self, graph, argspec, outspec, erase_class=False):
    """Expand Tuple in graph parameters whenever possible.

    This should be run on the specialized graph.

    Inputs:
        graph: The graph to prepare.

    Outputs:
        graph: The prepared graph.
    """
    assert erase_class
    mng = self.resources.manager
    erase_tuple(graph, mng)
    new_argspec = tuple(dict(p.inferred) for p in graph.parameters)
    graph = self.resources.inferrer.renormalize(graph, new_argspec)
    new_outspec = dict(graph.output.inferred)
    return {'graph': graph,
            'argspec': new_argspec,
            'outspec': new_outspec,
            'erase_tuple': True}


############
# Validate #
############


class Validator(PipelineStep):
    """Pipeline step to validate a graph prior to compilation.

    Inputs:
        graph: The graph to validate.

    Outputs:
        None.
    """

    def __init__(self,
                 pipeline_init,
                 whitelist=default_whitelist,
                 validate_type=default_validate_type):
        """Initialize a Validator."""
        super().__init__(pipeline_init)
        self.whitelist = whitelist
        self.validate_type = validate_type

    def step(self, graph, argspec=None, outspec=None):
        """Validate the graph."""
        graph = self.resources.inferrer.renormalize(
            graph, argspec, outspec
        )
        validate(graph,
                 whitelist=self.whitelist,
                 validate_type=self.validate_type)
        return {'graph': graph}


step_validate = Validator.partial()


######################
# Closure conversion #
######################


@pipeline_function
def step_cconv(self, graph):
    """Closure convert the graph.

    Inputs:
        graph: The graph to closure convert.

    Outputs:
        graph: The closure converted graph.
    """
    closure_convert(graph)
    return {'graph': graph}


############################
# Wrap the output function #
############################


@overload
def _convert_arg(arg, orig_t: dtype.Tuple):
    if not isinstance(arg, tuple):
        raise TypeError('Expected tuple')
    oe = orig_t.elements
    if len(arg) != len(oe):
        raise TypeError(f'Expected {len(oe)} elements')
    return list(flatten(convert_arg(x, o)
                        for x, o in zip(arg, oe)))


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.List):
    if not isinstance(arg, list):
        raise TypeError('Expected list')
    ot = orig_t.element_type
    return [list(flatten(convert_arg(x, ot) for x in arg))]


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.Class):
    dc = dtype.tag_to_dataclass[orig_t.tag]
    if not isinstance(arg, dc):
        raise TypeError(f'Expected {dc.__qualname__}')
    arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
    oe = list(orig_t.attributes.values())
    return list(flatten(convert_arg(x, o)
                        for x, o in zip(arg, oe)))


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.Array):
    if not isinstance(arg, np.ndarray):
        raise TypeError('Expected ndarray')
    et = orig_t.elements
    assert dtype.ismyiatype(et, dtype.Number)
    dt = dtype.type_to_np_dtype(et)
    if arg.dtype != dt:
        raise TypeError('Wrong dtype')
    return [arg]


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.Int):
    if not isinstance(arg, int):
        raise TypeError(f'Expected int')
    return [arg]


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.Float):
    if not isinstance(arg, float):
        raise TypeError(f'Expected float')
    return [arg]


@overload  # noqa: F811
def _convert_arg(arg, orig_t: dtype.Bool):
    if not isinstance(arg, bool):
        raise TypeError(f'Expected bool')
    return [arg]


def convert_arg(arg, orig_t):
    """Check that arg matches orig_t, and convert to vm_t."""
    return _convert_arg[orig_t](arg, orig_t)


@overload
def _convert_result(res, orig_t, vm_t: dtype.Class):
    dc = dtype.tag_to_dataclass[orig_t.tag]
    oe = orig_t.attributes.values()
    ve = vm_t.attributes.values()
    tup = tuple(convert_result(getattr(res, attr), o, v)
                for attr, o, v in zip(orig_t.attributes, oe, ve))
    return dc(*tup)


@overload  # noqa: F811
def _convert_result(res, orig_t, vm_t: dtype.List):
    ot = orig_t.element_type
    vt = vm_t.element_type
    return [convert_result(x, ot, vt) for x in res]


@overload  # noqa: F811
def _convert_result(res, orig_t, vm_t: dtype.Tuple):
    # If the EraseClass opt was applied, orig_t may be Class
    orig_is_class = dtype.ismyiatype(orig_t, dtype.Class)
    if orig_is_class:
        oe = orig_t.attributes.values()
    else:
        oe = orig_t.elements
    ve = vm_t.elements
    tup = tuple(convert_result(x, o, v)
                for x, o, v in zip(res, oe, ve))
    if orig_is_class:
        dc = dtype.tag_to_dataclass[orig_t.tag]
        return dc(*tup)
    else:
        return tup


@overload  # noqa: F811
def _convert_result(arg, orig_t,
                    vm_t: (dtype.Int, dtype.Float, dtype.Bool, dtype.Array)):
    return arg


def convert_result(res, orig_t, vm_t):
    """Convert result from vm_t to orig_t."""
    return _convert_result[vm_t](res, orig_t, vm_t)


@pipeline_function
def step_wrap(self,
              graph,
              output,
              argspec,
              outspec,
              orig_argspec=None,
              orig_outspec=None,
              erase_class=False,
              erase_tuple=False):
        """Convert args to vm format, and output from vm format."""
        if not (erase_class and erase_tuple):
            raise AssertionError(
                'OutputWrapper step requires the erase_class/tuple steps'
            )
        fn = output
        orig_arg_t = [arg['type'] for arg in orig_argspec or argspec]
        orig_out_t = (orig_outspec or outspec)['type']
        vm_out_t = graph.type.retval

        def wrapped(*args):
            args = tuple(flatten(convert_arg(arg, ot) for arg, ot in
                                 zip(args, orig_arg_t)))
            res = fn(*args)
            res = convert_result(res, orig_out_t, vm_out_t)
            return res
        return {'output': wrapped}


################
# Debug export #
################


class DebugVMExporter(PipelineStep):
    """Pipeline step to export a callable.

    Inputs:
        graph: The graph to wrap into a callable.

    Outputs:
        output: The callable.
    """

    def __init__(self, pipeline_init, implementations):
        """Initialize an DebugVMExporter."""
        super().__init__(pipeline_init)
        self.vm = VM(self.pipeline.resources.convert,
                     self.pipeline.resources.manager,
                     self.pipeline.resources.py_implementations,
                     implementations)

    def step(self, graph):
        """Make a Python callable out of the graph."""
        return {'output': self.vm.export(graph)}


step_debug_export = DebugVMExporter.partial(
    implementations=vm_implementations
)
