"""Pipeline steps.

The steps are listed in roughly the same order they should be called.
"""


import numpy as np
from itertools import count

from .. import dtype
from ..abstract import AbstractTuple, AbstractList, AbstractClass, \
    AbstractArray, TYPE, AbstractScalar, AbstractUnion
from ..cconv import closure_convert
from ..ir import Graph
from ..opt import lib as optlib, CSE, erase_class, erase_tuple, NodeMap, \
    LocalPassOptimizer
from ..prim import vm_registry
from ..utils import overload, flatten, no_prof
from ..validate import validate, whitelist as default_whitelist, \
    validate_abstract as default_validate_abstract
from ..vm import VM
from ..compile import load_backend

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
        self.names = []
        for name, spec in phases.items():
            if spec == 'renormalize':
                pass
            elif isinstance(spec, list):
                nmap = NodeMap()
                for opt in spec:
                    nmap.register(getattr(opt, 'interest', None), opt)
                spec = LocalPassOptimizer(nmap, optimizer=self)
            else:
                spec = spec(optimizer=self)
            self.names.append(name)
            self.phases.append(spec)

        if len(self.phases) == 1:
            self.run_only_once = True

    def step(self, graph, argspec=None, outspec=None, profile=no_prof):
        """Optimize the graph using the given patterns."""
        with profile:
            counter = count(1)
            changes = True
            while changes:
                with profile.lap(next(counter)):
                    changes = False
                    nn = iter(self.names)
                    for opt in self.phases:
                        with profile.step(next(nn)):
                            if opt == 'renormalize':
                                assert argspec is not None
                                graph = self.resources.inferrer.renormalize(
                                    graph, argspec, outspec
                                )
                            elif opt(graph):
                                changes = True
                    if self.run_only_once:
                        break
            with profile.step('keep_roots'):
                self.resources.manager.keep_roots(graph)
            res = {'graph': graph}
            return res


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
                'argspec': argspec,
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
    new_argspec = tuple(p.abstract for p in graph.parameters)
    graph = self.resources.inferrer.renormalize(graph, new_argspec)
    new_outspec = graph.output.abstract
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
            optlib.elim_identity,

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

            # Array simplifications
            optlib.elim_distribute,
            optlib.elim_array_reduce,
            optlib.merge_transposes,
            optlib.elim_transpose,

            # Miscellaneous
            optlib.elim_identity,
            optlib.getitem_tuple,
            optlib.getitem_setitem_tuple,
            optlib.setitem_tuple,
            optlib.setitem_tuple_ct,
            optlib.getitem_setitem_list,
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
        main=[
            optlib.unfuse_composite,
            optlib.getitem_tuple,
            optlib.getitem_setitem_tuple,
            optlib.setitem_tuple,
            optlib.setitem_tuple_ct,
            optlib.getitem_setitem_list,
            optlib.float_tuple_getitem_through_switch,
            optlib.incorporate_getitem,
            optlib.incorporate_getitem_through_switch,
            optlib.inline_trivial,
            optlib.inline_unique_uses,
            optlib.inline_inside_marked_caller,
            optlib.inline_core,
        ],
    )
)


####################
# Erase Tuple type #
####################


@pipeline_function
def step_erase_tuple(self, graph, argspec, erase_class=False):
    """Expand tuples in the main graph parameters.

    Inputs:
        graph: The graph to transform
        argspec: The argument types.
        erase_tuple: Must be True

    Outputs:
        graph: The prepared graph.
        argspec: The transformed argspec
        erase_tuple: True

    """
    assert erase_class
    mng = self.resources.manager
    new_graph = erase_tuple(graph, mng)
    if new_graph != graph:
        new_argspec = tuple(p.abstract for p in new_graph.parameters)
        new_graph = self.resources.inferrer.renormalize(new_graph, new_argspec)
        return {'graph': new_graph,
                'argspec': new_argspec,
                'erase_tuple': True}
    else:
        return {'graph': graph,
                'argspec': argspec,
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
                 validate_abstract=default_validate_abstract):
        """Initialize a Validator."""
        super().__init__(pipeline_init)
        self.whitelist = whitelist
        self.validate_abstract = validate_abstract

    def step(self, graph, argspec=None, outspec=None):
        """Validate the graph."""
        graph = self.resources.inferrer.renormalize(
            graph, argspec, outspec
        )
        validate(graph,
                 whitelist=self.whitelist,
                 validate_abstract=self.validate_abstract)
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


###############
# Compilation #
###############


class CompileStep(PipelineStep):
    """Step to compile a graph to a configurable backend.

    Inputs:
        graph: a graph (must be typed)

    Outputs:
        output: a callable

    """

    def __init__(self, pipeline_init, backend=None, backend_options=None):
        """Initialize a CompileStep.

        Arguments:
            backend: (str) the name of the backend to use
            backend_options: (dict) options for the backend

        """
        super().__init__(pipeline_init)
        backend_class = load_backend(backend)
        if backend_options is None:
            backend_options = {}
        self.backend = backend_class(**backend_options)

    def step(self, graph, argspec, outspec):
        """Compile the set of graphs."""
        out = self.backend.compile(graph, argspec, outspec, self.pipeline)
        return {'output': out}


step_compile = CompileStep.partial()


############################
# Wrap the output function #
############################

class NumpyChecker:
    """Dummy backend used for debug mode."""

    def from_numpy(self, n):
        """Returns n."""
        return n

    def to_numpy(self, n):
        """Returns n."""
        return n

    def from_scalar(self, s, dt):
        """Returns s."""
        return s

    def to_scalar(self, s):
        """Returns s."""
        return s

    def check_array(self, arg, t):
        """Checks that arg has elements of the right dtype."""
        if not isinstance(arg, np.ndarray):
            raise TypeError('Expected ndarray')
        if arg.dtype != dtype.type_to_np_dtype(t):
            raise TypeError('Wrong dtype')
        return arg


class SlowdownWarning(UserWarning):
    """Used to indicate a potential slowdown source."""


@overload(bootstrap=True)
def convert_arg(self, arg, orig_t: AbstractTuple, backend):
    if not isinstance(arg, tuple):
        raise TypeError('Expected tuple')
    oe = orig_t.elements
    if len(arg) != len(oe):
        raise TypeError(f'Expected {len(oe)} elements')
    return list(flatten(self(x, o, backend)
                        for x, o in zip(arg, oe)))


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractList, backend):
    if not isinstance(arg, list):
        raise TypeError('Expected list')
    ot = orig_t.element
    return [list(flatten(self(x, ot, backend) for x in arg))]


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractClass, backend):
    dc = dtype.tag_to_dataclass[orig_t.tag]
    if not isinstance(arg, dc):
        raise TypeError(f'Expected {dc.__qualname__}')
    arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
    oe = list(orig_t.attributes.values())
    return list(flatten(self(x, o, backend)
                        for x, o in zip(arg, oe)))


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractArray, backend):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.values[TYPE]
    assert dtype.ismyiatype(et, dtype.Number)
    if isinstance(arg, np.ndarray):
        arg = backend.from_numpy(arg)
    backend.check_array(arg, et)
    return [arg]


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractUnion, backend):
    for opt in orig_t.options:
        try:
            return self(arg, opt, backend)
        except TypeError:
            continue
    else:
        opts = ", ".join(map(str, orig_t.options))
        raise TypeError(f'Expected one of {opts}')


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractScalar, backend):
    t = orig_t.values[TYPE]
    if dtype.ismyiatype(t, dtype.Int):
        if not isinstance(arg, int):
            raise TypeError(f'Expected int')
    elif dtype.ismyiatype(t, dtype.Float):
        if not isinstance(arg, float):
            raise TypeError(f'Expected float')
    elif dtype.ismyiatype(t, dtype.Bool):
        if not isinstance(arg, bool):
            raise TypeError(f'Expected bool')
    else:
        raise TypeError(f'Invalid type: {t}')
    arg = backend.from_scalar(arg, t)
    return [arg]


@overload(bootstrap=True)
def convert_result(self, res, orig_t, vm_t: AbstractClass, backend):
    dc = dtype.tag_to_dataclass[orig_t.tag]
    oe = orig_t.attributes.values()
    ve = vm_t.attributes.values()
    tup = tuple(self(getattr(res, attr), o, v, backend)
                for attr, o, v in zip(orig_t.attributes, oe, ve))
    return dc(*tup)


@overload  # noqa: F811
def convert_result(self, res, orig_t, vm_t: AbstractList, backend):
    ot = orig_t.element
    vt = vm_t.element
    return [self(x, ot, vt, backend) for x in res]


@overload  # noqa: F811
def convert_result(self, res, orig_t, vm_t: AbstractTuple, backend):
    # If the EraseClass opt was applied, orig_t may be Class
    orig_is_class = isinstance(orig_t, AbstractClass)
    if orig_is_class:
        oe = orig_t.attributes.values()
    else:
        oe = orig_t.elements
    ve = vm_t.elements
    tup = tuple(self(x, o, v, backend)
                for x, o, v in zip(res, oe, ve))
    if orig_is_class:
        dc = dtype.tag_to_dataclass[orig_t.tag]
        return dc(*tup)
    else:
        return tup


@overload  # noqa: F811
def convert_result(self, arg, orig_t, vm_t: AbstractScalar, backend):
    return backend.to_scalar(arg)


@overload  # noqa: F811
def convert_result(self, arg, orig_t, vm_t: AbstractArray, backend):
    return backend.to_numpy(arg)


class Wrap(PipelineStep):
    def __init__(self, pipeline_init, return_backend=False):
        super().__init__(pipeline_init)
        self.return_backend = return_backend

    def step(self,
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
        orig_arg_t = orig_argspec or argspec
        orig_out_t = orig_outspec or outspec
        vm_out_t = graph.return_.abstract

        def wrapped(*args):
            steps = self.pipeline.steps
            if hasattr(steps, 'compile'):
                backend = steps.compile.backend
            else:
                backend = NumpyChecker()
            args = tuple(flatten(convert_arg(arg, ot, backend) for arg, ot in
                                 zip(args, orig_arg_t)))
            res = fn(*args)
            if self.return_backend:
                backend = NumpyChecker()
            res = convert_result(res, orig_out_t, vm_out_t, backend)
            return res

        return {'output': wrapped}


step_wrap = Wrap.partial()


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
    implementations=vm_registry
)
