"""Pipeline steps.

The steps are listed in roughly the same order they should be called.
"""

from itertools import count

import numpy as np

from .. import xtype
from ..abstract import (
    ANYTHING,
    AbstractArray,
    AbstractClassBase,
    AbstractDict,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    empty,
    find_aliases,
    typecheck,
)
from ..cconv import closure_convert
from ..compile import BackendValue, load_backend
from ..ir import Graph
from ..opt import (
    CSE,
    DeadDataElimination,
    LocalPassOptimizer,
    NodeMap,
    lib as optlib,
    simplify_types,
    str_to_tag,
    type_to_tag,
)
from ..opt.clean import _strmap_tag
from ..prim import vm_registry
from ..utils import (
    Cons,
    Empty,
    MyiaInputTypeError,
    TaggedValue,
    overload,
    overload_wrapper,
    tracer,
)
from ..validate import (
    validate,
    validate_abstract as default_validate_abstract,
    whitelist as default_whitelist,
)
from ..vm import VM
from .pipeline import PipelineStep, pipeline_function

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

    def step(self, graph, argspec=None, outspec=None):
        """Optimize the graph using the given patterns."""
        counter = count(1)
        changes = True
        while changes:
            with tracer(f'lap{next(counter)}'):
                changes = False
                nn = iter(self.names)
                for opt in self.phases:
                    with tracer(next(nn)):
                        if opt == 'renormalize':
                            assert argspec is not None
                            graph = self.resources.inferrer.renormalize(
                                graph, argspec, outspec
                            )
                        elif opt(graph):
                            changes = True
                if self.run_only_once:
                    break
        with tracer('keep_roots'):
            self.resources.manager.keep_roots(graph)
        res = {'graph': graph}
        return res


#########
# Parse #
#########


@pipeline_function
def step_parse(self, input, argspec=None):
    """Assert that input is a Graph, and set it as the 'graph' key.

    Inputs:
        input: A function.
        argspec: Information about argument types.

    Outputs:
        graph: A graph.
    """
    g = self.resources.convert(input)
    sig = g.make_signature(argspec)
    g = g.generate_graph(sig)
    g = self.resources.convert(g)
    assert type(g) is Graph
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
    new_graph = self.resources.inferrer.monomorphize(inference_context)
    return {'graph': new_graph}


####################
# Erase Class type #
####################


@pipeline_function
def step_simplify_types(self, graph, argspec, outspec):
    """Replace the Class type by Tuple type.

    This should be run on the specialized graph.

    Inputs:
        graph: The graph to prepare.

    Outputs:
        graph: The prepared graph.
    """
    mng = self.resources.manager
    simplify_types(graph, mng)
    new_argspec = tuple(p.abstract for p in graph.parameters)
    graph = self.resources.inferrer.renormalize(graph, new_argspec)
    new_outspec = graph.output.abstract
    return {'graph': graph,
            'orig_argspec': argspec,
            'argspec': new_argspec,
            'orig_outspec': outspec,
            'outspec': new_outspec,
            'simplify_types': True}


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
            optlib.combine_switches_array,

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
            optlib.getitem_setitem_tuple,
            optlib.setitem_tuple,
            optlib.setitem_tuple_ct,
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
            optlib.cancel_env_set_get,
            optlib.getitem_newenv,
            optlib.getitem_env_add,
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
        grad=[
            optlib.expand_J,
        ],
        renormalize='renormalize',
        cse=CSE.partial(report_changes=False),
        jelim=optlib.JElim.partial(),
    )
)


step_opt2 = Optimizer.partial(
    phases=dict(
        renormalize='renormalize',
        dde=DeadDataElimination.partial(),
        main=[
            optlib.unfuse_composite,
            optlib.getitem_tuple,
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
        ],
        cse=CSE.partial(report_changes=False),
    )
)


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
        self.backend = load_backend(backend, backend_options)

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

    def to_backend_value(self, v, t):
        """Returns v."""
        return v

    def from_backend_value(self, v, t):
        """Returns v."""
        return v


class SlowdownWarning(UserWarning):
    """Used to indicate a potential slowdown source."""


#####################################
# Converts args while running model #
#####################################


@overload_wrapper(bootstrap=True)
def convert_arg(fn, self, arg, orig_t):
    if isinstance(arg, BackendValue):
        if not typecheck(orig_t, arg.orig_t):
            raise MyiaInputTypeError("Bad type for backend value.")
        return arg
    return fn(self, arg, orig_t)


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractTuple):
    if not isinstance(arg, tuple):
        raise MyiaInputTypeError('Expected tuple')
    oe = orig_t.elements
    if len(arg) != len(oe):
        raise MyiaInputTypeError(f'Expected {len(oe)} elements')
    return tuple(self(x, o) for x, o in zip(arg, oe))


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractDict):
    if not isinstance(arg, dict):
        raise MyiaInputTypeError('Expected dict')
    types = orig_t.entries
    if len(arg) != len(types):
        raise MyiaInputTypeError(
            "Dictionary input doesn't have the expected size"
        )
    if set(arg.keys()) != set(types.keys()):
        raise MyiaInputTypeError("Mismatched keys for input dictionary.")
    return tuple(self(arg[k], o) for k, o in orig_t.entries.items())


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractClassBase):
    if orig_t.tag is Empty:
        if arg != []:
            raise MyiaInputTypeError(f'Expected empty list')
        return ()
    elif orig_t.tag is Cons:
        if arg == []:
            raise MyiaInputTypeError(f'Expected non-empty list')
        if not isinstance(arg, list):
            raise MyiaInputTypeError(f'Expected list')
        ot = orig_t.attributes['head']
        li = list(self(x, ot) for x in arg)
        rval = TaggedValue(type_to_tag(empty), ())
        for elem in reversed(li):
            rval = TaggedValue(type_to_tag(orig_t), (elem, rval))
        return rval.value
    else:
        if not isinstance(arg, orig_t.tag):
            raise MyiaInputTypeError(f'Expected {orig_t.tag.__qualname__}')
        arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
        oe = list(orig_t.attributes.values())
        res = tuple(self(x, o) for x, o in zip(arg, oe))
        return res


@overload
def convert_arg_array(arg, t: xtype.NDArray, et, orig_t):
    if not isinstance(arg, np.ndarray):
        raise MyiaInputTypeError(f"Expected numpy.ndarray but got {arg}.")
    return arg


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractArray):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.xtype()
    assert issubclass(et, xtype.Number)
    t = orig_t.xtype()
    arg = convert_arg_array[t](arg, t, et, orig_t)
    arg_dtype = xtype.np_dtype_to_type(str(arg.dtype))
    if arg_dtype != et:
        raise MyiaInputTypeError(
            f"Expected array of type {et}, but got {arg_dtype}."
        )
    shp = orig_t.xshape()
    if (shp is not ANYTHING and arg.shape != shp):
        raise MyiaInputTypeError(
            f"Expected array with shape {shp}, but got {arg.shape}."
        )
    return arg


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractUnion):
    for opt in orig_t.options:
        try:
            value = self(arg, opt)
            tag = type_to_tag(opt)
        except TypeError:
            continue
        return TaggedValue(tag, value)
    else:
        opts = ", ".join(map(str, orig_t.options))
        raise MyiaInputTypeError(f'Expected one of {opts}, not {arg}')


@overload  # noqa: F811
def convert_arg(self, arg, orig_t: AbstractScalar):
    t = orig_t.xtype()
    if issubclass(t, xtype.Int):
        if not isinstance(arg, (int, np.integer)):
            raise MyiaInputTypeError(f'Expected int')
    elif issubclass(t, xtype.Float):
        if not isinstance(arg, (float, np.floating)):
            raise MyiaInputTypeError(f'Expected float')
    elif issubclass(t, xtype.Bool):
        if not isinstance(arg, bool):
            raise MyiaInputTypeError(f'Expected bool')
    elif issubclass(t, xtype.Nil):
        if arg is not None:
            raise MyiaInputTypeError(f'Expected None')
    elif issubclass(t, xtype.String):
        if not isinstance(arg, str):
            raise MyiaInputTypeError(f'Expected string')
    else:
        raise MyiaInputTypeError(f'Invalid type: {t}')
    expected_value = orig_t.xvalue()
    if expected_value is not ANYTHING and expected_value != arg:
        raise MyiaInputTypeError(f'Invalid value: {arg}')
    if issubclass(t, xtype.String):
        arg = str_to_tag(arg)
    return arg


@overload(bootstrap=True)
def convert_result(self, res, orig_t, vm_t: AbstractTuple):
    # If the EraseClass opt was applied, orig_t may be Class
    orig_is_class = isinstance(orig_t, AbstractClassBase)
    if orig_is_class:
        if orig_t.tag in (Empty, Cons):
            rval = []
            while res:
                value = self(res[0], orig_t.attributes['head'],
                             vm_t.elements[0])
                rval.append(value)
                res = res[1].value
            return rval
    orig_is_dict = isinstance(orig_t, AbstractDict)
    if orig_is_class:
        oe = orig_t.attributes.values()
    elif orig_is_dict:
        oe = orig_t.entries.values()
    else:
        oe = orig_t.elements
    ve = vm_t.elements
    tup = tuple(self(x, o, v)
                for x, o, v in zip(res, oe, ve))
    if orig_is_class:
        return orig_t.constructor(*tup)
    elif orig_is_dict:
        return dict(zip(orig_t.entries.keys(), tup))
    else:
        return tup


@overload  # noqa: F811
def convert_result(self, arg, orig_t, vm_t: AbstractScalar):
    if orig_t.xtype() == xtype.String:
        arg = _strmap_tag[arg]
    return arg


@overload
def convert_result_array(arg, orig_t: xtype.NDArray):
    return arg


@overload  # noqa: F811
def convert_result(self, arg, orig_t, vm_t: AbstractArray):
    t = orig_t.xtype()
    return convert_result_array[t](arg, t)


@overload  # noqa: F811
def convert_result(self, arg, orig_t, vm_t: AbstractTaggedUnion):
    assert isinstance(orig_t, AbstractUnion)
    for typ in orig_t.options:
        tag = type_to_tag(typ)
        if tag == arg.tag:
            return self(arg.value, typ, vm_t.options.get(tag))
    else:
        raise AssertionError(f'Badly formed TaggedValue')


class Wrap(PipelineStep):
    """Pipeline step to export a callable.

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

    def __init__(self, pipeline_init, return_backend=False):
        """Initialize the Wrap."""
        super().__init__(pipeline_init)
        self.return_backend = return_backend

    def step(self,
             graph,
             output,
             argspec,
             outspec,
             orig_argspec=None,
             orig_outspec=None,
             aliasspec=None,
             simplify_types=False):
        """Convert args to vm format, and output from vm format."""
        if not simplify_types:
            raise AssertionError(
                'OutputWrapper step requires the simplify_types step'
            )
        fn = output
        orig_arg_t = orig_argspec or argspec
        orig_out_t = orig_outspec or outspec
        vm_arg_t = graph.abstract.get_sync()[0].args
        vm_out_t = graph.return_.abstract

        def wrapped(*args):
            if aliasspec:
                alias_tracker, orig_aid_to_paths = aliasspec
                _, aid_to_paths = find_aliases(args, alias_tracker)
                if aid_to_paths != orig_aid_to_paths:
                    raise MyiaInputTypeError('Incompatible aliasing pattern.')
            steps = self.pipeline.steps
            if hasattr(steps, 'compile'):
                backend = steps.compile.backend
            else:
                backend = NumpyChecker()
            if len(args) != len(orig_arg_t):
                raise MyiaInputTypeError('Wrong number of arguments.')
            args = tuple(backend.to_backend_value(convert_arg(arg, ot), vt)
                         for arg, ot, vt in zip(args, orig_arg_t, vm_arg_t))
            res = fn(*args)
            if self.return_backend:
                if isinstance(orig_out_t, AbstractTuple):
                    res = tuple(BackendValue(r, ot, vt, backend)
                                for r, ot, vt in zip(res, orig_out_t.elements,
                                                     vm_out_t.elements))
                else:
                    res = BackendValue(res, orig_out_t, vm_out_t, backend)
            else:
                res = backend.from_backend_value(res, vm_out_t)
                res = convert_result(res, orig_out_t, vm_out_t)
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
