"""Myia Python backend.

Generate and compile a Python code string from graphs.
"""
import operator
import re
import sys
from collections import Counter
from types import ModuleType

from ovld import ovld

from myia import operations
from myia.abstract import to_abstract
from myia.compile.backends import Backend, Converter
from myia.compile.transform import convert_grad, get_prim_graph
from myia.debug.label import NodeLabeler
from myia.graph_utils import toposort
from myia.ir import Graph, manage
from myia.lib import ANYTHING, AbstractArray, AbstractHandle, AbstractTuple
from myia.operations import Operation, Primitive, primitives as P
from myia.xtype import Dict, Tuple, type_to_np_dtype


class ConstantString:
    """Helper class to represent a string to be treated as a constant.

    Constant strings will be inlined and replace node names in final code.
    """

    def __init__(self, value):
        """Initialize."""
        self.value = value

    def __str__(self):
        return self.value


def python_array_map(c, fn, *arrays):
    """Implementation for primitive array_map."""
    assert fn.is_constant(Primitive)
    return f"np.vectorize({c.ref(fn)})({', '.join(c.ref(a) for a in arrays)})"


def python_scalar_to_array(c, x, t):
    """Implementation for primitive scalar_to_array."""
    assert t.is_constant(AbstractArray)
    if t.value.element is ANYTHING or t.value.element.xtype() is ANYTHING:
        return f"np.array({c.ref(x)})"
    dtype = type_to_np_dtype(t.value.element.xtype())
    return f"np.array({c.ref(x)}, dtype='{dtype}')"


def python_array_cast(c, x, t):
    """Implementation for primitive array_cast."""
    assert t.is_constant()
    dtype = type_to_np_dtype(t.value.xtype())
    return f"{c.ref(x)}.astype('{dtype}')"


def python_scalar_cast(c, x, t):
    """Implementation for primitive scalar_cast."""
    assert t.is_constant()
    dtype = type_to_np_dtype(t.value.xtype())
    return f"np.{dtype}({c.ref(x)})"


def python_tuple_setitem(c, data, item, value):
    """Implementation for primitive setitem."""
    return f"tuple({c.ref(value)} if i == {c.ref(item)} else x for i, x in enumerate({c.ref(data)}))"


def python_unsafe_static_cast(c, x, t):
    """Implementation for primitive unsafe_static_cast."""
    return c.ref(x)


def python_random_uint32(c, rstate, shape):
    """Implementation for primitive random_uint32."""
    name_high = c.get_new_name("high")
    name_value = c.get_new_name("value")
    return [
        f"{name_high} = np.uint64(np.iinfo(np.uint32).max) + np.uint64(1)",
        f"{name_value} = {c.ref(rstate)}.state.randint(low=0, high={name_high}, size={c.ref(shape)}, dtype='uint64')",
        f"{c.ref(rstate)}, {name_value}.astype(np.uint32)",
    ]


def python_make_tuple(c, *args):
    """Implementation for primitive make_tuple."""
    return f"({', '.join(c.ref(arg) for arg in args)},)"


def python_switch(c, cond, x, y):
    """Implementation for primitive switch."""
    return f"{c.ref(x)} if {c.ref(cond)} else {c.ref(y)}"


def python_scalar_div(c, x, y):
    """Implementation for primitive scalar_div."""
    return f"type({c.ref(x)})({c.ref(x)} / {c.ref(y)})"


def python_env_setitem(c, env, key, x):
    """Implementation for primitive env_setitem."""
    return [f"{c.ref(env)}[{c.ref(key)}] = {c.ref(x)}", c.ref(env)]


def python_tagged(c, x, tag):
    """Implementation for primitive tagged."""
    return f"{c.ref(x)} if {c.ref(tag)} is None else TaggedValue({c.ref(tag)}, {c.ref(x)})"


def python_array_getitem(c, data, begin, end, strides):
    """Implementation for primitive array_getitem."""
    idx = c.get_new_name("idx")
    return [
        f"{idx} = tuple(slice(b, e, s) for b, e, s in zip({c.ref(begin)}, {c.ref(end)}, {c.ref(strides)}))",
        f"{c.ref(data)}[{idx}]",
    ]


def python_array_setitem(c, data, begin, end, strides, value):
    """Implementation for primitive array_setitem."""
    idx = c.get_new_name("idx")
    data2 = c.get_new_name("data2")
    return [
        f"{idx} = tuple(slice(b, e, s) for b, e, s in zip({c.ref(begin)}, {c.ref(end)}, {c.ref(strides)}))",
        f"{data2} = {c.ref(data)}.copy()",
        f"{data2}[{idx}] = {c.ref(value)}",
        f"{data2}",
    ]


def python_split(c, x, sections, dim):
    """Implementation for primitive split."""
    x = c.ref(x)
    sections = c.ref(sections)
    dim = c.ref(dim)
    return [
        f"{sections} = tuple(np.cumsum({sections}))[:-1]",
        f"np.split({x}, {sections}, axis={dim})",
    ]


def python_take_grad_inp(c, nb_indices, indices, values):
    """Implementation for primitive take_grad_inp."""
    nb_indices = c.ref(nb_indices)
    indices = c.ref(indices)
    values = c.ref(values)
    return f"IMPL.take_grad_inp({nb_indices}, {indices}, {values})"


def python_gather(c, x, dim, index):
    """Implementation for primitiv gather."""
    x = c.ref(x)
    dim = c.ref(dim)
    index = c.ref(index)
    return f"np.take_along_axis({x}, {index}, {dim})"


def python_max_pool2d(
    c, inp, kernel_size, stride, padding, dilation, ceil_mode
):
    """Implementation for primitive max_pool2d."""
    assert dilation.is_constant(tuple) and dilation.value == (1, 1)
    inp = c.ref(inp)
    kernel_size = c.ref(kernel_size)
    stride = c.ref(stride)
    padding = c.ref(padding)
    ceil_mode = c.ref(ceil_mode)
    return f"IMPL.max_pool2d({inp}, {kernel_size}, {stride}, {padding}, {ceil_mode})"


def python_max_pool2d_grad(
    c, inp, kernel_size, stride, padding, dilation, ceil_mode, dout
):
    """Implementation for primitive max_pool2d_grad."""
    assert dilation.is_constant(tuple) and dilation.value == (1, 1)
    inp = c.ref(inp)
    kernel_size = c.ref(kernel_size)
    stride = c.ref(stride)
    padding = c.ref(padding)
    ceil_mode = c.ref(ceil_mode)
    dout = c.ref(dout)
    return f"IMPL.max_pool2d_grad({inp}, {kernel_size}, {stride}, {padding}, {ceil_mode}, {dout})"


def python_make_handle(c, typ, universe):
    """Implement primitive `make_handle`."""
    universe = c.ref(universe)
    handle = c.get_new_name("handle")
    return [f"{handle} = HandleInstance(None, None)", f"({universe}, {handle})"]


def python_universe_setitem(c, universe, handle, value):
    """Implement `universe_setitem`."""
    universe = c.ref(universe)
    handle = c.ref(handle)
    value = c.ref(value)
    new_universe = c.get_new_name("new_universe")
    return [
        f"{new_universe} = {universe}.set({handle}, {value})",
        f"{new_universe}.commit()",
        f"{new_universe}",
    ]


def python_universe_getitem(c, universe, handle):
    """Implement `universe_getitem`."""
    universe = c.ref(universe)
    handle = c.ref(handle)
    return f"{universe}.get({handle})"


def op_grad(f, *required_args):
    """Implement operation grad using finite difference."""
    from myia.debug.finite_diff import GradTester
    import inspect

    argnames = inspect.getfullargspec(f).args

    def gradient(*args, dout=1):
        gt = GradTester(f, None, args, argnames)
        ret = gt.compute_finite_diff()
        out = []
        for i in range(len(args)):
            argname = argnames[i]
            if not required_args or argname in required_args:
                k = f"d{f.__name__}/d{argname}"
                dx = ret[k]
                out.append(dx * type(dx)(dout))
        return out[0] if len(out) == 1 else out

    return gradient


def op_value_and_grad(f, *required_args):
    """Implement operation value_and_grad."""
    gradient = op_grad(f, *required_args)

    def v_and_g(*args, dout=1):
        return f(*args), gradient(*args, dout=dout)

    return v_and_g


def op_make_dict(dct: dict, *values):
    """Implement operation make_dict.

    dct must be a dict with keys to map to given values.
    """
    return {key: val for key, val in zip(dct.keys(), values)}


@ovld  # noqa: F811
def myia_iter(obj: object):
    return obj.__myia_iter__()


@ovld  # noqa: F811
def myia_iter(obj: range):
    return obj


@ovld  # noqa: F811
def myia_iter(obj: tuple):
    return obj


@ovld  # noqa: F811
def myia_hasnext(obj: object):
    return obj.__myia_hasnext__()


@ovld  # noqa: F811
def myia_hasnext(obj: range):
    return obj.start < obj.stop


@ovld  # noqa: F811
def myia_hasnext(obj: tuple):
    return bool(obj)


@ovld  # noqa: F811
def myia_next(obj: object):
    return obj.__myia_next__()


@ovld  # noqa: F811
def myia_next(obj: range):
    return obj.start, range(obj.start + obj.step, obj.stop, obj.step)


@ovld  # noqa: F811
def myia_next(obj: tuple):
    return obj[0], obj[1:]


SIMPLE_MAP = {
    P.argmax: f"IMPL.argmax(%s, %s)",
    P.array_max: "np.array(np.max(%s, %s))",
    P.array_reduce: "IMPL.array_reduce(%s, %s, %s)",
    P.array_to_scalar: "%s.item()",
    P.bool_and: "%s and %s",
    P.bool_eq: "%s == %s",
    P.bool_not: "not %s",
    P.bool_or: "%s or %s",
    P.casttag: "%s.cast(%s)",
    P.concat: "np.concatenate(%s, axis=%s)",
    P.conv2d: "IMPL.conv2d(%s, %s, %s, %s, %s, %s)",
    P.conv2d_weight_grad: f"IMPL.conv2d_weight_grad(%s, %s, %s, %s, %s, %s, %s)",
    P.conv_transpose2d: f"IMPL.conv_transpose2d(%s, %s, %s, %s, %s, %s, %s)",
    P.distribute: "np.broadcast_to(%s, %s)",
    P.dot: "np.dot(%s, %s)",
    P.env_getitem: "%s.get(%s, %s)",
    P.hastag: "%s.has(%s)",
    P.random_initialize: "RandomStateWrapper(np.random.RandomState(%s))",
    P.reshape: "np.reshape(%s, %s)",
    P.scalar_abs: "abs(%s)",
    P.scalar_add: "%s + %s",
    P.scalar_bit_and: "%s & %s",
    P.scalar_bit_lshift: "%s << %s",
    P.scalar_bit_not: "~%s",
    P.scalar_bit_or: "%s | %s",
    P.scalar_bit_rshift: "%s >> %s",
    P.scalar_bit_xor: "%s ^ %s",
    P.scalar_cos: "math.cos(%s)",
    P.scalar_eq: "%s == %s",
    P.scalar_exp: "np.exp(%s)",
    P.scalar_floor: "math.floor(%s)",
    P.scalar_ge: "%s >= %s",
    P.scalar_gt: "%s > %s",
    P.scalar_le: "%s <= %s",
    P.scalar_log: "np.log(%s)",
    P.scalar_lt: "%s < %s",
    P.scalar_max: "max(%s, %s)",
    P.scalar_mod: "%s %% %s",
    P.scalar_mul: "%s * %s",
    P.scalar_ne: "%s != %s",
    P.scalar_pow: "%s ** %s",
    P.scalar_sign: "np.sign(%s)",
    P.scalar_sin: "math.sin(%s)",
    P.scalar_sub: "%s - %s",
    P.scalar_tan: "math.tan(%s)",
    P.scalar_tanh: "np.tanh(%s)",
    P.scalar_trunc: "np.trunc(%s)",
    P.scalar_uadd: "%s",
    P.scalar_usub: "-%s",
    P.scatter: "IMPL.scatter(%s, %s, %s, %s)",
    P.scatter_add: f"IMPL.scatter_add(%s, %s, %s, %s)",
    P.take: "np.take(%s, %s, axis=0)",
    P.transpose: "np.transpose(%s, %s)",
    P.tuple_getitem: "%s[%s]",
}
COMPLEX_MAP = {
    P.array_cast: python_array_cast,
    P.array_getitem: python_array_getitem,
    P.array_map: python_array_map,
    P.array_setitem: python_array_setitem,
    P.env_setitem: python_env_setitem,
    P.gather: python_gather,
    P.make_tuple: python_make_tuple,
    P.max_pool2d: python_max_pool2d,
    P.max_pool2d_grad: python_max_pool2d_grad,
    P.random_uint32: python_random_uint32,
    P.scalar_cast: python_scalar_cast,
    P.scalar_div: python_scalar_div,
    P.scalar_to_array: python_scalar_to_array,
    P.split: python_split,
    P.switch: python_switch,
    P.tagged: python_tagged,
    P.take_grad_inp: python_take_grad_inp,
    P.tuple_setitem: python_tuple_setitem,
    P.unsafe_static_cast: python_unsafe_static_cast,
    P.make_handle: python_make_handle,
    P.universe_getitem: python_universe_getitem,
    P.universe_setitem: python_universe_setitem,
}
OPERATION_MAP = {
    operations.add: operator.add,
    operations.sub: operator.sub,
    operations.mul: operator.mul,
    operations.mod: operator.mod,
    operations.pow: operator.mod,
    operations.eq: operator.eq,
    operations.ne: operator.ne,
    operations.lt: operator.lt,
    operations.gt: operator.gt,
    operations.le: operator.le,
    operations.ge: operator.ge,
    operations.pos: operator.pos,
    operations.neg: operator.neg,
    operations.not_: operator.not_,
    operations.and_: operator.and_,
    operations.or_: operator.or_,
    operations.getitem: operator.getitem,
    operations.bool: operator.truth,
}
FUNCTION_MAP = {
    operations.grad: op_grad,
    operations.value_and_grad: op_value_and_grad,
    operations.myia_next: myia_next,
    operations.myia_iter: myia_iter,
    operations.myia_hasnext: myia_hasnext,
    operations.make_dict: op_make_dict,
}


def convert_operation(c, node, op, *inputs):
    """Convert an operation apply node to a Python code."""
    if op.value is operations.resolve:
        namespace = inputs[0].value
        name = inputs[1].value
        resolved = namespace[name]
        code = None

        if resolved is operations.make_tuple:
            # Return an empty string, to get
            # `(*tuple_vals)` instead of `operation_make_tuple(*tuple_vals)`
            code = ""
        elif resolved in OPERATION_MAP:
            # Resolve as Python's operator module function.
            code = f"operator.{OPERATION_MAP[resolved].__name__}"
        elif resolved in FUNCTION_MAP:
            # Resolve as a specific function.
            # Function will be available as global symbol in compiled function.
            fn = FUNCTION_MAP[resolved]
            code = c.register_global(fn.__name__, fn)
        elif isinstance(resolved, Operation):
            # Use operation's python implementation if available.
            impl = resolved.defaults().get("python_implementation", None)
            if impl is not None:
                code = c.register_global(f"operation_{resolved.name}", impl)
        elif isinstance(resolved, ModuleType):
            # Module imported, probably to call an external function
            # (e.g. `torch.argmax`)
            code = c.register_global(resolved.__name__, resolved)
        elif callable(resolved):
            # Resolved is a callable.
            # Callable will be available as global symbol in compiled function.
            symbol_name = (
                resolved.__name__
                if getattr(resolved, "__name__", "")
                else f"callable_{type(resolved).__name__}"
            )
            code = c.register_global(symbol_name, resolved)

        if code is None:
            raise NotImplementedError(
                f"Unable to resolve: {resolved}, inputs: {' '.join(str(inp) for inp in inputs)}"
            )

        # Register code as a ConstantString, so that
        # node name will be directly replaced with that code.
        c.force_node_constant(node, ConstantString(code))
        return None

    raise NotImplementedError(
        f"Unsupported operation {op} {' '.join(str(inp) for inp in inputs)}"
    )


class PythonMapper:
    """Maps myia operations to Python operations. Copied from relay backend."""

    def __init__(self, simple_map=None, complex_map=None):
        """Create a mapper."""
        self.mapping = {}
        if simple_map is not None:
            self.register_simple(simple_map)
        if complex_map is not None:
            self.register_complex(complex_map)

    def register(self, prim, fn):
        """Register the conversion function for a primitive."""
        assert prim not in self.mapping
        self.mapping[prim] = fn

    def register_simple(self, map_):
        """Register simple conversions (1:1 map to Python ops)."""
        for k, v in map_.items():
            self.register(
                k, lambda c, *args, v=v: v % tuple(c.ref(n) for n in args)
            )

    def register_complex(self, map_):
        """Register complex conversions."""
        for k, v in map_.items():
            self.register(k, v)

    def get(self, fn):
        """Get the mapping for the primitive."""
        return self.mapping[fn]

    def has(self, fn):
        """Return True if given primitive is registered."""
        return fn in self.mapping


MAP = PythonMapper(simple_map=SIMPLE_MAP, complex_map=COMPLEX_MAP)


class NodeVisitor:
    """Visitor for node enumeration. Copied from relay backend."""

    def _visit_array_cast(self, node):
        return [node.inputs[1]]

    def _visit_scalar_to_array(self, node):
        return [node.inputs[1]]

    def _visit_unsafe_static_cast(self, node):
        return [node.inputs[1]]

    def _visit_scalar_cast(self, node):
        return [node.inputs[1]]

    def __call__(self, node):
        """Don't visit called primitives."""
        if node.inputs:
            fn = node.inputs[0]
            if fn.is_constant(Primitive):
                prim = fn.value
                visit = getattr(self, f"_visit_{prim}", None)
                if visit is None:
                    return node.inputs[1:]
                return visit(node)
            else:
                return node.inputs
        elif node.is_constant_graph():
            return [
                fv
                if not isinstance(fv, Graph)
                else list(fv.manager.graph_constants[fv])[0]
                for fv in node.value.free_variables_total
            ]
        return []


def in_graph(g):
    """Generate a filter callable for toposort. Copied from relay backend."""

    def filter(node):
        if node.graph is None:
            return "follow"
        elif node.graph is g:
            return "follow"
        else:
            return "exclude"

    return filter


class _PythonConverter(Converter):
    """Abstract class to convert values between myia and backend."""

    def _default_convert(self, v, t):
        """Default conversion. Just return given value."""
        return v

    def convert_array(self, v, t):
        """Converts array values."""
        return self._default_convert(v, t)

    def convert_scalar(self, v, t):
        """Convert numeric scalars."""
        return self._default_convert(v, t)

    def convert_nil(self, v, t):
        """Convert Nil values."""
        return self._default_convert(v, t)

    def convert_bool(self, v, t):
        """Convert boolean values."""
        return self._default_convert(v, t)

    def convert_universe(self, v, t):
        """Convert a Universe."""
        return self._default_convert(v, t)

    def convert_handle(self, v, t):
        """Convert a Handle."""
        return self._default_convert(v, t)

    def convert_tuple(self, v: tuple, t: AbstractTuple):
        """Convert a tuple."""
        return self._default_convert(v, t)

    def convert_tagged(self, v, t):
        """Convert a union value."""
        return self._default_convert(v, t)

    def convert_type(self, v, t):
        """Convert a type value."""
        return self._default_convert(v, t)

    def convert_random_state(self, v, t):
        """Convert a random state value."""
        return self._default_convert(v, t)


class PythonInputConverter(_PythonConverter):
    """Convert an intermediate value to a backend value."""


class PythonOutputConverter(_PythonConverter):
    """Convert a backend value to an intermediate value."""


class PythonConstantConverter(_PythonConverter):
    """Convert constant values to printable values."""

    def convert_scalar(self, v, t):
        """Convert numeric scalars."""
        numpy_typename = type_to_np_dtype(t)
        # For type names below, we return raw value.
        if numpy_typename in ("bool", "int64", "uint64", "float64"):
            return {
                "bool": bool,
                "int64": int,
                "uint64": int,
                "float64": float,
            }[numpy_typename](v)
        # Otherwise, we return a Python string code for this value.
        return f"np.{numpy_typename}({v})"

    def convert_dead(self, v, t):
        """Convert dead values."""
        return "None"

    def convert_env(self, v, t):
        """Convert a grad env."""
        return "{}"

    def convert_tuple(self, v: tuple, t: AbstractTuple):
        """Convert a tuple."""
        if len(v):
            return f"({', '.join(str(self(v[i], t.elements[i])) for i in range(len(v)))},)"
        return "()"

    def convert_type(self, v, t):
        """Return type name as a string."""
        if isinstance(t.element, AbstractHandle):
            return ConstantString("HandleInstance")
        else:
            myia_type = t.element.xtype()
            if myia_type is Tuple:
                return ConstantString("tuple")
            elif myia_type is Dict:
                # We may need to remember dict definition keys,
                # e.g. for operation make_dict. So, let's return a dict
                # with keys associated to None.
                return repr({key: None for key in v.entries})
            else:
                return ConstantString(f"np.{type_to_np_dtype(myia_type)}")

    def convert_handle(self, v, t):
        """Convert a Handle."""
        return f"HandleInstance({self(v.state, v.abstract or to_abstract(v.state))})"

    def convert_default(self, v):
        """Convert a value given without abstract type."""
        # Return None for constants that must not appear in final code.
        # As inline constants, they won't be converted to a variable,
        # and associated nodes should be handled specifically
        # (e.g. for inputs of operations.resolve apply nodes).
        from myia.utils.misc import Namespace

        if isinstance(v, str):
            # Return a ConstantString so that given string will be inlined.
            return ConstantString(repr(v))
        if isinstance(v, (Namespace, Operation)):
            return None
        raise NotImplementedError(
            f"No default conversion for value {v}, type {type(v)}"
        )


def nested_list_to_code_string(structure, indentation=""):
    """Convert a nested list of strings into a correctly indented Python code."""
    code = ""
    for entry in structure:
        if not isinstance(entry, list):
            code += f"{indentation}{entry}\n"
        elif entry:
            # Indent with 4 spaces.
            entry_code = nested_list_to_code_string(entry, indentation + "    ")
            code += f"{entry_code}\n"
    return code


class PdbRunCall:
    """Helper class to run code with PDB.

    We want PDB to be able to display code using `list .` command, so
    we need to save code into a file and import it as a valid module later.
    """

    def __init__(self, code):
        """Initialize."""
        self.code = code

    def __call__(self, *args):
        """Execute main function with given args."""
        import importlib
        import os
        import pdb
        import tempfile
        import sys

        # Create temporary code file.
        code_fd, code_path = tempfile.mkstemp(
            prefix="myia_backend_python_code_", suffix=".py"
        )
        # Get module directory and name.
        module_dir = os.path.dirname(code_path)
        module_name = os.path.splitext(os.path.basename(code_path))[0]
        # Add module to sys.path
        sys.path.append(module_dir)
        try:
            # Save code into module file.
            with open(code_path, "w") as code_file:
                code_file.write(self.code)
            # Import module.
            module = importlib.import_module(module_name)
            # Run main function.
            output = pdb.runcall(getattr(module, "main"), *args)

        # NB: I don't know why, but code executed after PDB call is
        # systematically reported as uncovered by pytest-cov, so I am
        # excluding following lines from coverage.
        finally:  # pragma: no cover
            # Reset sys.path
            sys.path.remove(module_dir)
            # Close and delete code file.
            os.close(code_fd)
            os.remove(code_path)
        return output  # pragma: no cover


class _Compiler:
    """Base class for Python backend compiler."""

    def has_node(self, node):
        """Return True if given node has already an associated name."""
        raise NotImplementedError()

    def has_constant(self, name):
        """Return True if a constant with given name is already registered."""
        raise NotImplementedError()

    def get_new_name(self, desired_name):
        """Generate a new unique variable name."""
        raise NotImplementedError()

    def get_label(self, node):
        """Generate a label for given node."""
        raise NotImplementedError()

    def ref(self, node):
        """Return name for given node."""
        raise NotImplementedError()

    def get_graph_cache(self):
        """Return graph cache (for graph compilation caching)."""
        raise NotImplementedError()

    def make_const(self, v, t):
        """Convert a value to a Python constant."""
        raise NotImplementedError()

    def register_global(self, name, value):
        """Register a symbol with given name to given value.

        Symbol will be available as a global symbol in compiled function.

        Must return name used to register value
        (either given name, or another if necessary).
        """
        raise NotImplementedError()


class FunctionCompiler(_Compiler):
    """Compiler for a single graph. Compile it to code for a Python function."""

    def __init__(self, graph, parent):
        """Initialize."""
        self.graph = graph
        self.parent = parent
        self.node_to_name = {}
        self.const_name_to_value = {}
        self.closure_name_to_code = {}
        # Associate closure graph to closure name.
        self.closure_to_name = {}

    def local_ref(self, node):
        """Return name for a node inside this function."""
        return self.node_to_name[node]

    def has_node(self, node):
        """Return True if given node has already an associated name."""
        return (
            (node.is_constant_graph() and node.value in self.closure_to_name)
            or node in self.node_to_name
            or self.parent.has_node(node)
        )

    def has_constant(self, name):
        """Return True if a constant with given name is already registered."""
        return (
            name in self.const_name_to_value
            or name in self.closure_name_to_code
            or self.parent.has_constant(name)
        )

    def get_new_name(self, desired_name):
        """Generate a new unique variable name."""
        return self.parent.get_new_name(desired_name)

    def get_label(self, node):
        """Generate a label for given node."""
        return self.parent.get_label(node)

    def ref(self, node):
        """Return a code representation for given node.

        Either the node name, or a constant value (as a string)
        if node is a constant that can be inlined.
        """
        if node.is_constant_graph() and node.value in self.closure_to_name:
            return self.closure_to_name[node.value]
        if node in self.node_to_name:
            name = self.node_to_name[node]
            if name in self.const_name_to_value and self._is_inline_const(
                self.const_name_to_value[name]
            ):
                return str(self.const_name_to_value[name])
            return name
        return self.parent.ref(node)

    def _is_inline_const(self, const_value):
        """Return True if given value can be inlined."""
        return isinstance(
            const_value, (bool, int, float, type(None), ConstantString)
        )

    def get_graph_cache(self):
        """Return graph cache (for graph compilation caching)."""
        return self.parent.get_graph_cache()

    def make_const(self, v, t):
        """Convert a value to a Python constant."""
        return self.parent.make_const(v, t)

    def register_global(self, name, value):
        """Register global symbol."""
        return self.parent.register_global(name, value)

    def force_node_constant(self, node, constant):
        """Associate a constant value to a node.

        Node name will be inlined using given constant in compiled code.
        """
        self.const_name_to_value[self.ref(node)] = constant

    def on_constant(self, node):
        """Generate code for a constant node."""
        if not self.has_constant(self.ref(node)):
            if node.is_constant(Primitive):
                self.on_function(
                    get_prim_graph(
                        self.get_graph_cache(), node.value, node.abstract
                    ),
                    node,
                )
            else:
                self.const_name_to_value[self.ref(node)] = self.make_const(
                    node.value, node.abstract
                )
            return None

    def on_apply(self, node):
        """Generate inline code or call for an apply node."""

        # Make sure all input nodes are registered.
        for input_node in node.inputs:
            self._add_node(input_node)

        # Convert primitive call to an inline code, if possible.
        if node.inputs[0].is_constant(Primitive):
            fn = node.inputs[0].value
            conv = MAP.get(fn)
            if conv is not None:
                return conv(self, *node.inputs[1:])

        # Convert operation call to an inline code, if possible.
        if node.inputs[0].is_constant(Operation):
            return convert_operation(self, node, *node.inputs)

        # Otherwise generate a raw call.
        return f"{self.ref(node.inputs[0])}({', '.join(self.ref(n) for n in node.inputs[1:])})"

    def _add_node(self, node):
        """Associate a name to a node. Return true if a new name was generated."""
        if self.has_node(node):
            # Node already registered.
            return False

        if node.is_constant_graph() and node.value.parent is None:
            # Const graphs without parent are already registered in
            # root PythonCompiler object, hence available from self.ref(node).
            return False

        self.node_to_name[node] = self.get_label(node)

        # If added node is an inlinable constant node, let's register
        # it right now, as we may need to use and inline this node
        # before we meet the node definition itself.
        # E.g. A closure may use a constant defined after closure code
        # in parent function.
        if node.is_constant() and self._is_inline_const(node.value):
            self.on_constant(node)

        return True

    def on_function(self, graph, node):
        """Convert a graph to a function and register it as a closure (constant value)."""
        if not self.has_constant(self.ref(node)):
            self.closure_to_name[graph] = self.ref(node)
            self.closure_name_to_code[self.ref(node)] = FunctionCompiler(
                graph, self
            ).compile()
        return None

    def compile(self):
        """Compile graph to a Python function code.

        Return function parameters names (list) and function body code (nested list).
        """
        graph = self.graph
        # Register parameters.
        param_names = []
        for p in graph.parameters:
            # A parameter should always be a local variable.
            self.node_to_name[p] = self.get_label(p)
            param_names.append(self.local_ref(p))
        seq = []
        # Register nodes.
        for node in toposort(
            graph.output, NodeVisitor(), in_graph(graph), allow_cycles=True
        ):
            if self._add_node(node):
                seq.append(node)
        output = []
        # Get code for each node.
        for op in seq:
            op_name = self.local_ref(op)
            if op.is_apply():
                op_code = self.on_apply(op)
            elif op.is_constant_graph():
                op_code = self.on_function(op.value, op)
            elif op.is_constant():
                op_code = self.on_constant(op)
            else:
                raise AssertionError(f"Unsupported node: {op}")
            # If latest op is graph.output, we can write code
            # to return it immediately.
            if op is seq[-1] is graph.output:
                prefix = "return"
            else:
                prefix = f"{op_name} ="
            # Code for a node may be a list of lines of code.
            # In such case, latest code line should be only the return value.
            if isinstance(op_code, list):
                build_code = op_code[:-1]
                return_code = op_code[-1]
                output.extend(build_code)
                output.append(f"{prefix} {return_code}")
            elif op_code:
                output.append(f"{prefix} {op_code}")
        # Output may be empty, for e.g. if function just returns a parameter.
        if not output:
            # I don't know why, but there may be functions that
            # return a value not even reachable from function code
            # (thus, self.ref() might not find node).
            # In such case, let s just put an exception raising in code.
            # If such function is called, then code will fail.
            try:
                output_name = self.ref(graph.output)
            except KeyError:
                output.append(
                    f'raise RuntimeError("Unreachable: {type(graph.output).__name__} {graph.output}")'
                )
            else:
                output.append(f"return {output_name}")

        constants_code = []
        closures_code = []
        for cst_name, cst_val in self.const_name_to_value.items():
            if not self._is_inline_const(cst_val):
                # Write only non-inlinable constants.
                # Other are inlined, hence it's useless to define them.
                constants_code.append(f"{cst_name} = {cst_val}")
        if constants_code:
            constants_code = ["# Constants"] + constants_code + [""]
        for fn_name, (fn_params, fn_body) in self.closure_name_to_code.items():
            fn_signature = f"def {fn_name}({', '.join(fn_params)}):"
            closures_code.append(fn_signature)
            closures_code.append(fn_body)
        # Body code contains constants, then closures, then function code.
        return param_names, constants_code + closures_code + output


class PythonCompiler(_Compiler):
    """Compile a myia graph into a Python function."""

    REG_PYTHON_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    REG_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]+")
    REG_INVALID_START = re.compile(r"^[^A-Za-z0-9_]+")
    REG_INVALID_END = re.compile(r"[^A-Za-z0-9_]+$")
    REG_DIGIT_START = re.compile(r"^[0-9]")

    def __init__(self):
        """Initialize."""
        self._prim_graph_cache = {}
        self.make_const = PythonConstantConverter()
        self.graph_to_name = {}
        self.fn_name_to_code = {}
        self.node_labeler = NodeLabeler(
            relation_symbols={"copy": "", "opt": ""}
        )
        self.name_counter = Counter()
        self.global_counter = Counter()
        self.globals = {}
        self.graphs_used = set()

    def is_valid_python_name(self, text):
        """Return True if text is a valid Python name."""
        return self.REG_PYTHON_NAME.match(text)

    def convert_to_valid_python_name(self, label):
        """Convert given string to a valid Python variable name."""
        formatted_label = label
        formatted_label = self.REG_INVALID_START.sub("", formatted_label)
        formatted_label = self.REG_INVALID_END.sub("", formatted_label)
        formatted_label = self.REG_INVALID_CHARS.sub("_", formatted_label)
        if self.REG_DIGIT_START.match(formatted_label):
            formatted_label = f"_{formatted_label}"
        if not formatted_label:
            formatted_label = "var_"
        return formatted_label

    def get_label(self, node):
        """Generate a label for given node."""
        label = self.node_labeler.label(node)
        if not self.is_valid_python_name(label):
            label = self.convert_to_valid_python_name(label)
        assert self.is_valid_python_name(label), label
        if label[0] != "_":
            label = f"_{label}"
        self.name_counter.update([label])
        count = self.name_counter[label]
        return label if count == 1 else f"{label}_v{count}"

    def run(self, graph, backend):
        """Compile given graph.

        :type backend: PythonBackend
        """
        mng = manage(graph)
        mng.keep_roots(graph)
        # Graph to name
        for g in mng.graphs:
            if g is graph:
                self.graph_to_name[g] = "main"
            else:
                self.graph_to_name[g] = self.get_label(g)
        # Graph name to function code
        for g, g_name in self.graph_to_name.items():
            self.fn_name_to_code[g_name] = self.convert_func(g)
        # Compilation.
        pre_code = [
            "import math",
            "import operator",
            "import numpy as np",
            "from myia.utils import RandomStateWrapper",
            "from myia.lib import TaggedValue",
            "from myia.utils.universe import HandleInstance",
            "import myia.compile.backends.python.implementations as IMPL",
        ]
        dynamic_imports = [
            f"# Dynamic external import: {name}" for name in self.globals
        ]
        other_functions = []
        main_body = None
        main_signature = None
        for fn_name, (fn_params, fn_body) in self.fn_name_to_code.items():
            if fn_name == "main":
                main_body = fn_body
                main_signature = f"def main({', '.join(fn_params)}):"
            else:
                fn_signature = f"def {fn_name}({', '.join(fn_params)}):"
                other_functions.append(fn_signature)
                other_functions.append(fn_body)
        final_structure = (
            pre_code
            + ([""] if pre_code else [])
            + dynamic_imports
            + ([""] if dynamic_imports else [])
            + other_functions
            + [main_signature, main_body]
        )
        final_code = nested_list_to_code_string(final_structure)

        if backend.debug:
            backend.debug.write(f"\n{final_code}")

        if backend.pdb:
            return PdbRunCall(final_code)

        # Compile code string to a Python executable function
        # reference: https://stackoverflow.com/a/19850183
        compiled = compile(final_code, "", "exec")
        assert "main" not in self.globals
        exec(compiled, self.globals)
        return self.globals["main"]

    def has_node(self, node):
        """Return True if given node has already an associated name."""
        return isinstance(node, Graph) and node in self.graph_to_name

    def has_constant(self, name):
        """Return True if a constant with given name is already registered."""
        return name in self.fn_name_to_code

    def get_new_name(self, desired_name):
        """Generate a new name usable as a node name."""
        if desired_name[0] != "_":
            desired_name = f"_{desired_name}"
        self.name_counter.update([desired_name])
        count = self.name_counter[desired_name]
        return desired_name if count == 1 else f"{desired_name}_v{count}"

    def ref(self, node):
        """Return a name (string) associated to given node."""
        if not (node.is_constant_graph() and node.value.parent is None):
            raise KeyError(f"Expected a root graph, got {node}")
        self.graphs_used.add(self.graph_to_name[node.value])
        return self.graph_to_name[node.value]

    def get_graph_cache(self):
        """Return graph cache (for graph compilation caching)."""
        return self._prim_graph_cache

    def register_global(self, name, value):
        """Register global symbol for compiled code.

        :param name: name to register
        :param value: value to register
        :return: name used to register value. May be different
            from given name if given name was already used.
        """
        # If name is already associated to given value, then nothing to do.
        if name in self.globals and self.globals[name] is value:
            return name
        self.global_counter.update([name])
        count = self.global_counter[name]
        name = name if count == 1 else f"{name}_v{count}"
        self.globals[name] = value
        return name

    def convert_func(self, graph):
        """Convert a graph to Python function code."""
        return FunctionCompiler(graph, self).compile()


class PythonBackend(Backend):
    """Python backend."""

    def __init__(self, debug=False, pdb=False):
        """Initialize.

        :param debug: if False or None, do nothing.
            If True, print generated code in stdout.
            Otherwise, should be an output stream (e.g. stdout or a StringIO)
            and generated code will be written into given stream.
        :param pdb: if True, compiled function will be run in a pdb instance
        """
        if debug:
            debug = sys.stdout if debug is True else debug
            assert hasattr(debug, "write")

        self.to_backend_value = PythonInputConverter()
        self.from_backend_value = PythonOutputConverter()
        self.debug = debug
        self.pdb = bool(pdb)

    def compile(self, graph, argspec, outspec):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        # Remove symbolic key instances.
        graph = convert_grad(graph)
        # Then compile the graph.
        # Create a PythonCompiler object each time we want to compile,
        # to avoid sharing states through different compilations
        # (e.g. for "PythonCompiler.globals" member).
        return PythonCompiler().run(graph, self)

    def supports_prim_group(self, prim_group):
        """Return True if given primitive group is supported."""
        return all(MAP.has(prim) for prim in prim_group.primitives)


def compile_graph(graph, **options):
    """Easy-to-use public method to convert a graph to a Python function.

    :param graph: myia graph to compile
    :param options: backend options
    :return: callable Python function representing compiled graph
    """
    return PythonBackend(**options).compile(graph, None, None)


def load_options(debug=False, pdb=False):
    """Load backend options."""
    return {"debug": debug, "pdb": pdb}


def load_backend(options):
    """Load backend."""
    return PythonBackend(**options)
