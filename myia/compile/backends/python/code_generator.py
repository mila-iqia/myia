"""Code generator used to generate compiled code."""

import builtins
import operator
import types
from collections import Counter

from ovld.core import _Ovld

from myia import basics
from myia.abstract.data import Placeholder
from myia.compile.backends.python.optimizer import ASSIGN
from myia.ir import Constant, NodeLabeler
from myia.utils.directed_graph import DirectedGraph


class PythonGenLabeler(NodeLabeler):
    """Labeler for Python code generation."""

    def translate_relation(self, rel):
        """Translate a relation as _ separated name."""
        return ["_", rel]

    def generate_name(self, identifier):
        """Generate a new name."""
        return f"_{identifier}"

    def disambiguate(self, label, element_id):
        """Disambiguate identical symbols."""
        return f"_{label}_{element_id}"

    def describe_object(self, node):
        """Describe an object by value."""
        if isinstance(node, Constant) and not node.is_constant_graph():
            if isinstance(node.value, Placeholder):
                return "None"
            return str(node.value)


class CodeGenerator:
    """Helper class to convert graph and nodes to code string."""

    def __init__(self, *, skip=None, nonlocals=None, replace=None, rename=None):
        """Initialize.

        Optional dictionaries can be passed to control code generation.

        Arguments:
            skip: nodes to skip
                Dict mapping a myia graph to a sequence of myia nodes to skip.
                If found, these nodes won't be converted to code lines.
            nonlocals: non-local variables
                Dict mapping a myia graph to a sequence of non-local myia nodes.
                Nodes in `nonlocals` will be marked with `nonlocal` keyword
                at the top of closure code.
            replace: nodes to replace
                Dictionary mapping a myia node to a myia replacement node.
                During code generation, node will be replaced everywhere
                it appears, ie., for its code line and inside any apply node
                that uses it.
            rename: nodes to rename
                Dictionary mapping a myia node to a label-provider myia node
                Node label will be label of associated node in this dictionary.
                Node label will be replaced everywhere the node appears, ie.,
                in its code line (`label` = `expr`) and inside any apply node
                that uses it.
        """
        self.skip = skip or {}
        self.nonlocals = nonlocals or {}
        self.replace = replace or {}
        self.rename = rename or {}
        self.inline_nodes = {}
        self.lbl = PythonGenLabeler(reverse_order=True)
        self.global_counter = Counter()
        self.globals = {}

        self.simple_map = {
            operator.add: "{} + {}",
            operator.and_: "{} & {}",
            operator.contains: "{} in {}",
            operator.eq: "{} == {}",
            operator.floordiv: "{} // {}",
            operator.ge: "{} >= {}",
            operator.getitem: "{}[{}]",
            operator.gt: "{} > {}",
            operator.invert: "~{}",
            operator.is_: "{} is {}",
            operator.is_not: "{} is not {}",
            operator.le: "{} <= {}",
            operator.lshift: "{} << {}",
            operator.lt: "{} < {}",
            operator.mod: "{} % {}",
            operator.mul: "{} * {}",
            operator.ne: "{} != {}",
            operator.neg: "-{}",
            operator.not_: "not {}",
            operator.or_: "{} | {}",
            operator.pos: "+{}",
            operator.pow: "{} ** {}",
            operator.rshift: "{} >> {}",
            operator.sub: "{} - {}",
            operator.truediv: "{} / {}",
            operator.truth: "bool({})",
            operator.xor: "{} ^ {}",
            ASSIGN: "{}",
        }
        self.complex_map = {
            basics.user_switch: self._str_user_switch,
            basics.switch: self._str_user_switch,
            basics.make_tuple: self._str_make_tuple,
            basics.make_list: self._str_make_list,
            basics.make_dict: self._str_make_dict,
            basics.make_set: self._str_make_set,
            basics.apply: self._str_apply,
            getattr: self._str_getattr,
        }

    def directed_graph_to_code(self, directed: DirectedGraph):
        """Convert directed graph to a function code.

        Arguments:
            directed: directed graph to convert

        Returns:
            list: a nested list with two values:
                [function header code: str, function body code: list]
        """
        graph = directed.data

        # Generate graph label before anything else, to make sure
        # graph gets the smallest default label number if necessary.
        graph_label = self.label(graph)

        # Generate function arguments list.
        nb_var_params = bool(graph.varargs) + bool(graph.kwargs)
        required_params = graph.parameters[
            : (len(graph.parameters) - nb_var_params)
        ]
        posonly = required_params[: graph.posonly]
        args = required_params[
            graph.posonly : (len(required_params) - graph.kwonly)
        ]
        kwonly = required_params[(len(required_params) - graph.kwonly) :]
        param_strings = []
        # Positional arguments.
        param_strings.extend(self.label(p) for p in posonly)
        if posonly:
            param_strings.append("/")
        # Arguments.
        param_strings.extend(self.label(p) for p in args)
        # "*" or "*args"
        if graph.varargs:
            param_strings.append(f"*{self.label(graph.varargs)}")
        elif kwonly:
            param_strings.append("*")
        # Keyword-only arguments.
        param_strings.extend(self.label(p) for p in kwonly)
        # "**kwargs"
        if graph.kwargs:
            param_strings.append(f"**{self.label(graph.kwargs)}")

        # Generate function header with correct arguments list.
        header = f"def {graph_label}({', '.join(param_strings)}):"
        # Register nonlocal variables.
        code = [
            f"nonlocal {self.label(outer_variable)}"
            for outer_variable in self.nonlocals.get(graph, ())
        ]

        # Get nodes from user to used nodes.
        sequence = list(directed.visit())
        # Reverse sequences to have used nodes then user nodes.
        sequence.reverse()
        # Remove None node.
        assert sequence.pop() is None
        # We skip graph.return_, as it is converted later.
        assert sequence.pop() is graph.return_
        # If graph.output was just before graph.return_, we can inline return statement as `return <expr>`
        inline_return = False
        if sequence and sequence[-1] is graph.output:
            sequence.pop()
            inline_return = True

        for element in sequence:
            # Skip element if necessary.
            if element in self.skip.get(graph, ()):
                continue
            if isinstance(element, DirectedGraph):
                code.extend(
                    ([""] if code else [])
                    + self.directed_graph_to_code(element)
                )
            else:
                # Replace node before passing to `_node_to_line`.
                line = self._node_to_line(self.replace.get(element, element))
                if line:
                    code.append(line)

        # We then convert graph return node.
        # Replace node if necessary.
        output = self.replace.get(graph.output, graph.output)
        code.append(
            f"return {self._node_to_expr(output) if inline_return else self._rvalue(output)}"
        )
        return [header, code]

    def _node_to_line(self, replaced_node):
        """Convert an already-replaced apply node to a line of code."""
        assert replaced_node.is_apply()
        expr = self._node_to_expr(replaced_node)
        return (
            None if expr is None else f"{self._label(replaced_node)} = {expr}"
        )

    def _node_to_expr(self, replaced_node):
        """Convert an already-replaced apply node to an expr."""
        # Replace node function and inputs if necessary.
        fn = self.replace.get(replaced_node.fn, replaced_node.fn)
        inputs = [self.replace.get(inp, inp) for inp in replaced_node.inputs]
        if fn.is_constant():
            if fn.value in self.simple_map:
                return self._default_formatter(
                    self.simple_map[fn.value], inputs
                )
            elif fn.value in self.complex_map:
                return self.complex_map[fn.value](*inputs)
            elif fn.value is basics.resolve:
                namespace = inputs[0].value
                symbol_name = inputs[1].value
                symbol = namespace[symbol_name]
                # We register node as an inline node.
                # Node usage will be directly replaced with resolved name.
                self.inline_nodes[replaced_node] = self._register_global(
                    symbol_name, symbol
                )
                # We return None to notify that node does not need an assignment.
                return None
            elif isinstance(
                fn.value,
                (
                    type,
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                    _Ovld,
                ),
            ):
                name = self._register_global(fn.value.__name__, fn.value)
                return f"{name}({', '.join(map(self._rvalue, inputs))})"
        return f"{self._label(fn)}({', '.join(map(self._rvalue, inputs))})"

    def _rvalue(self, replaced_node):
        """Inline constant strings or get name for given already-replaced node."""
        if isinstance(replaced_node, Constant) and replaced_node.is_constant(
            str
        ):
            return repr(replaced_node.value)
        return self._label(replaced_node)

    def _label(self, replaced_node):
        """Get name for given already-replaced node."""
        replaced_node = self.rename.get(replaced_node, replaced_node)
        if replaced_node in self.inline_nodes:
            return self.inline_nodes[replaced_node]
        return self.lbl(replaced_node)

    def label(self, node):
        """Get name for given node."""
        # This method is public and can be called to a not-yet replaced node,
        # so, replace node if necessary.
        return self._label(self.replace.get(node, node))

    def _register_global(self, name, value):
        """Register global symbol for compiled code.

        Arguments:
            name: name to register
            value: value to register

        Returns:
            str: name used to register value. May be different
                from given name if given name was already used.
        """
        # If name is a builtin, no need to register.
        if getattr(builtins, name, None) is value:
            return name
        # If name is already associated to given value, then nothing to do.
        if name in self.globals and self.globals[name] is value:
            return name
        self.global_counter.update([name])
        count = self.global_counter[name]
        name = name if count == 1 else f"{name}_v{count}"
        self.globals[name] = value
        return name

    # Node-to-code converters. Always received already replaced nodes.

    def _default_formatter(self, format_string, inputs):
        """Generate code using given format string and input nodes."""
        return format_string.format(*[self._rvalue(inp) for inp in inputs])

    def _str_user_switch(self, cond, if_true, if_false):
        """Formatter for apply `user_switch`."""
        return f"{self._label(if_true)} if {self._label(cond)} else {self._label(if_false)}"

    def _str_make_tuple(self, *inputs):
        """Formatter for apply `make_tuple`."""
        return f"({', '.join(self._rvalue(inp) for inp in inputs)})"

    def _str_make_list(self, *inputs):
        """Formatter for apply `make_list`."""
        return f"[{', '.join(self._rvalue(inp) for inp in inputs)}]"

    def _str_make_set(self, *inputs):
        """Formatter for apply `make_set`."""
        return "{" + (", ".join(self._rvalue(inp) for inp in inputs)) + "}"

    def _str_make_dict(self, *inputs):
        """Formatter for apply `make_dict`."""
        assert not len(inputs) % 2
        pairs = [self._rvalue(inp) for inp in inputs]
        return (
            "{"
            + ", ".join(
                "{}: {}".format(pairs[2 * i], pairs[2 * i + 1])
                for i in range(len(pairs) // 2)
            )
            + "}"
        )

    def _str_apply(self, fn, args, kwargs):
        """Formatter for apply `apply`, which seems to represent a call with args and kwargs."""
        return f"{self._label(fn)}(*{self._rvalue(args)}, **{self._rvalue(kwargs)})"

    def _str_getattr(self, *params):
        """Formatter for apply getattr."""
        assert len(params) == 2
        obj, symbol = params
        return f"{self._label(obj)}.{self._label(symbol)}"
