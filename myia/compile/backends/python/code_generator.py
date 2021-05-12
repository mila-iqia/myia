"""Code generator used to generate compiled code."""

import builtins
import operator
from collections import Counter

from myia.compile.backends.python.directed_graph import DirectedGraph
from myia.compile.backends.python.implementations import (
    Handle,
    Universe,
    myia_hasnext,
    myia_iter,
    myia_next,
)
from myia.ir import Constant
from myia.utils.info import Labeler


class NodeLabeler:
    """Node labeler used in Python backend.

    Combine a Labeler (from myia.utils.info) and a node cache to generate default names when necessary.
    """

    def __init__(self):
        self.cache = {}
        self.default_name_counter = Counter()
        self.lbl = Labeler(
            relation_translator=self._relation_translator,
            name_generator=self._name_generator,
            disambiguator=self._disambiguator,
            object_describer=self._object_describer,
        )

    def __call__(self, node):
        """Generate label for given node."""
        if isinstance(node, Constant) and node.is_constant_graph():
            # Use labeler for graph in constant node.
            return self.lbl(node.value)
        else:
            # Use labeler for anything else.
            return self.lbl(node)

    @classmethod
    def _relation_translator(cls, rel):
        return f"{rel}_"

    @classmethod
    def _name_generator(cls, identifier):
        return f"_{identifier}"

    @classmethod
    def _disambiguator(cls, label, element_id):
        return f"_{label}_{element_id}"

    def _object_describer(self, node):
        if isinstance(node, Constant) and not node.is_constant_graph():
            return str(node.value)


class CodeGenerator:
    """Helper class to convert graph and nodes to code string."""

    def __init__(self, *, skip=None, nonlocals=None, replace=None, rename=None):
        """Initialize.

        Optional dictionaries can be passed to control code generation.

        :param skip: nodes to skip
            Dictionary mapping a myia graph to a sequence of myia nodes to skip.
            If found, these nodes won't be converted to code lines.
        :param nonlocals: non-local variables
            Dictionary mapping a myia graph to a sequence of non-local myia nodes.
            Nodes in `nonlocals` will be marked with `nonlocal` keyword at the top
            of closure code.
        :param replace: nodes to replace
            Dictionary mapping a myia node to a myia replacement node.
            During code generation, node will be replaced eeverywhere it appears, ie.,
            for its code line and inside any apply node that uses it.
        :param rename: nodes to rename
            Dictionary mapping a myia node to a label-provider myia node
            Node label will be label of associated node in this dictionary.
            Node label will be replaced everywhere the node appears, ie.,
            in its code line (`label` = `expr`) and inside any apply noe that uses it.
        """
        universe = Universe()
        self.skip = skip or {}
        self.nonlocals = nonlocals or {}
        self.replace = replace or {}
        self.rename = rename or {}
        self.inline_nodes = {}
        self.lbl = NodeLabeler()
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
            "typeof": "type({})",
            "assign": "{}",
        }
        self.complex_map = {
            "user_switch": self._str_user_switch,
            "switch": self._str_user_switch,
            "make_tuple": self._str_make_tuple,
            "make_list": self._str_make_list,
            "make_dict": self._str_make_dict,
            "apply": self._str_apply,
            getattr: self._str_getattr,
        }
        self.module_implementations = {
            "make_handle": Handle,
            "universe_setitem": universe.setitem,
            "universe_getitem": universe.getitem,
            "python_iter": myia_iter,
            "python_hasnext": myia_hasnext,
            "python_next": myia_next,
        }

    def directed_graph_to_code(self, directed: DirectedGraph):
        """Convert directed graph to a function code.

        :param directed: directed graph to convert
        :return: a nested list with two values:
            [function header code: str, function body code: list]
        """
        graph = directed.data
        header = f"def {self.label(graph)}({', '.join(self.label(p) for p in graph.parameters)}):"
        # Register nonlocal variables.
        code = [
            f"nonlocal {self.label(outer_variable)}"
            for outer_variable in self.nonlocals.get(graph, ())
        ]

        # Get nodes from user to used nodes.
        sequence = list(directed.visit())
        # Reverse sequences to have used nodes then user nodes.
        sequence.reverse()
        # We skip graph.return_, as it is converted later.
        assert sequence[-1] is graph.return_
        sequence.pop()
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
            elif fn.value in self.module_implementations:
                name = self._register_global(
                    fn.value, self.module_implementations[fn.value]
                )
                return f"{name}({', '.join(map(self._rvalue, inputs))})"
            elif fn.value == "resolve":
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

        :param name: name to register
        :param value: value to register
        :return: name used to register value. May be different
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
