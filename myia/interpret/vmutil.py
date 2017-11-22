
from typing import Any, List, Callable
from types import FunctionType
from numpy import ndarray
from ..util import EventDispatcher, HReprBase
from ..util.debug import Breakpoint
from ..lib import Closure, Primitive, IdempotentMappable, Record, ZERO
from ..stx import MyiaASTNode, Symbol, ValueNode, LambdaNode
from ..symbols import builtins, object_map
from ..parse import parse_function


###################
# Function object #
###################


class Function(HReprBase, IdempotentMappable):
    """
    Represents a Myia-transformed function.
    """
    def __init__(self,
                 ast: LambdaNode,
                 eval_env: 'EvaluationEnv') -> None:
        assert isinstance(ast, LambdaNode)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.ast = ast
        self.code = VMCode(ast.body, lbda=ast)
        self.eval_env = eval_env
        self.primal_sym = ast.primal
        self.grad: Callable[[int], Function] = None

    def configure(self, **config):
        env = self.eval_env.reconfigure(config)
        return Function(self.ast, env)

    def __call__(self, *args):
        ast = self.ast
        assert len(args) == len(ast.args)
        return self.eval_env.run(
            self.code,
            {s: arg for s, arg in zip(ast.args, args)}
        )

    def __str__(self):
        return f'Func({self.ast.ref or self.ast})'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.ast)

    def __eq__(self, other):
        return type(other) is Function \
            and self.ast == other.ast

    def __add__(self, other):
        # TODO: Fix the following issue, which happens sometimes.
        #       I believe the main reason is that for the backpropagator,
        #       Grad builds a closure on the wrong function (?)
        # if self != other:
        #     raise Exception('The functions being added are different.')
        return self

    def __hrepr__(self, H, hrepr):
        return hrepr.titled_box('Func', [hrepr(self.ast.ref or self.ast)])


##########################################
# Code representation for stack-based VM #
##########################################


class Instruction:
    """
    An instruction for the stack-based VM.

    Attributes:
        command: The instruction name.
        node: The Myia node that this instruction is computing.
        args: Instruction-specific arguments.
    """
    def __init__(self,
                 command: str,
                 node: MyiaASTNode,
                 *args: Any) -> None:
        self.command = command
        self.node = node
        self.args = args

    def __str__(self):
        args = ", ".join(map(str, self.args))
        return f'{self.command}:{args}'


class VMCode(HReprBase):
    """
    Compile a MyiaASTNode into a list of instructions compatible
    with the stack-based VM.

    See VMFrame's ``instruction_<name>`` methods for more
    information.

    Attributes:
        node: The original node.
        instructions: A list of instructions to implement this
            node's behavior.
    """
    def __init__(self,
                 node: MyiaASTNode,
                 instructions: List[Instruction] = None,
                 lbda: LambdaNode = None) -> None:
        assert instructions or (lbda and lbda.body is node)
        self.node = node
        self.lbda = lbda
        if instructions is None:
            self.instructions: List[Instruction] = []
            self.process(self.node)
        else:
            self.instructions = instructions

    def instr(self, name, node, *args) -> None:
        self.instructions.append(Instruction(name, node, *args))

    def process(self, node) -> None:
        # Dispatch to process_<node_type>
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'process_' + cls)
        except AttributeError:
            msg = f"No 'process_{cls}' method to process {node!r}."
            raise AttributeError(msg)
        rval = method(node)

    def process_ApplyNode(self, node) -> None:
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('reduce', node, len(node.args))

    def process_BeginNode(self, node) -> None:
        for stmt in node.stmts:
            self.process(stmt)

    def process_ClosureNode(self, node) -> None:
        self.process(node.fn)
        self.process(builtins.mktuple)
        for arg in node.args:
            self.process(arg)
        # self.instr('tuple', node, len(node.args))
        self.instr('reduce', node, len(node.args))
        self.instr('closure', node)

    def process_LetNode(self, node) -> None:
        for k, v in node.bindings:
            self.process(v)
            self.instr('store', node, k)
        self.process(node.body)

    def process_Symbol(self, node) -> None:
        self.instr('fetch', node, node)

    def process_TupleNode(self, node) -> None:
        # for x in node.values:
        #     self.process(x)
        # self.instr('tuple', node, len(node.values))
        self.process(builtins.mktuple)
        for arg in node.values:
            self.process(arg)
        self.instr('reduce', node, len(node.values))

    def process_ValueNode(self, node) -> None:
        self.instr('push', node, node.value)

    def __hrepr__(self, H, hrepr):
        rows = []
        for instr in self.instructions:
            row = H.tr()
            for x in (instr.command, *instr.args):
                row = row(H.td(hrepr(x)))
            rows.append(row)
        return H.table['VMCodeInstructions'](*rows)


###############################
# EvaluationEnv (VM-agnostic) #
###############################


class EvaluationEnv(dict):
    """
    Context for evaluating Myia code.

    Attributes:
        primitives: The functions to use for Myia primitives. They can
            be straightforward evaluators, or abstract evaluators, etc.
        pool: To resolve global variables.
        vm_class: The class to use to instantiate a VM.
        setup: A nullary function executed each time evaluate() is
            called.
        config: Configuration that will be given to VM's constructor
            as keyword arguments.
    """
    def __init__(self, primitives, pool, config={}):
        self.compile_cache = {}
        self.primitives = primitives
        self.pool = pool
        self.config = config

    def devolve_value(self, v):
        """
        Convert a value to a corresponding ``Symbol`` or ``LambdaNode``,
        if one can be found, otherwise return the value unchanged.

        * Check if the value is mapped to a symbol in ``symbols.object_map``
        * Check for the ``__myia_symbol__`` or ``__myia_lambda__`` field.
        """
        try:
            x = object_map[v]
        except (TypeError, KeyError):
            pass
        else:
            return x
        if hasattr(v, '__myia_symbol__'):
            return v.__myia_symbol__
        if hasattr(v, '__myia_lambda__'):
            return v.__myia_lambda__
        return v

    def convert_value(self, v):
        """
        Convert a value to a value compatible with the VM. This differs
        from ``import_value`` by not handling callables.
        """
        # TODO: convert recursively with smap...
        accepted_types = (bool, int, float,
                          Primitive, Function, Closure,
                          ndarray, list, tuple, Record, str,
                          Breakpoint)
        if isinstance(v, accepted_types) or v is ZERO or v is None:
            return v
        elif isinstance(v, Symbol):
            raise ValueError(f'Myia cannot resolve {v} '
                             f'from namespace {v.namespace}')
        else:
            raise TypeError(f'Myia cannot convert value: {v}')

    def import_value(self, v):
        """
        Import the Python value v into the corresponding format that is
        suitable for the environment's VM. This proceeds as follows:

        * If v is a ``Symbol``, we check if it is in ``primitives``.
        * If v is a ``LambdaNode``, we compile it with ``self.compile``.
        * If v is a Python function, we parse it and convert it to a
          ``LambdaNode``, which we import.
        """
        try:
            x = self.primitives[v]
        except (TypeError, KeyError):
            pass
        else:
            return self.import_value(x)

        if isinstance(v, (type, FunctionType)):
            # Note: Python's FunctionType, i.e. actual Python functions
            try:
                lbda = parse_function(v)
            except (TypeError, OSError):
                raise ValueError(f'Myia cannot interpret value: {v}')
            return self.import_value(lbda)
        elif isinstance(v, LambdaNode):
            return self.get_compiled(v)
        else:
            return self.convert_value(v)

    def export_value(self, value):
        return value

    def get_compiled(self, lbda):
        fimpl = self.compile_cache.get(lbda, None)
        if fimpl is None:
            fimpl = self.compile(lbda)
            self.compile_cache[lbda] = fimpl
        return fimpl

    def compile(self, lbda):
        return Function(lbda, self)

    def setup(self):
        pass

    def vm(self, code, local_env):
        raise NotImplementedError('Must override vm method in subclass.')

    def reconfigure(self, new_config):
        cfg = {**self.config}
        cfg.update(new_config)
        return self.__class__(self.primitives, self.pool, cfg)

    def run(self, code, local_env):
        return self.vm(code, local_env).run()

    def evaluate(self, node):
        self.setup()
        if isinstance(node, (Symbol, LambdaNode)):
            return self[node]
        else:
            return self.run(VMCode(node), {})

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            if isinstance(item, Symbol):
                try:
                    raw_value = self.pool[item]
                except (TypeError, KeyError, NameError):
                    raw_value = item
            else:
                raw_value = item
            value = self.devolve_value(raw_value)
            self[item] = self.import_value(value)
            return self[item]


class EvaluationEnvCollection:
    """
    Constructs an EvaluationEnv from a configuration dictionary,
    and optionally caches the association between a configuration
    and an EvaluationEnv.
    """
    def __init__(self, eenv_class, *args, cache=True):
        self.args = args
        self.eenv_class = eenv_class
        self.eenvs = {}
        self.cache = cache

    def get_env(self, **config):
        cfg = frozenset(config.items())
        if self.cache and cfg in self.eenvs:
            return self.eenvs[cfg]
        else:
            eenv = self.eenv_class(*self.args, config=config)
            self.eenvs[cfg] = eenv
            return eenv

    def run_env(self, node, **config):
        return self.get_env(**config).evaluate(node)
