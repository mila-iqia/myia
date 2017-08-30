
from .main import symbol_associator, impl_bank
from ..stx import Value
from ..util import Keyword
from unification import var
from ..inference.types import Array


ANY = Keyword('ANY')


@symbol_associator('flow')
def impl_flow(sym, name, fn):
    impl_bank['flow'][sym] = fn
    return fn


@impl_flow
def flow_switch(dfa, node):
    @dfa.on_flow_from(node.args[0])
    def on_cond(value):
        if value == Value(True):
            dfa.flow_to(node.args[1], node)
        elif value == Value(False):
            dfa.flow_to(node.args[2], node)
        elif value == ANY:
            dfa.flow_to(node.args[1], node)
            dfa.flow_to(node.args[2], node)
        else:
            raise TypeError('Condition for switch is not boolean.')


# def std_type(nargs):
#     T = var()
#     return ([Array[T] for _ in range(nargs)], Array[T])


# @impl_flow_type
# def type_add():
#     return std_type(2)


# @impl_flow_type
# def type_dot():
#     return std_type(2)


# @impl_flow_type
# def type_equal():
#     T = var()
#     return ((T, T), Bool)


# @impl_flow_type
# def type_switch():
#     T = var()
#     return ((Bool, T, T), T)


def default_flow(dfa, node):
    dfa.add_value(node, ANY)
