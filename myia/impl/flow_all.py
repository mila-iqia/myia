
from .main import symbol_associator, impl_bank
from ..stx import Value, Tuple
from ..util import Keyword
from unification import var
from ..inference.types import Array, Bool, Float64, Int64, Type


ANY = Keyword('ANY')


impl_bank['flow'] = {
    'value': {},
    'type': {}
}


@symbol_associator('flow')
def impl_flow(sym, name, fn):
    impl_bank['flow']['value'][sym] = fn
    return fn


@symbol_associator('flow_type')
def impl_flow_type(sym, name, fn):
    def flow(dfa, node):
        ttrack = dfa.tracks['type']

        # TODO: this is basically redoing a brute force test
        # every time, it's super inefficient
        def signature_present(sig):
            ins, out = sig
            for i, arg in zip(ins, node.args):
                if i not in dfa.values[ttrack][arg]:
                    break
            else:
                return out
            return False

        for arg in node.args:
            @dfa.on_flow_from(arg)
            def on_arg(track, value):
                if isinstance(value, Type):
                    for sig in fn():
                        out = signature_present(sig)
                        if out:
                            dfa.propagate(node, ttrack, out)

    impl_bank['flow']['type'][sym] = flow
    return fn


@impl_flow
def flow_switch(dfa, node):
    @dfa.on_flow_from(node.args[0], 'value')
    def on_cond(track, value):
        if value == Value(True):
            dfa.flow_to(node.args[1], node)
        elif value == Value(False):
            dfa.flow_to(node.args[2], node)
        elif value == ANY:
            dfa.flow_to(node.args[1], node)
            dfa.flow_to(node.args[2], node)
        else:
            raise TypeError('Condition for switch is not boolean.')


@impl_flow
def flow_index(dfa, node):
    indexes = set()
    tups = set()
    track = dfa.value_track
    dat, idx = node.args

    @dfa.on_flow_from(idx, track)
    def on_index(track, value):
        if isinstance(value, Value):
            vidx = value.value
            indexes.add(vidx)
            for tup in tups:
                dfa.flow_to(tup.values[vidx], node)

    @dfa.on_flow_from(dat, track)
    def on_data(track, value):
        if isinstance(value, Tuple):
            tups.add(value)
            for vidx in indexes:
                dfa.flow_to(value.values[vidx], node)


def std_type(nargs):
    # T = var()
    # return ([Array[T] for _ in range(nargs)], Array[T])
    return [((T,) * nargs, T) for T in [Float64, Int64]]


@impl_flow_type
def flow_type_add():
    return std_type(2)


@impl_flow_type
def flow_type_dot():
    return std_type(2)


@impl_flow_type
def flow_type_less():
    return [((Float64, Float64), Bool),
            ((Int64, Int64), Bool)]


def default_flow(dfa, node):
    dfa.propagate(node, 'value', ANY)
