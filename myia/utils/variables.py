"""Define variables for use in patterns all over Myia."""

from ..ir import Graph
from .misc import Namespace
from .unify import SVar, Var, var


def constvar(cls=object):
    """Return a variable matching a Constant of the given type."""
    def _is_c(n):
        return n.is_constant(cls)
    return var(_is_c)


#####################
# Generic variables #
#####################


X = Var('X')
Y = Var('Y')
Z = Var('Z')
X1 = Var('X1')
X2 = Var('X2')
X3 = Var('X3')
X4 = Var('X4')
X5 = Var('X5')


#############
# Constants #
#############


C = constvar()
C1 = constvar()
C2 = constvar()
CNS = constvar(Namespace)
G = constvar(Graph)
G1 = constvar(Graph)
G2 = constvar(Graph)
NIL = var(lambda x: x.is_constant() and x.value == ())


######################
# Sequence variables #
######################


Xs = SVar(Var())
Ys = SVar(Var())
Cs = SVar(constvar())


__all__ = [
    'X',
    'Y',
    'Z',
    'X1',
    'X2',
    'X3',
    'X4',
    'X5',
    'C',
    'C1',
    'C2',
    'CNS',
    'G',
    'G1',
    'G2',
    'NIL',
    'Xs',
    'Ys',
    'Cs',
    'constvar',
]
