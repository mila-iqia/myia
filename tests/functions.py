
from myia.front import myia

def ann(**annotations):
    def annotate(fn):
        fn.annotations = annotations
        return fn
    return annotate


############
# Features #
############

@ann(test=((1, 2), 3))
def fn_just_add(x, y):
    return x + y

@ann(test=(13, -33))
def fn_shadow_variable(x):
    x = x * 2
    x = x + 7
    x = -x
    return x

@ann(tests=[(-10, -1), (0, -1), (10, 1)])
def fn_if(x):
    if x > 0:
        return 1
    else:
        return -1

@ann(tests=[(-1, 303), (0, 303), (1, 30)])
def fn_if2(x):
    if x > 0:
        a = 10
        b = 20
    else:
        a = 101
        b = 202
    return a + b

@ann(tests=[((100, 10), 0),
            ((50, 7), -6)])
def fn_while(x, y):
    while x > 0:
        x -= y
    return x


##########
# Errors #
##########

@ann(stxerr="Varargs")
def fn_varargs1(x, y, *args):
    return 123

@ann(stxerr="Varargs")
def fn_varargs2(**kw):
    return 123



##################
# Known failures #
##################

@ann(test=(50, 55), xfail=True)
def fn_forward_ref(x):
    def h():
        return g(5)
    def g(y):
        return x + y
    return h()
