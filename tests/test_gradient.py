from myia.api import myia
from myia.composite import grad
import pdb

def f(x, y):
    a = x ** 3
    b = y ** 4
    c = a * b
    return c

def f1(l1, l2):
	ll1 = l1[0:3]
	ll2 = l2[0:3]
	return ll2[0]*ll2[0]


def f2(l1, l2):
	return l2[0:3][0]*l2[0:3][1]*l1

def f3(x, y):
   y = y[::]
   a = y[0] * y[1] * x
   return a

def f4(x, y):
   x = x[::]
   a = x[0] * x[1] * y
   return a

@myia
def main(x, y):
	#return f1(x, y)
    dfdx = grad(f4)(x, y)
    return dfdx

l = 2.0
l1 = [1.0, 2.0, 3.0, 4.0]
l2 = [5.0, 6.0, 7.0, 8.0]
#print(f1(l1,l2))


#out = main(l1,l2)
#out = main(3.0, 2.0)
out = main(l1,l) # f4
#out = main(l,l1) # f2
print(out)