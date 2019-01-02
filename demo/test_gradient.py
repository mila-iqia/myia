from myia.api import myia
from myia.composite import grad
import numpy as np
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
   x = x[::]
   return x[0] * x[1]* y

def f4(x, y):
   x = x[::]
   a = x[0] * x[1] * y
   return a

@myia
def main(x, y):
	#return f1(x, y)
    dfdx = grad(f3)(x, y)
    return dfdx

l = 2.0
l1 = [1.0, 2.0, 3.0, 4.0]
l2 = [5.0, 6.0, 7.0, 8.0]
n1 = np.array([1.0, 2.0, 3.0, 4.0])
n2 = np.array([5.0, 6.0, 7.0, 8.0])
n3 = np.array([1, 2, 3, 4])
n4 = np.array([5, 6, 7, 8])
#print(f1(l1,l2))

# list test
#out = main(l1,l2)
#out = main(3.0, 2.0)
#out = main(l1, l) # f4
out = main(l1,l) # f3
#out = main(l,l1) # f2

# array test
# out = main(n3,n4) # f3
print(out)
