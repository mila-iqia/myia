
from myia.api import myia
from myia.composite import grad
import numpy as np

def f1(x, y):
   y = y[::]
   return f2(x, y)
#   return y[0]                                                                                                        


def f2(x, y):
#   a = y[0] * y[1] * x                                                                                                
#   a = x * y[0]                                                                                                       
   a = y[0]
   return a

@myia
def main(x, y):
    dfdx = grad(f1)(x,y)
    return dfdx

x = np.array([4.0, 2.0])                                                                                               
y = np.array([4.0, 2.0])
# f3 and f4                                                                                                            
#x = [5.0, 6.0, 7.0]                                                                                                   
#y = 10.0                                                                                                              
print(main(x, y))