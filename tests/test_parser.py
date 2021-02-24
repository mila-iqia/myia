from myia.parser import parse
from myia.ir.print import str_graph

def test_simple():
    def f(x):
        return x

    print(str_graph(parse(f)))
    
