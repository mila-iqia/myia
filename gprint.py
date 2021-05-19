from snektalk import pastevar

from myia.ir.visualization import GraphPrinter
from myia.parser import parse
from myia.utils.info import enable_debug


def main():
    def f(x):
        while x:
            x = x - 1
        return x

    with enable_debug():
        graph = parse(f)
    print(GraphPrinter(graph, on_node=pastevar))
    # from hrepr import hrepr
    # hrepr.page(graph, file="output.html")


if __name__ == "__main__":
    main()
