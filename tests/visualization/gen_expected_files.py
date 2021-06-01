"""Generate expected file f*_***.html from tested functions in test_graph_printer.py"""
import itertools
import os

import test_graph_printer
from hrepr import hrepr

from myia.ir.visualization import GraphPrinter
from myia.parser import parse
from myia.utils.info import enable_debug


def _generate_expected_files(*functions):
    for (
        function,
        show_fn_constants,
        show_args,
        link,
    ) in itertools.product(functions, (0, 1), (0, 1), (0, 1)):
        with enable_debug():
            graph = parse(function)
        gp = GraphPrinter(
            graph,
            show_fn_constants=bool(show_fn_constants),
            show_args=bool(show_args),
            link_fn_graphs=bool(link),
            link_inp_graphs=bool(link),
        )
        html = hrepr.page(gp)
        output_name = (
            f"{function.__name__}_{show_fn_constants}{show_args}{link}.html"
        )
        with open(
            os.path.join(os.path.dirname(__file__), output_name),
            "w",
        ) as file:
            file.write(str(html))
        print("Generated", output_name)


def main():
    functions = (
        test_graph_printer.f0,
        test_graph_printer.f1,
    )
    _generate_expected_files(*functions)


if __name__ == "__main__":
    main()
