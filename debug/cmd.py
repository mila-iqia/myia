
from buche import buche

from myia.ir import ANFNode

from . import steps


def parse(o):
    from myia.parser import parse
    fn, = o.options['fns']
    buche(
        parse(fn),
        graph_width='95vw',
        graph_height='95vh',
        function_in_node=not o['--function-nodes'],
        graph_beautify=not o['--no-beautify'] and not o['--function-nodes'],
    )


def show(o):
    res = o.run(
        default=[steps.parse,
                 steps.resolve],
    )

    g = res['graph']

    def ttip(node):
        if isinstance(node, ANFNode):
            return node.abstract

    buche(
        g,
        graph_width='95vw',
        graph_height='95vh',
        node_tooltip=ttip,
        function_in_node=not o['--function-nodes'],
        graph_beautify=not o['--no-beautify'] and not o['--function-nodes'],
    )

    return res
