from ..graph_utils import always_include, dfs
from ..ir.utils import succ_deep, succ_deeper, succ_incoming
from ..ir.anf import Constant
from ..ir.anf import Graph

def get_text(node):
    name = node.__class__.__name__ + ':' + str(node)
    if node.graph:
        name = str(node.graph) + ":" + name 
    return name
def get_label(node):
    label = '['+node.__class__.__name__+']' 
    if(isinstance(node, Constant)):
        label += node.value.__class__.__name__ + '\n'
    label += str(node)
    return label
def getGraphViz(g: Graph):
    from graphviz import Digraph as GraphV
    gptop = GraphV(name='G')
    #label = short_labeler
    subGraphs = {}
    for node in dfs(g.return_, succ_deep):
        #print('node:', get_text(node), node)
        gp = gptop
        if node.graph:
            name = 'cluster_'+str(node.graph)
            if not name in subGraphs:
                gpsub = GraphV(name=name)
                gpsub.attr(label=str(node.graph))
                subGraphs[name] = gpsub
            gp = subGraphs[name]
        gp.node(get_text(node), get_label(node))
    for name in subGraphs:
        gptop.subgraph(subGraphs[name])
    for node in dfs(g.return_, succ_deep):
        for income in succ_incoming(node):
            gptop.body.append('"%s" -> "%s"' %(get_text(income), get_text(node)))       
    return gptop
