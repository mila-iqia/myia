
from buche import buche
from ..util import Singleton
from ..stx import top as about_top, is_builtin, is_global, Symbol, TMP
import json
import os


class NO_VALUE(Singleton):
    pass


NO_VALUE = NO_VALUE()   # type: ignore


_css_path = f'{os.path.dirname(__file__)}/graph.css'
_css = open(_css_path).read()


# Roles
# FN: unique, function used to compute node
# IN(idx): unique, idxth input to computation for the node


class FN(Singleton):
    """
    Edge label for the FN relation between two nodes.
    """
    pass


class IN:
    """
    Edge label for the IN(i) relation between two nodes.
    """
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'IN({self.index})'

    def __hash__(self):
        return hash(IN) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


class OUT(Singleton):
    """
    Edge label for the OUT relation between a graph and a node.
    """
    pass


FN = FN()    # type: ignore
OUT = OUT()  # type: ignore


def commit(ops):
    for op, n1, n2, r in ops:
        n1.process_operation(op, n2, r)


class AbstractNode:
    def process_operation(self, op, node, role):
        method = getattr(self, f'process_operation_{op}')
        method(node, role)

    def commit(self, ops):
        commit(ops)

    def set_succ(self, role, node):
        """
        Create an edge toward node with given role
        """
        return commit(self.set_succ_operations(role, node))


class IRNode(AbstractNode):
    """
    Node in the intermediate representation. It can represent:

    * A computation:
      - fn != None, inputs == list of inputs, value == NO_VALUE
    * A numeric/string/etc. constant
      - fn == None, value == the constant
    * A builtin function like add or multiply
      - fn == None, value == the Symbol for the builtin
    * A pointer to another Myia function
      - fn == None, value == an IRGraph
    * An input
      - fn == None, value == NO_VALUE

    Attributes:
        graph: Parent IRGraph for this node. This should be None for
            constant nodes.
        tag: Symbol representing the node name.
        fn: Points to an IRNode providing the function to call.
        inputs: List of IRNode arguments.
        users: Set of incoming edges. Each edge is a (role, node)
            tuple where role is FN or IN(i)
        value: Value taken by this node.
        inferred: TODO
        about: Tracks source code location and sequence of
            transformations and optimizations for this node.
    """
    def __init__(self, graph, tag, value=NO_VALUE):
        # Graph the node belongs to
        self.graph = graph
        # Node name (Symbol instance)
        self.tag = tag
        # Outgoing edges
        self.fn = None
        self.inputs = []
        # Incoming edges as (role, node) tuples
        self.users = set()
        # Value this node will take, if it can be determined
        self.value = value
        # Information inferred about this node (type, shape, etc.)
        self.inferred = {}
        # Optimization/source code trace
        self.about = about_top()

    def is_input(self):
        return self.fn is None and self.value is NO_VALUE

    def is_computation(self):
        return self.fn is not None

    def is_constant(self):
        return self.value is not NO_VALUE

    def is_builtin(self):
        return is_builtin(self.value)

    def is_global(self):
        return is_global(self.value)

    def is_graph(self):
        return isinstance(self.value, IRGraph)

    def edges(self):
        """
        List of nodes that this node depends on.
        """
        succ = [(FN, self.fn)] + \
            [(IN(i), inp) for i, inp in enumerate(self.inputs)]
        return {(r, s) for r, s in succ if s}

    def successors(self):
        succ = [self.fn] + list(self.inputs)
        return {s for s in succ if s}

    def predecessors(self):
        return {s for _, s in self.users}

    def sexp(self):
        """
        If this node is an application, return:

            (n.fn, *n.inputs)

        Otherwise, return None.
        """
        if self.fn is None:
            return None
        else:
            return (self.fn,) + tuple(self.inputs)

    def set_sexp(self, fn, inputs):
        """
        Make this node an application of fn on the specified inputs.
        fn and the inputs must be IRNode instances.
        """
        return commit(self.set_sexp_operations(fn, inputs))

    def redirect(self, new_node):
        """
        Transfer every user of this node to new_node.
        """
        return commit(self.redirect_operations(new_node,))

    def subsume(self, node):
        """
        Transfer every user of the given node to this one.
        """
        return node.redirect(self)

    # The following methods return a list of atomic "operations" to
    # execute in order to perform the task. Atomic operations are
    # ('link', from, to, role) and ('unlink', from, to, node)

    def set_succ_operations(self, role, node):
        assert isinstance(node, IRNode) or node is None
        if role is FN:
            unl = self.fn
        elif isinstance(role, IN):
            unl = self.inputs[role.index]
        else:
            raise KeyError(f'Invalid role: {role}')
        rval = []
        if unl:
            if unl == node:
                # Nothing to do because the new successor is the
                # same as the old one.
                return []
            else:
                rval.append(('unlink', self, unl, role))
        if node is not None:
            rval.append(('link', self, node, role))
        return rval

    def set_sexp_operations(self, fn, inputs):
        rval = self.set_succ_operations(FN, fn)
        if fn:
            for i, inp in enumerate(self.inputs):
                if inp is not None:
                    rval.append(('unlink', self, inp, IN(i)))
            for i, inp in enumerate(inputs):
                if inp is not None:
                    rval.append(('link', self, inp, IN(i)))
        return rval

    def redirect_operations(self, node):
        rval = []
        for role, n in set(self.users):
            rval += n.set_succ_operations(role, node)
        return rval

    def trim_inputs(self):
        while self.inputs and self.inputs[-1] is None:
            self.inputs.pop()

    def process_operation_link(self, node, role):
        if role is FN:
            assert self.fn is None
            self.fn = node
        elif isinstance(role, IN):
            idx = role.index
            nin = len(self.inputs)
            if nin <= idx:
                self.inputs += [None for _ in range(idx - nin + 1)]
            assert self.inputs[idx] is None
            self.inputs[idx] = node
        node.users.add((role, self))

    def process_operation_unlink(self, node, role):
        if role is FN:
            assert self.fn is node
            self.fn = None
        elif isinstance(role, IN):
            idx = role.index
            nin = len(self.inputs)
            assert self.inputs[idx] is node
            self.inputs[idx] = None
            self.trim_inputs()
        node.users.remove((role, self))

    def __getitem__(self, role):
        if role is FN:
            return self.fn
        elif isinstance(role, IN):
            return self.inputs[role.index]
        else:
            raise KeyError(f'Invalid role: {role}')

    def __setitem__(self, role, node):
        self.set_succ(role, node)

    def __hrepr__(self, H, hrepr):
        if self.value is NO_VALUE:
            return hrepr(self.tag)
        else:
            return hrepr(self.value)


class IRGraph(AbstractNode):
    """
    Graph with inputs and an output. Represents a Myia function or
    a closure.

    Attributes:
        parent: The IRGraph for the parent function, if this graph
            represents a closure. Otherwise, None.
        tag: A Symbol representing the name of this graph.
        gen: A GenSym instance to generate new tags within this
            graph.
        inputs: A tuple of input IRNodes for this graph.
        output: The IRNode representing the output of this graph.
    """
    def __init__(self, parent, tag, gen):
        self.parent = parent
        self.tag = tag
        self.inputs = []
        self._output = None
        self.gen = gen
        # Legacy
        self.lbda = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, out):
        if self._output:
            self.process_operation('unlink', self._output, OUT)
        self.process_operation('link', out, OUT)

    def set_succ_operations(self, role, node):
        if role is OUT:
            lnk = ('link', self, node, role)
            if self._output:
                return [('unlink', self, self._output, role), lnk]
            else:
                return [lnk]
        else:
            raise ValueError(f'Invalid outgoing edge type for IRGraph: {role}')

    def process_operation_link(self, node, role):
        if role is OUT:
            assert self._output is None
            self._output = node
        node.users.add((role, self))

    def process_operation_unlink(self, node, role):
        if role is OUT:
            assert self._output is node
            self._output = None
        node.users.remove((role, self))

    def dup(self, g=None, no_mangle=False):
        """
        Duplicate this graph, optionally setting g as the parent of
        every node in the graph.

        Return the new graph (or g), a list of inputs, and the output
        node.
        """
        set_io = g is None
        if not g:
            g = IRGraph(self.parent, self.tag, self.gen)
        mapping = {}
        for node in self.inputs + tuple(self.iternodes()):
            if no_mangle:
                mapping[node] = IRNode(g, node.tag, node.value)
            else:
                mapping[node] = IRNode(g, g.gen(node.tag, '+'), node.value)
        for n1, n2 in mapping.items():
            sexp = n1.sexp()
            if sexp:
                f, *args = sexp
                f2 = mapping.get(f, f)
                args2 = [mapping.get(a, a) for a in args]
                n2.set_sexp(f2, args2)
        output = mapping.get(self.output, self.output)
        inputs = tuple(mapping[i] for i in self.inputs)
        if set_io:
            g.output = output
            g.inputs = inputs
        return g, inputs, output

    def toposort(self):
        if not self.output.is_computation():
            return []
        pred = {}
        ready = {self.output}
        processed = set()
        results = []
        while ready:
            node = ready.pop()
            if not node.is_computation():
                continue
            if node in processed:
                raise Exception('Cannot toposort: cycle detected.')
            results.append(node)
            processed.add(node)
            for succ in node.successors():
                if succ not in pred:
                    pred[succ] = {n for _, n in succ.users}
                d = pred[succ]
                d.remove(node)
                if len(d) == 0:
                    ready.add(succ)
                    pred[succ] = None
        results.reverse()
        return results

    def iternodes(self, boundary=False):
        # Basic BFS from output node
        to_visit = {self.output}
        seen = set()
        while to_visit:
            node = to_visit.pop()
            if not node or node in seen:
                continue
            if node.graph is not self:
                if boundary:
                    yield node
                continue
            yield node
            seen.add(node)
            to_visit.add(node.fn)
            for inp in node.inputs:
                to_visit.add(inp)

    def iterboundary(self):
        return self.iternodes(True)

    def iterparents(self, stop=None):
        g = self
        while g and g is not stop:
            yield g
            g = g.parent

    @classmethod
    def __hrepr_resources__(cls, H):
        return H.bucheRequire(name='cytoscape')

    def __hrepr__(self, H, hrepr):
        rval = H.cytoscapeGraph(H.style(_css))
        options = {
            'layout': {
                'name': 'dagre',
                'rankDir': 'TB'
            }
        }
        rval = rval(H.options(json.dumps(options)))

        opts = {
            'duplicate_constants': hrepr.config.duplicate_constants,
            'function_in_node': hrepr.config.function_in_node,
            'follow_references': hrepr.config.follow_references
        }
        nodes_data, edges_data = GraphPrinter({self}, **opts).process()
        for elem in nodes_data + edges_data:
            rval = rval(H.element(json.dumps(elem)))
        return rval


class GraphPrinter:
    """
    Helper class to print Myia graphs.

    Arguments:
        entry_points: A collection of graphs to print.
        duplicate_constants: If True, each use of a constant will
            be shown as a different node.
        function_in_node: If True, applications of a known function
            will display the function's name in the node like this:
            "node_name:function_name". If False, the function will
            be a separate constant node, with a "F" edge pointing to
            it.
        follow_references: If True, graphs encountered while walking
            the initial graphs will also be processed.
    """
    def __init__(self,
                 entry_points,
                 duplicate_constants=True,
                 function_in_node=True,
                 follow_references=True):
        # Graphs left to process
        self.graphs = set(entry_points)
        self.duplicate_constants = duplicate_constants
        self.function_in_node = function_in_node
        self.follow_references = follow_references
        # Nodes left to process
        self.pool = set()
        # ID system for the nodes that will be sent to buche
        self.currid = 0
        self.ids = {}
        # Nodes and edges are accumulated in these lists
        self.nodes = []
        self.edges = []

    def next_id(self):
        self.currid += 1
        return f'X{self.currid}'

    def register(self, obj):
        if not self.should_dup(obj) and obj in self.ids:
            return False, self.ids[obj]
        id = self.next_id()
        self.ids[obj] = xzz = id
        return True, id

    def const_fn(self, node):
        cond = self.function_in_node \
            and node.is_computation() \
            and node.fn.is_constant()
        if cond:
            return node.fn.tag

    def should_dup(self, node):
        return isinstance(node, IRNode) \
            and self.duplicate_constants \
            and node.value is not NO_VALUE

    def add_graph(self, g):
        new, id = self.register(g)
        if new:
            self.nodes.append({'data': {'id': id, 'label': str(g.tag)},
                               'classes': 'function'})
        return id

    def label(self, node):
        if node.is_constant() and not node.is_graph():
            return str(node.value)
        rels = ''
        tag = node.tag
        while tag.relation:
            if tag.relation != '+':
                rels = rels + tag.relation
            tag = tag.label
        lbl = tag.label
        if '/out' in lbl or '/in' in lbl or lbl == TMP:
            lbl = "•"
        lbl = rels + lbl
        cfn = self.const_fn(node)
        if cfn:
            if lbl == "•":
                lbl = f'{cfn}'
            else:
                lbl = f'{lbl}←{cfn}'
        return lbl

    def add_node(self, node, g=None):

        new, id = self.register(node)
        if not new:
            return id

        if not g:
            g = node.graph

        if node.is_graph():
            if self.follow_references:
                self.graphs.add(node.value)

        if node.graph is None:
            cl = 'constant'
        elif node is g.output and node.is_computation():
            cl = 'output'
        elif node in g.inputs:
            cl = 'input'
        else:
            cl = 'intermediate'

        lbl = self.label(node)

        data = {'id': id, 'label': lbl}
        if g:
            data['parent'] = self.add_graph(g)
        self.nodes.append({'data': data, 'classes': cl})
        self.pool.add(node)
        return id

    def process_graph(self, g):
        for inp in g.inputs:
            self.add_node(inp)
        self.add_node(g.output)

        if not g.output.is_computation():
            oid = self.next_id()
            self.nodes.append({'data': {'id': oid,
                                        'label': '',
                                        'parent': self.add_graph(g)},
                               'classes': 'const_output'})
            self.edges.append({'data': {
                'id': self.next_id(),
                'label': '',
                'source': self.ids[g.output],
                'target': oid
            }})

        while self.pool:
            node = self.pool.pop()
            if self.const_fn(node):
                if self.follow_references \
                        and isinstance(node.fn.value, IRGraph):
                    self.graphs.add(node.fn.value)
                edges = []
            else:
                edges = [(node, FN, node.fn)] \
                    if node.is_computation() \
                    else []
            edges += [(node, IN(i), inp)
                      for i, inp in enumerate(node.inputs) or []]
            for edge in edges:
                src, role, dest = edge
                if role is FN:
                    lbl = 'F'
                else:
                    lbl = str(role.index)
                dest_id = self.add_node(dest, self.should_dup(dest) and g)
                data = {
                    'id': self.next_id(),
                    'label': lbl,
                    'source': dest_id,
                    'target': self.ids[src]
                }
                self.edges.append({'data': data})

    def process(self):
        while self.graphs:
            g = self.graphs.pop()
            self.process_graph(g)
        return self.nodes, self.edges
