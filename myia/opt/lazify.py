
# from collections import defaultdict

# from ..ir import succ_incoming, freevars_boundary, Graph
# from ..graph_utils import dfs


# def lazify(g):
#     rval = Graph()


#     return g


# # def lazify(g, roots):
# #     """Replace each specified node of g by a thunk."""
# #     nodemap = defauldict(set)
# #     for root in roots:
# #         for node in dfs(root, succ_incoming, freevars_boundary):
# #             nodemap[node].add(root)

# #     groups = defaultdict(set)
# #     for node, roots in nodemap.items():
# #         groups[frozenset(roots)].add(node)

# #     pass



# # def toast(x, y, z):
# #     q = a * a * a  # 1, 2, 3

# #     a = x * y + q  # 1, 3
# #     b = y * z + q  # 1, 2
# #     c = x * y  # 2, 3, 4

# #     aa = a + b  # 1
# #     bb = b + c  # 2
# #     cc = a + c  # 3

# #     return aa, bb, cc, c



# # def toast():
# #     x = f() # 1, 3
# #     y = g() # 1, 2
# #     z = h() + q # 2, 3

# #     a = x * y # 1
# #     b = y * z + q # 2
# #     c = x * z + q # 3

# #     return a, b, c
