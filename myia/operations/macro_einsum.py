"""Implementation of the 'einsum' macro."""

from ..lib import macro, InferenceError
from opt_einsum import contract_expression, contract
from . import primitives as P


def _tensordot(g, a, b, *, axes):
    axes_a, axes_b = axes
    axes_a = list(axes_a)
    axes_b = list(axes_b)

    as_ = a[1]
    nda = len(as_)
    bs = b[1]
    ndb = len(bs)

    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = tuple(notin + axes_a)
    N1 = 1
    for axis in notin:
        N1 *= as_[axis]
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (N1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = tuple(axes_b + notin)
    N1 = 1
    for axis in notin:
        N1 *= bs[axis]
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, N1)
    oldb = [bs[axis] for axis in notin]

    at = g.apply(P.reshape, g.apply(P.transpose, a[0], newaxes_a), newshape_a)
    bt = g.apply(P.reshape, g.apply(P.transpose, b[0], newaxes_b), newshape_b)
    res = g.apply(P.dot, at, bt)
    res_shp = tuple(olda + oldb)

    return (g.apply(P.reshape, res, res_shp), res_shp)


def _reduce_transpose(g, input_spec, output_spec, arg):
    if input_spec == output_spec:
        return arg

    out_idx = set(output_spec)
    reduce_axes = [i for i, c in enumerate(input_spec) if c not in out_idx]
    shp = arg[1]

    target_shape = [s if i not in reduce_axes else 1
                    for i, s in enumerate(shp)]
    res = g.apply(P.array_reduce, P.scalar_add, arg[0], tuple(target_shape))

    for i in reversed(reduce_axes):
        del target_shape[i]
    res = g.apply(P.reshape, res, tuple(target_shape))

    mid_spec = [c for c in input_spec if c in out_idx]
    transpose_pattern = tuple(mid_spec.index(c) for c in output_spec)
    res = g.apply(P.transpose, res, transpose_pattern)

    final_shape = tuple(target_shape[i] for i in transpose_pattern)
    return (res, final_shape)


def _elemwise(g, a_spec, b_spec, a, b):
    pass


def _simple_einsum(g, spec, *args):
    input_spec, output_spec = spec.split('->')

    if len(input_spec) == len(set(input_spec)):
        # Pure reduce/transpose
        assert len(args) == 1
        arg = args[0]
        return _reduce_transpose(g, input_spec, output_spec, arg)

    elif ',' in input_spec:
        input_list = input_spec.split(',')
        assert len(input_list) == 2
        if set(input_list[0]) == set(input_list[1]):
            # elemwise
            a, b = args
            av = a[0]
            bv = b[0]
            tmp_spec = input_list[0]
            if input_list[1] != tmp_spec:
                tt = tuple(tmp_spec.find(c) for c in input_list[1])
                bv = g.apply(P.transpose, bv, tt)
            res = (g.apply(P.array_map, P.scalar_mul, av, bv), a[1])
            return _reduce_transpose(g, input_list[0], output_spec, res)
        else:
            raise InferenceError(f"Can't support this pattern in einsum: {spec}")

    else:
        raise InferenceError(f"Can't support this pattern in einsum: {spec}")


@macro
async def einsum(info, r_spec, *r_args):
    """Macro implementation for 'einsum'."""
    _, *args = await info.abstracts()
    spec = await info.build(r_spec)
    shapes = tuple(a.xshape() for a in args)
    try:
        path = contract_expression(spec, *shapes,
                                   optimize='dynamic-programming')
    except ValueError as e:
        raise InferenceError(*e.args)
    g = info.graph

    nodes = [(a.node, sh) for a, sh in zip(r_args, shapes)]

    for contraction in path.contraction_list:
        inds, idx_rm, einsum_str, remaining, blas_flag = contraction

        tmp_nodes = [nodes.pop(x) for x in inds]
        if blas_flag:
            input_str, result_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = "".join(s for s in input_left + input_right
                                    if s not in idx_rm)

            left_pos, right_pos = [], []
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            new_node = _tensordot(g, *tmp_nodes,
                                  axes=(tuple(left_pos), tuple(right_pos)))

            if (tensor_result != result_index):
                transpose = tuple(map(tensor_result.index, result_index))
                new_node = (g.apply(P.transpose, new_node[0],
                                    tuple(transpose)),
                            tuple(new_node[1][i] for i in transpose))

        else:
            new_node = _simple_einsum(g, einsum_str, *tmp_nodes)

        nodes.append(new_node)

    return nodes[0][0]


__operation_defaults__ = {
    'name': 'einsum',
    'registered_name': 'einsum',
    'mapping': einsum,
    'python_implementation': contract,
}
