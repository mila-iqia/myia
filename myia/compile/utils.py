from ..ir import is_apply


def get_outputs(lst, uses, seen):
    outputs = []
    for n in lst:
        if is_apply(n) and any(u[0] not in seen for u in uses[n]):
            outputs.append(n)
    return outputs
