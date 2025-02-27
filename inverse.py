import numpy as np


def get_sel(i, M, dim):
    return [x for x in range(M.shape[dim]) if x != i]


def determinant(M):
    if M.shape[-2] != M.shape[-1]:
        raise np.linalg.LinAlgError("Must be square matrix")

    if M.shape == (2,2):
        return M[0,0] * M[1,1] - M[1,0] * M[0,1]

    else:
        det = 0
        for i in range(M.shape[-1]):
            sel = get_sel(i, M, -1)
            val = M[0, i] * determinant(M[1:, sel])

            if i % 2 == 1:
                val = -val
            det += val

        return det


def det_matrix(M):
    N = np.zeros(M.shape)

    if M.shape[-2] != M.shape[-1]:
        raise np.linalg.LinAlgError("Must be square matrix")

    for i in range(M.shape[-2]):
        for j in range(M.shape[-1]):
            sel_row = get_sel(i, M, -2)
            sel_col = get_sel(j, M, -1)
            M_star = M[sel_row, :]
            val = determinant(M_star[:, sel_col])

            if (i + j) % 2 == 1:
                val = -val

            N[i, j] = val

    return N


def inverse(M):

    if M.shape[-2] != M.shape[-1]:
        raise np.linalg.LinAlgError("Must be square matrix")

    return (1 / determinant(M)) * det_matrix(M).T
