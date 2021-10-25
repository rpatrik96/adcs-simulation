import numpy
import numpy as np


def vec2diad(vec):
    return np.outer(vec, vec)


def cross_prod_mat(vec):
    identity = np.eye(3)
    return np.cross(identity, vec.reshape(-1, ))


def vector_angle(a, b, normalized=False):
    if normalized:
        return np.arccos(np.clip(np.dot(a, b), -1., 1.))
    else:
        return np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))


def xyz2ra_dec(vec):
    ra = np.arctan2(vec[1], vec[0])
    sin_ra = np.sin(ra)
    cos_ra = np.cos(ra)

    # two branches to increase accuracy
    # as if abs(sin(x)) is close to 1, its slope is small
    if np.abs(sin_ra) < .8:
        dec = np.arctan2(vec[2] * sin_ra, vec[1])
    else:
        dec = np.arctan2(vec[2] * cos_ra, vec[0])

    return ra, dec


def matrix_minor(mat, i, j):
    return np.delete(np.delete(mat, i, axis=0), j, axis=1)


def adjoint_matrix(mat, symmetric=False):
    adjoint = np.zeros_like(mat)

    num_rows, num_columns = adjoint.shape[0], adjoint.shape[1]
    for i in range(num_rows):
        for j in range(num_columns):
            if not symmetric:
                # transpose is included
                adjoint[j, i] = ((-1) ** (i + j)) * np.linalg.det(matrix_minor(mat, i, j))
            else:
                if j < i:
                    continue
                adjoint[j, i] = adjoint[i, j] = ((-1) ** (i + j)) * np.linalg.det(matrix_minor(mat, i, j))

    return adjoint


def chol_update(L: np.ndarray, x: np.ndarray, weight: float = 1.):
    """
    Rank-1 Cholesky-update of the Cholesky-factor (if x is a matrix, x_num_col updates are
    carried out)

    :param L: Cholesky factor (triangular matrix)
    :param x: update vector (if matrix it is also handled)
    :param weight: sign of the update (-: downdate/+:update)
    :return:
    """

    # todo: consider this version for efficiency: https://christian-igel.github.io/paper/AMERCMAUfES.pdf, Alg. 3.1

    def chol_vec_update(L: np.ndarray, x: np.ndarray, vec_dim: int, weight: float = 1.):
        """

        Rank-1 Cholesky-update of the Cholesky-factor
        :param L: Cholesky factor (triangular matrix)
        :param x: update vector
        :param vec_dim: dimension of the vector
        :param weight: weight of the update (-: downdate/+:update)
        :return:
        """
        for i in range(vec_dim):
            if L[i, i] ** 2 + weight * x[i] ** 2 < 0:
                # breakpoint()
                # raise ValueError(f"negative value\n, {L}")
                print(f"negative value\n, {L}")
            pass
            r = np.sqrt(L[i, i] ** 2 + weight * (x[i] ** 2))
            inv_diag_i = 1 / L[i, i]
            c = r * inv_diag_i
            s = x[i] * inv_diag_i
            L[i, i] = r
            L[i, i + 1:vec_dim] = (L[i, i + 1:vec_dim] + weight * s * x[i + 1:vec_dim]) / c
            x[i + 1:vec_dim] = c * x[i + 1:vec_dim] - s * L[i, i + 1:vec_dim]

    if len(x.shape) == 1:
        chol_vec_update(L, x, x.size, weight)
    else:
        vec_dim = x.shape[0]
        num_updates = x.shape[1]
        for i in range(num_updates):
            chol_vec_update(L, x[:, i], vec_dim, weight)

    return L


def separated_chol_update(pos_sigma_points, neg_sigma_points, noise_cov_sqrt):
    S_hat = np.linalg.qr(np.concatenate((pos_sigma_points, noise_cov_sqrt), axis=1).T, mode="r").T
    if neg_sigma_points.shape[-1] is not 0:
        S_hat = chol_update(S_hat, neg_sigma_points, -1.)
    return S_hat