from copy import deepcopy
from enum import Enum

import numpy as np

from alg_utils import adjoint_matrix
from quaternion import Quaternion


class ESOQ2(object):
    class SeqRotAxis(Enum):
        """
        Enum class for describing the by sequential rotation affected axis
        """
        NONE = 0
        X = 1
        Y = 2
        Z = 3

    def __init__(self) -> None:
        super().__init__()

        self.q_opt = Quaternion.from_axis_angle()
        self.min_num_stars = 2

        self.cond_fro = 0
        self.cond_inf = 0
        self.cond_inf_neg = 0
        self.lambda_max = 0
        self.q_norm = 1.

    def step(self, observations: np.ndarray, reference: np.ndarray) -> None:
        """

        :param observations: array of the unit vectors in body frame
        :param reference: array of the unit vectors in the reference frame
        :return:
        """

        num_stars = len(observations)

        if num_stars >= self.min_num_stars:
            alpha = (np.ones(num_stars) / num_stars).reshape(-1, 1)

            B = np.matmul((alpha * observations).transpose(), reference)  # attitude profile matrix
            B = self._calc_seq_rot_mode(B)

            z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]]).reshape((3, 1))

            lambda_max = self._newton_raphson(alpha.sum(), B, z, 1, num_stars == 2)

            # two notations for S are used, here denoted by SS
            # SS from Second Estimator of the Optimal Quaternion
            # http://www.malcolmdshuster.com/FC_Mortari_ESOQ2_JGCD_1997_AIAA.pdf
            SS = B + B.transpose() - (B.trace() + lambda_max) * np.eye(3)
            M = (B.trace() - lambda_max) * SS - np.outer(z, z)

            # rotation vector
            e1 = np.cross(M[1, :], M[2, :])
            e2 = np.cross(M[2, :], M[0, :])
            e3 = np.cross(M[0, :], M[1, :])

            # select the rotation vector based on the norm of the diagonal elements
            ek = [e1, e2, e3][np.argmax([e1[0], e2[1], e3[2]])]

            # optimal quaternion
            self.q_norm = np.linalg.norm((Quaternion.from_sw(np.dot(z.reshape(-1), ek), ((lambda_max - B.trace()) * ek).reshape(-1, ), False)).q)
            self.q_opt = Quaternion.from_sw(np.dot(z.reshape(-1), ek), ((lambda_max - B.trace()) * ek).reshape(-1, ))
            self._seq_rot_quat()
        else:
            print(f"Less than min_num_stars={self.min_num_stars} found!")

    def _newton_raphson(self, lambda_0: float, B: np.ndarray, z: np.ndarray, num_iter: int = 0,
                        two_observations=False) -> float:
        """

        :param lambda_0: initial guess for the maximal eigenvalue
        :param B: attitude profile matrix
        :param z: weighed sum of the cross product of reference and observation vectors
        :param num_iter: number of iterations for the Newton-Raphson approximation algorithm
        :param two_observations: boolean flag to indicate whether the deterministic formula in case of two observations can be used
        :return: the maximal eigenvalue of the K matrix
        """

        lambda_i = lambda_0

        if num_iter == 0:
            pass
        else:
            """Calculate coefficients"""
            # this part is from Three-Axis Attitude Determination from Vector Observations
            # https://doi.org/10.2514/3.19717
            # Eq. 70

            S = B + B.transpose()
            S_times_z = np.matmul(S, z)

            sigma = .5 * S.trace()
            kappa = adjoint_matrix(S).trace()
            delta = np.linalg.det(S)

            aa = sigma ** 2 - kappa
            bb = sigma ** 2 + np.dot(z.transpose(), z)

            c = -(np.matmul(z.transpose(), S_times_z) + delta)  # for c no tmp exist
            dd = np.matmul(z.transpose(), np.matmul(S, S_times_z))

            b = - (aa + bb)
            d = (aa * bb - c * sigma - dd)

            if not two_observations:
                """Newton-Raphson iteration"""
                for i in range(num_iter):
                    nominator = lambda_i ** 4 + b * lambda_i ** 2 + c * lambda_i + d
                    denominator = 4 * lambda_i ** 3 + 2 * b * lambda_i + c
                    lambda_i -= (nominator / denominator)
            else:
                d_flatten_ = d.flatten()[0]
                c_flatten_ = c.flatten()[0]
                b_flatten_ = b.flatten()[0]
                char_mat = np.array([[0, 0, 0, -d_flatten_],
                                     [1, 0, 0, -c_flatten_],
                                     [0, 1, 0, -b_flatten_],
                                     [0, 0, 1, 0]])

                # char_mat_inv = np.array([[-c_flatten_/d_flatten_, 1, 0, 0],
                #                      [-b_flatten_/d_flatten_, 0, 1, 0],
                #                      [0, 0, 0, 1],
                #                      [-1/d_flatten_, 0, 0, 0]])
                self.cond_fro = np.linalg.cond(char_mat, "fro")
                self.cond_inf = np.linalg.cond(char_mat, np.inf)
                self.cond_inf_neg = np.linalg.cond(char_mat, -np.inf)

                """Deterministic formula for two observations"""
                two_sqrt_d = 2 * np.sqrt(d)
                lambda_i = .5 * (np.sqrt(two_sqrt_d - b) + np.sqrt(-two_sqrt_d - b))
                lambda_i = lambda_i.flatten()[0]

                self.lambda_max = lambda_i

        return lambda_i

    def _calc_seq_rot_mode(self, B: np.ndarray, phi_min: float = 0.01) -> np.ndarray:
        """
        Calculates the sequential rotation based on the attitude profile matrix
        according to "Second Estimator of the Optimal Quaternion" from
        http://www.malcolmdshuster.com/FC_Mortari_ESOQ2_JGCD_1997_AIAA.pdf (Table 1)

        :param B: attitude profile matrix
        :param phi_min: threshold for the sequential rotation
        :return: the rotated (if necessary) B matrix
        """

        cos_min = np.cos(np.deg2rad(phi_min))

        # shortcut for main digaonal elements
        b11 = B[0, 0]
        b22 = B[1, 1]
        b33 = B[2, 2]

        """Calculate rotation mode"""
        if b11 + b22 + b33 < cos_min:
            self.seq_rot_mode = self.SeqRotAxis.NONE

        elif b11 - b22 - b33 < cos_min:
            self.seq_rot_mode = self.SeqRotAxis.X
            B = np.stack([B[:, 0], -B[:, 1], -B[:, 2]], axis=-1)

        elif b22 - b11 - b33 < cos_min:
            self.seq_rot_mode = self.SeqRotAxis.Y
            B = np.stack([-B[:, 0], B[:, 1], -B[:, 2]], axis=-1)

        elif b33 - b11 - b22 < cos_min:
            self.seq_rot_mode = self.SeqRotAxis.Z
            B = np.stack([-B[:, 0], B[:, 1], -B[:, 2]], axis=-1)

        return B

    def _seq_rot_quat(self) -> None:
        """
        Undoes the sequential rotation of self._calc_seq_rot_mode on the orientation quaternion,
        according to "Second Estimator of the Optimal Quaternion" from
        http://www.malcolmdshuster.com/FC_Mortari_ESOQ2_JGCD_1997_AIAA.pdf (Table 1)

        :return:
        """

        if self.seq_rot_mode != self.SeqRotAxis.NONE:
            q_tmp = deepcopy(self.q_opt)

        if self.seq_rot_mode == self.SeqRotAxis.X:
            # scalar
            self.q_opt.q[0] = -q_tmp.q[1]
            # vector
            self.q_opt.q[1] = q_tmp.q[0]
            self.q_opt.q[2] = -q_tmp.q[3]
            self.q_opt.q[3] = q_tmp.q[2]
        elif self.seq_rot_mode == self.SeqRotAxis.Y:
            # scalar
            self.q_opt.q[0] = -q_tmp.q[2]
            # vector
            self.q_opt.q[1] = q_tmp.q[3]
            self.q_opt.q[2] = q_tmp.q[0]
            self.q_opt.q[3] = -q_tmp.q[1]

        elif self.seq_rot_mode == self.SeqRotAxis.Z:
            # scalar
            self.q_opt.q[0] = -q_tmp.q[3]
            # vector
            self.q_opt.q[1] = -q_tmp.q[2]
            self.q_opt.q[2] = q_tmp.q[1]
            self.q_opt.q[3] = q_tmp.q[0]
