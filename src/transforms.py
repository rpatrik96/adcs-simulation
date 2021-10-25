import numpy as np

import alg_utils


class Rotation(object):
    def __init__(self, axis, angle, matrix):
        self.matrix = matrix
        self.axis = axis
        self.angle = angle

    @classmethod
    def from_axis_angle(cls, axis, angle):
        # dimensionality check
        if len(axis.squeeze().shape) != 1 and axis.squeeze().shape != 3:
            raise TypeError("axis should be a 3D vector, got dimension ", axis.squeeze().shape)

        # normalize axis
        if np.abs(np.linalg.norm(axis) - 1.0) > 1e-5:
            axis /= np.linalg.norm(axis)

        matrix = cls.rodrigues2rotmat(angle, axis)

        return cls(axis, angle, matrix)

    @classmethod
    def from_matrix(cls, matrix=np.eye(3)):

        # dimensionality check
        if matrix.squeeze().shape != (3, 3):
            raise TypeError("matrix should be a 3x3 matrix, got dimension ", matrix.squeeze().shape)
        axis, angle = cls.rotmat2rodrigues(matrix.astype(np.float64))

        return cls(axis, angle, matrix)

    @staticmethod
    def rodrigues2rotmat(angle, axis):
        identity = np.eye(3)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        return cos_angle * identity + (1 - cos_angle) * alg_utils.vec2diad(
            axis) + sin_angle * alg_utils.cross_prod_mat(axis)

    @staticmethod
    def rotmat2rodrigues(matrix):
        cos_angle = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2] - 1.) / 2.

        mat_21_12 = matrix[2, 1] - matrix[1, 2]
        mat_02_20 = matrix[0, 2] - matrix[2, 0]
        mat_10_01 = matrix[1, 0] - matrix[0, 1]

        sin_angle = np.sqrt(np.square(mat_21_12) + np.square(mat_02_20) + np.square(mat_10_01)) / 2.

        angle = np.arctan2(sin_angle, cos_angle)

        if np.abs(angle) > 1e-8:
            one_minus_cos_inv_sqrt = np.sqrt(1. / (1. - cos_angle))

            axis_x = one_minus_cos_inv_sqrt * np.sqrt(matrix[0, 0] - cos_angle) * np.sign(mat_21_12)
            axis_y = one_minus_cos_inv_sqrt * np.sqrt(matrix[1, 1] - cos_angle) * np.sign(mat_02_20)
            axis_z = one_minus_cos_inv_sqrt * np.sqrt(matrix[2, 2] - cos_angle) * np.sign(mat_10_01)

        else:
            axis_x = 1.
            axis_y = 0.
            axis_z = 0.

        return np.array([axis_x, axis_y, axis_z]), angle

    def transpose(self):
        return self.__class__.from_matrix(self.matrix.transpose())

    def vec_mult(self, vec):
        return self.matrix @ vec

    def __matmul__(self, other):
        return self.__class__.from_matrix(self.matrix @ other.matrix)


class HomogeneousTransform(object):
    def __init__(self, rot: Rotation = Rotation.from_matrix(np.eye(3)), trans: np.ndarray = np.array([0., 0., 0.])):
        self.rotation = rot
        self.translation = trans.astype(np.float64)

    @property
    def matrix(self):
        m = np.zeros((4, 4))
        m[0:3, 0:3] = self.rotation.matrix
        m[0:3, 3] = self.translation
        m[3, 3] = 1.

        return m

    def transform(self, transform):
        rot = self.rotation @ transform.rotation
        tr = self.rotation.matrix @ transform.translation + self.translation

        return self.__class__(rot, tr)

    def translate(self, vec):
        self.translation += vec.astype(np.float64)

    def rotate(self, rotation):
        self.rotation = self.rotation @ rotation

    def inverse(self):
        rot_transpose = self.rotation.transpose()

        return self.__class__(rot_transpose, -rot_transpose.vec_mult(self.translation))
