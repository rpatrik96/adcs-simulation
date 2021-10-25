import numpy as np

import alg_utils
from logger import Logger


class Quaternion(object):
    def __init__(self, q=np.array([1, 0, 0, 0]), normalize=True):

        # dimensionality check
        if len(q.squeeze().shape) != 1 and q.squeeze().shape != 4:
            raise TypeError("q should be 4D, got dimension ", q.squeeze().shape)

        # upcast for the norm division
        q = q.astype(np.float64)

        if normalize and np.abs(np.linalg.norm(q) - 1.) > 1e-8:
            # print(np.linalg.norm(q))

            if np.linalg.norm(q) < 1e-8:
                print("pech")
                # raise ValueError
                # breakpoint()
                pass

            q /= np.linalg.norm(q)

        self.q = q

    @classmethod
    def from_sw(cls, scalar, vector, normalize=True):
        # dimensionality check
        if len(vector.squeeze().shape) != 1 and vector.squeeze().shape != 3:
            raise TypeError("vector part should be a 3D, got dimension ", vector.squeeze().shape)
        return cls(np.hstack((scalar, vector)), normalize)

    @classmethod
    def from_vector(cls, vec: np.ndarray = np.array([1., 0., 0.])):
        return cls.from_sw(0, vec)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray = np.array([1., 0., 0.]), angle: float = 0.):
        axis_len = np.linalg.norm(axis)
        if np.abs(axis_len - 1.) > 1e-8:
            axis = axis / axis_len
        scalar = np.cos(angle / 2)
        vector = np.sin(angle / 2) * axis

        return cls.from_sw(scalar, vector, False)

    @classmethod
    def from_euler(cls, x_angle: float, y_angle: float, z_angle: float):
        """
        Quaternion from Euler-angles, using the Z-Y-X sequence (starting with a rotation around Z)
        :param x_angle: rotation angle around x-axis (in radians)
        :param y_angle: rotation angle around y-axis (in radians)
        :param z_angle: rotation angle around z-axis (in radians)
        :return:
        """
        # precompute
        cosz = np.cos(z_angle * 0.5)
        sinz = np.sin(z_angle * 0.5)
        cosy = np.cos(y_angle * 0.5)
        siny = np.sin(y_angle * 0.5)
        cosx = np.cos(x_angle * 0.5)
        sinx = np.sin(x_angle * 0.5)

        # assemble
        w = cosz * cosy * cosx + sinz * siny * sinx
        x = cosz * cosy * sinx - sinz * siny * cosx
        y = sinz * cosy * sinx + cosz * siny * cosx
        z = sinz * cosy * cosx - cosz * siny * sinx

        return cls(np.array([w, x, y, z]))

    @classmethod
    def from_lon_lat(cls, lon, lat):
        """
        Quaternion from spherical cooridnates (modeled as two axis-angle rotations)
        Source: - https://stackoverflow.com/questions/5437865/longitude-latitude-to-quaternion
                - https://github.com/moble/quaternion/blob/306630d69f382827ef097357ca6ee057a42c2103/quaternion.c#L19
        :param lon:
        :param lat:
        :return:
        """
        q_lon = cls.from_axis_angle(np.array([0., 0., 1.]), lon / 2.)
        q_lat = cls.from_axis_angle(np.array([0., 1., 0.]), lat / 2.)

        return q_lon.quatprod(
            q_lat)  # only in this case will lon/2 and lat/2 be the corresponding Euler-angles (same as GitHub code)

    @classmethod
    def from_error_vec(cls, error: np.ndarray):
        return cls(np.hstack((np.sqrt(1.0 - error[0:3] @ error[0:3]), error[0:3])), True)

    @property
    def conj(self):
        return self.from_sw(self.scalar, -1. * self.vector)

    @property
    def scalar(self):
        return self.q[0]

    @property
    def vector(self):
        return self.q[1:]

    @vector.setter
    def vector(self, value):
        self.q[1:] = value

    def to_rodrigues(self):
        identity = np.eye(3)
        w_cross = alg_utils.cross_prod_mat(self.vector)
        return identity + 2 * self.scalar * w_cross + 2 * np.matmul(w_cross, w_cross)

    def to_rotation(self):
        return Rotation.from_matrix(self.to_rodrigues())

    def quatprod(self, q):
        scalar = self.scalar * q.scalar - np.dot(self.vector, q.vector)
        vector = self.scalar * q.vector + self.vector * q.scalar + np.cross(self.vector, q.vector)

        return self.from_sw(scalar, vector, False)

    @property
    def quatprod_matrix(self):
        return np.concatenate((np.concatenate(([self.scalar], -self.vector)).reshape(1, 4),
                               np.concatenate((self.vector.reshape(3, 1), -alg_utils.cross_prod_mat(self.vector)),
                                              axis=1)))

    def vecprod(self, vec):
        scalar = - np.dot(self.vector, vec)
        vector = self.scalar * vec + np.cross(self.vector, vec)

        return self.from_sw(scalar, vector, False)

    @property
    def vecprod_matrix(self):
        return np.concatenate(
            (-self.vector.reshape(1, 3), self.scalar * np.eye(3) + alg_utils.cross_prod_mat(self.vector)))

    def __repr__(self):
        return "<Quaternion scalar: %s vector: %s>" % (self.scalar, self.vector)

    def __getitem__(self, idx):
        return self.q[idx]

    def rotate_vector(self, vector):
        return self.vecprod(vector).quatprod(self.conj).vector
    def to_euler(self):
        phi = np.arctan2(2 * (self.q[0] * self.q[1] + self.q[2] * self.q[3]), 1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2))
        theta = np.arcsin(2 * (self.q[0] * self.q[2] - self.q[3] * self.q[1]))
        psi = np.arctan2(2 * (self.q[0] * self.q[3] + self.q[1] * self.q[2]), 1 - 2 * (self.q[2] ** 2 + self.q[3] ** 2))

        return np.array([phi, theta, psi])

    def to_axis_angle(self):
        pass
        """scalar = np.cos(angle / 2)
        vector = np.sin(angle / 2) * axis

        return cls.from_sw(scalar, vector)"""
        sin_angle_by_2 = np.linalg.norm(self.vector)
        angle = np.arctan2(sin_angle_by_2, self.scalar) * 2
        axis = self.vector / sin_angle_by_2

        return (axis, angle)


def logger_init(seed=42):
    """
    Generator function for the Logger initializer decorator -> needed to pass argument to decorator

    :param seed: random seed
    :return:
    """

    def logger_init_decorator(func):
        """
        Generator function for the Logger initializer decorator

        :param func: function to decorate
        :return:
        """

        def logger_init_wrapper(*args, **kwargs):
            """
            Logger initialization + save function

            :return:
            """
            # set random seed
            np.random.seed(seed)

            # init Logger
            logger = Logger(f"quat_norm_test_{func.__name__}",
                            data_items=("angles", "angles_normalized", "angles_noisy"))

            # run test
            func(logger, *args, **kwargs)

            # save log
            logger.save()

        return logger_init_wrapper

    return logger_init_decorator


class QuatNormTester(object):

    @staticmethod
    @logger_init(seed=42)
    def coarse(logger: Logger, coarse_step_deg: int = 5):
        """
        Test case for sweeping the whole angle manifold (Euler-angles are used)

        :param logger: Logger object
        :param coarse_step_deg: step size
        :return:
        """

        # iterate over the whole 3D polar coordinate space
        for yaw in range(-180, 181, coarse_step_deg):
            for pitch in range(-90, 91, coarse_step_deg):
                for roll in range(-180, 181, coarse_step_deg):
                    yaw = np.deg2rad(yaw)
                    pitch = np.deg2rad(pitch)
                    roll = np.deg2rad(roll)

                    # run the simulation loop with given initial parameters
                    angles, angles_normalized, angles_noisy = QuatNormTester.core(
                        Quaternion.from_euler(roll, pitch, yaw))

                    logger.log(angles=angles, angles_normalized=angles_normalized, angles_noisy=angles_noisy)

    @staticmethod
    @logger_init(seed=42)
    def fine(logger: Logger, coarse_step_deg: int = 90., fine_step_deg: float = .05):
        """
        Test case for sweeping the critical parts (i.e. singularities) angle manifold (Euler-angles are used)

        :param logger: Logger object
        :param coarse_step_deg: step size for the outer loop (to reach critical angle ranges)
        :param fine_step_deg: step size for the inner loop (to sweep critical angle ranges)
        :return:
        """

        # iterate over the critical parts of the 3D polar space
        for yaw in range(0, 361, coarse_step_deg):
            for pitch in range(-90, 91, coarse_step_deg):
                for roll in range(0, 361, coarse_step_deg):
                    if pitch == 0 and yaw == 0:
                        continue

                    # iterate over the environment of the critical parts with higher resolution
                    for yaw_fine in np.arange(0, 5.001, fine_step_deg):
                        for pitch_fine in np.arange(0, 5.001, fine_step_deg):
                            for roll_fine in np.arange(0, 5.001, fine_step_deg):
                                yaw = np.deg2rad(yaw + yaw_fine)
                                pitch = np.deg2rad(pitch + pitch_fine)
                                roll = np.deg2rad(roll + roll_fine)

                                # run the simulation loop with given initial parameters
                                angles, angles_normalized, angles_noisy = QuatNormTester.core(
                                    Quaternion.from_euler(roll, pitch, yaw))

                                logger.log(angles=angles, angles_normalized=angles_normalized,
                                           angles_noisy=angles_noisy)

    @staticmethod
    def core(q: Quaternion):

        noise = np.random.randn(4)
        q_normalized = Quaternion(q.q + noise, normalize=True)

        # todo: implement proposed regularization method here
        q_noisy = Quaternion(q.q + noise, normalize=False)
        # normalize

        return q.to_euler(), q_normalized.to_euler(), q_noisy.to_euler()


from transforms import Rotation

if __name__ == "__main__":
    print("\n-----rodriguez-quaternion conversion equivalency test-----")
    angle = np.pi / 2
    axis = np.array([1, 0, 0])
    r = Rotation.from_axis_angle(axis, angle)
    print("rodriguez matrix", r.matrix)

    q = Quaternion.from_axis_angle(axis, angle)
    print("matrix from q: ", q.to_rodrigues())

    print("\n-----rodriguez-quaternion rotation equivalency test-----")
    vec2rot = np.array([0, 1, 0])

    print(q.rotate_vector(vec2rot))
    print(r.matrix.dot(vec2rot))
