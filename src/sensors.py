import numpy as np

from satellite import SatellitePart
from transforms import HomogeneousTransform


class NoiseProcess(object):
    def __init__(self, mean, std, bias_mean=None, bias_std=None, dim=3):
        """

        :param mean: N-dimensional array or scalar of the mean
        :param std: N-dimensional array or scalar of the standard deviation
        :param bias_mean: flag to include bias in the process
        :param bias_std:
        :param dim: dimension of noise
        """
        self.mean = mean
        self.std = std
        self.dim = dim

        self.bias = np.zeros(self.dim)

        if (bias_mean is None and bias_std is not None) or (bias_mean is not None and bias_std is None):
            raise ValueError("Both bias_mean and bias_std should be None or an N-dimensional array, got ",
                             type(bias_mean), type(bias_std))

        self.bias_mean = bias_mean
        self.bias_std = bias_std

    def sample(self, t_int=None):
        noise = self.std * np.random.randn() + self.mean

        # only if bias is specified
        if self.bias_mean is not None:
            if t_int is None:
                raise ValueError("If bias is present, t_int should not be None!")
            bias_noise = self.bias_std * np.random.randn() + self.bias_mean
            self.bias += bias_noise * t_int

        return noise + self.bias


class Sensor(SatellitePart):
    def __init__(self, t_sample, f_update, noise_mean, noise_std, bias_mean=None, bias_std=None, dim=3,
                 transform=HomogeneousTransform(), parent=None):
        """

        :param t_sample: sample time of the control loop
        :param f_update: update frequency of the sensor
        :param noise_mean: N-dimensional array or scalar of the noise mean
        :param noise_std: N-dimensional array or scalar of the noise standard deviation
        :param bias_mean: N-dimensional array or scalar of the bias mean
        :param bias_std: N-dimensional array or scalar of the bias standard deviation
        :param dim: sensor dimensionality (4 only for the star tracker)
        :param transform: transformation of thesensor measurement frame w.r.t the parent frame it is attached to
        :param parent: parent frame of the sensor
        """
        super().__init__(transform, parent)

        # general
        self.dim = dim

        # timing
        self.t_sample = t_sample
        self.f_update = f_update
        self.counter = self.f_update * self.t_sample

        # noise
        self.noise_process = NoiseProcess(noise_mean, noise_std, bias_mean, bias_std, self.dim)

    def sample(self, **kwargs):

        # sample only if there is a new measurement
        if self.counter > 1.:
            self.counter = self.counter - 1. + self.f_update * self.t_sample

            return self.measure(**kwargs) + self.noise_process.sample(self.t_sample)
        else:
            self.counter += (self.f_update * self.t_sample)

    def measure(self, **kwargs):
        raise NotImplementedError("The _measure function should be implemented after subclassing Sensor!")


class SunSensor(Sensor):
    def __init__(self, normal, i_max, t_sample, f_update, noise_mean, noise_std, bias_mean=None, bias_std=None,
                 transform=HomogeneousTransform(), parent=None):
        """

        :param normal: normal vector of the measurement plane
        :param i_max: maximal current value in A
        :param t_sample: sample time of the control loop
        :param f_update: update frequency of the sensor
        :param noise_mean: N-dimensional array or scalar of the noise mean
        :param noise_std: N-dimensional array or scalar of the noise standard deviation
        :param bias_mean: N-dimensional array or scalar of the bias mean
        :param bias_std: N-dimensional array or scalar of the bias standard deviation
        :param transform: transformation of the sensor measurement frame w.r.t the parent frame it is attached to
        :param parent: parent frame of the sensor
        """
        super().__init__(t_sample, f_update, noise_mean, noise_std, bias_mean, bias_std, 1, transform, parent)

        if len(normal.squeeze().shape) != 1 and normal.squeeze().shape != 3:
            raise TypeError(f"normal should be 3D, got dimension {normal.squeeze().shape}")

        self.normal = normal
        self.i_max = i_max

    def measure(self, sun_vec):
        """

        :param sun_vec: sun vector transformed into the ABC frame
        :return: weighted dot product of the Sun vector and the sensor normal (>0)
        """

        # transform from ABC into the sensor's frame
        sun_vec = self.part2abc().rotation.transpose().vec_mult(sun_vec)

        dot_prod = np.dot(self.normal, sun_vec)
        if dot_prod > 0:
            sun_meas = self.i_max * dot_prod
        else:
            sun_meas = 0.

        return sun_meas


class IMUSensor(Sensor):
    def __init__(self, t_sample, f_update, noise_mean, noise_std, bias_mean=None, bias_std=None, skew=np.zeros((3, 3)),
                 transform=HomogeneousTransform(), parent=None):
        """

        :param t_sample: sample time of the control loop
        :param f_update: update frequency of the sensor
        :param noise_mean: N-dimensional array or scalar of the noise mean
        :param noise_std: N-dimensional array or scalar of the noise standard deviation
        :param bias_mean: N-dimensional array or scalar of the bias mean
        :param bias_std: N-dimensional array or scalar of the bias standard deviation
        :param skew: NxN skew matrix
        :param transform: transformation of the sensor measurement frame w.r.t the parent frame it is attached to
        :param parent: parent frame of the sensor
        """

        super().__init__(t_sample, f_update, noise_mean, noise_std, bias_mean, bias_std, 3, transform, parent)

        if skew.squeeze().shape != (3, 3):
            raise TypeError("normal should be 3D, got dimension ", skew.squeeze().shape)
        self.skew = skew

    def measure(self, ref_vec):
        """

        :param ref_vec: vector of the field/velocity (magnetic/gravitational/angular velocity) in the ABC frame
        :return: 3D measurement
        """

        # transform from ABC into the sensor's frame
        ref_vec = self.part2abc().rotation.transpose().vec_mult(ref_vec)

        return (np.eye(3) + self.skew).dot(ref_vec)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("\n-----Noise process test-----")

    n = NoiseProcess(np.array([.2, .5, .0]), np.array([.1, .8, .5]), np.array([.02, .005, .0]),
                     np.array([.001, .01, .03]))

    n_l = []
    for i in range(1000):
        n_l.append(n.sample(1.))

    plt.plot(n_l)
    # plt.show()

    print("\n-----SunSensor instantiation test-----")
    sun_sensor = SunSensor(np.array([1., 0., 0.]), 5., .01, 500, 0, .0)

    print("Sun sensor measurement", sun_sensor.sample(sun_vec=np.array([0., 1., 0.])))

    print("\n-----IMUSensor instantiation test-----")
    imu_sensor = IMUSensor(.01, 500, 0, .1)

    print("IMU sensor measurement", imu_sensor.sample(ref_vec=np.array([0., 1., 0.])))
