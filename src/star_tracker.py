from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from alg_utils import vector_angle
from esoq2 import ESOQ2
from quaternion import Quaternion
from satellite import SatellitePart
from star_catalog import StarCatalog
from transforms import HomogeneousTransform, Rotation
from voting_algorithm import VotingAlgorithm


class Camera(SatellitePart):
    def __init__(self, width: int = 400, height: int = 300, fov: int = 45,
                 transform: HomogeneousTransform = HomogeneousTransform(),
                 parent: Optional[SatellitePart] = None) -> None:
        """
        Initializes the Camera object

        :param width: sensor width in pixels
        :param height: sensor height in pixels
        :param fov: sensor Field of View (FoV) in degrees
        :param transform: transformation of the camera measurement frame w.r.t the parent frame it is attached to
        :param parent: parent frame of the camera
        """
        super().__init__(transform, parent)

        # parameters in PIXELS
        self.width = width
        self.height = height
        self.fov = np.deg2rad(fov)
        self.focal_length = np.sqrt(self.width ** 2 + self.height ** 2) / (2 * np.tan(self.fov / 2))

        self.camera_mat = Rotation.from_matrix(np.array([[self.focal_length, 0., self.width / 2.],
                                                         [0., self.focal_length, self.height / 2.],
                                                         [0., 0., 1.]]))  # todo: is not orthogonal

        # todo: check need for other parameters
        # self.shutter_speed = shutter_speed
        # self.aperture = aperture

    def project(self, vec_array: np.ndarray, corr_mat: Rotation = Rotation.from_matrix(), clip2sensor: bool = True) -> \
    List[np.ndarray]:
        """
        Performs a projection operation onto the camera sensor

        :param vec_array: array of 3D vectors in Euclidean space
        :param corr_mat: Rotation describing the transformation between the actual camera orientation and the standard
                        camera setup (sensor plane is the xy-plane, principal axis is the z-axis)
        :param clip2sensor: if True, returns only stars projected onto the sensor
        :return: 2D vector in camera space (in pixels)
        """

        projected = []

        # project each vector of the list
        for vec in vec_array:
            # homogeneous coordinates
            hom_coord = self.camera_mat.vec_mult(corr_mat.vec_mult(vec))

            # sensor frame
            inv_z = 1. / hom_coord[2]
            x, y = hom_coord[0] * inv_z, hom_coord[1] * inv_z

            # store only points which are projected onto the sensor
            if not clip2sensor or (0 < x < self.width and 0 < y < self.height):
                projected.append(np.array([x, y]))

        return projected

    def synthesize(self, stars: pd.DataFrame, corr_mat: Rotation = Rotation.from_matrix(), num_frac_bits: int = 16,
                   mag_noise_std=0.) -> np.ndarray:
        """
        Constructs a synthetic image, given a camera orientation and a set of stars

        :param mag_noise_std: magnitude noise standard deviation
        :param stars: star catalogue entries to be imaged
        :param corr_mat: Rotation describing the transformation between the actual camera orientation and the standard
                        camera setup (sensor plane is the xy-plane, principal axis is the z-axis)
        :param num_frac_bits: number of bits to represent the fractional part of a star centroid
        :return: synthetic image as a numpy.array
        """
        # project stars onto the sensor
        factor = (1 << num_frac_bits)
        star_pixels = self.project(stars[["x", "y", "z"]].values, corr_mat)

        # print(stars)
        # print(star_pixels)

        # initialize new image
        img = np.zeros((self.height, self.width), np.uint8)

        """Draw"""
        # draw each star as an ellipse with some noise
        for mag, sensor_coord in zip(stars.mag.tolist(), star_pixels):
            half_major = 2.4  # + .5*np.random.rand()
            half_minor = 2.4  # + .5*np.random.rand()
            rot_angle = 0  # np.random.randint(180)
            # todo: color should be determined based on magnitude
            img = cv2.ellipse(img, tuple(map(int, np.round(sensor_coord * factor))),
                              (int(half_major * factor), int(half_minor * factor)),
                              rot_angle, 0, 360,
                              np.clip(self._mag2intensity(mag) + np.random.randn(1)*mag_noise_std, 0, 255),
                              -1, shift=num_frac_bits)

        """Distort"""
        # 1. blur (defocus)
        # todo: sigma should be modified based on the satellite velocity
        img = cv2.GaussianBlur(img, (3, 3), 2, borderType=cv2.BORDER_ISOLATED)

        # 2. add salt & pepper noise
        sp_img = np.random.randint(8192, size=(self.height, self.width))
        salt_mask = sp_img > 8190
        pepper_mask = sp_img > 2  # False if less than 2 -> multiplying with this ensures pepper noise
        salt_img = (salt_mask * sp_img).astype(np.uint8)
        # img = cv2.add(img, salt_img) * pepper_mask

        return img

    def _mag2intensity(self, mag: float) -> int:
        """
        Converts magnitude to intensity
        :param mag: magnitude value
        :return: intensity value
        """
        col = 50 + np.random.randint(45)
        if mag < 1.38:
            col = 240 + np.random.randint(15)
        elif mag < 2.76:
            col = 226 + np.random.randint(10)
        elif mag < 4.14:
            col = 150 + np.random.randint(26)

        return col


# todo: tracking mode needed
class StarTracker(SatellitePart):
    def __init__(self, width: int = 400, height: int = 300, fov: int = 45, uncertainty: float = np.deg2rad(.03),
                 transform: HomogeneousTransform = HomogeneousTransform(),
                 camera_transform: HomogeneousTransform = HomogeneousTransform(), parent: SatellitePart = None) -> None:
        """

        :param width: sensor width in pixels
        :param height: sensor height in pixels
        :param fov: sensor Field of View (FoV) in degrees
        :param uncertainty: uncertainty of the centroid calculation
        :param transform: transformation of the star tracker frame w.r.t the parent frame it is attached to
        :param camera_transform: transformation of the camera measurement frame w.r.t the star tracker frame
        :param parent: parent frame of the star tracker
        """
        super().__init__(transform, parent)

        self.camera = Camera(width, height, fov, camera_transform, self)
        self.catalog = StarCatalog()
        self.voting_alg = VotingAlgorithm(uncertainty)
        self.esoq2 = ESOQ2()

    def synthesize(self, eci2abc_rot: Rotation) -> np.ndarray:
        """
        Constructs a synthetic image, given a camera orientation and a set of stars

        :param eci2abc_rot: Rotation describing the transformation between ECI and ABC frames
        :return: synthetic image as a numpy.array
        """

        # z axis in the ABC frame
        principal_axis_abc = self.camera.part2abc().rotation.vec_mult(np.array([0, 0, 1]))

        stars_in_fov = self.catalog.filter_stars(eci2abc_rot.transpose().vec_mult(principal_axis_abc), self.camera.fov)
        return self.camera.synthesize(stars_in_fov, eci2abc_rot)

    def predict(self, img: np.ndarray) -> Quaternion:
        """
        Predicts the stars on the image specified as input

        :param img: grayscale image
        :return: attitude matrix (from the frame of the star tracker to the ECI frame)
        """

        # identify stars
        stars_found = self.voting_alg.predict_voting(img, self.catalog.star_pairs, self.camera.fov)
        catalog_stars = self.catalog.stars.loc[list(stars_found.keys())][["x", "y", "z"]].values

        # calculate attitude
        self.esoq2.step(np.array(list(stars_found.values())), catalog_stars)

        return self.esoq2.q_opt


if __name__ == "__main__":
    """camera test"""

    """st_cat = StarCatalog()
    cam = Camera(1600, 1200, 40)

    # st_cat.filter_stars(cam.orientation.axis, cam.fov)

    # corr_mat = Rotation.from_matrix(np.array([[0, 0, 1],
    #                                            [0, 1, 0],
    #                                            [-1, 0, 0]]))

    # img, star_pixels = cam.synthesize(st_cat.filter_stars(cam.orientation.axis, cam.fov), corr_mat)
    # synth_img, star_pixels = cam.synthesize(st_cat.stars, corr_mat)

    indices = [744, 3172, 3814, 4417, 6672, 8867]
    corr_mat = Rotation.from_matrix(np.array(
        [[+0.900194, +0.022523, -0.434908], [-0.039122, +0.998807, -0.029250], [+0.433730, +0.043345, +0.900000]]))

    ids = st_cat.filter_stars(corr_mat.transpose().vec_mult(np.array([0, 0, 1])), cam.fov)
    pass

    # img = cam.synthesize(st_cat.stars.loc[indices], corr_mat)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # f = open("py_synth_img.bin", "wb")
    # f.write(bytearray(img.reshape(-1, ).tolist()))
    # img = np.fromfile("../data/star_img/IMG_9378_Cass.bin", np.uint8).reshape(1200, 1600)
    # img = np.fromfile("../data/star_img/cassiopeia_synth.bin", np.uint8).reshape(1200, 1600)"""

    """star tracker test"""
    st = StarTracker(1600, 1200, 20)
    indices = [744, 3172, 3814, 4417, 6672, 8867]
    corr_mat = Rotation.from_matrix(np.array(
        [[+0.900194, +0.022523, -0.434908], [-0.039122, +0.998807, -0.029250], [+0.433730, +0.043345, +0.900000]]))
    img = st.synthesize(corr_mat)
    x = st.predict(img)
    unit_vec = np.array([1., 0., 0.])
    orig_vec = corr_mat.vec_mult(unit_vec)
    est_vec = x.rotate_vector(orig_vec)
    print(f"original rotated vector:\n\t{orig_vec} \n"
          f"rotated back:\n\t{est_vec} \n"
          f"angle (arcmin): {60 * np.rad2deg(vector_angle(unit_vec, est_vec))}")

    pass
