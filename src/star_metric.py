from os.path import join, dirname, abspath

import numpy as np
import pandas as pd

from alg_utils import vector_angle
from logger import Logger, LogData
from quaternion import Quaternion
from satellite import Satellite
from transforms import Rotation
from voting_algorithm import VotingAlgorithm


class StarMetric(object):

    def __init__(self, sim_name="star_metric", num_experiments=10, num_stars=3,
                 noise_std_scales=(0.1, 0.5, 1, 5, 10)) -> None:
        super().__init__()

        # constants
        self.num_experiments = num_experiments
        self.num_stars = num_stars
        self.noise_std_scales = noise_std_scales
        self.gamma_estimates = [f"gamma_est_std_{std}" for std in self.noise_std_scales]
        self.angle_estimates = [f"angle_est_std_{std}" for std in self.noise_std_scales]

        # file system
        self.data_dir = join(dirname(dirname(abspath(__file__))), "data")

        pairs_data_path = join(self.data_dir, "star_pairs_max_90_deg_with_distance.tsv")
        self.df_pairs = pd.read_csv(pairs_data_path, "\t")

        # star catalog
        data_path = join(self.data_dir, "stars_mag5_5.tsv")
        self.df = pd.read_csv(data_path, "\t").set_index('Unnamed: 0')

        # setup logger
        self.logger = Logger(sim_name, data_items=["angle_gt", "gamma_gt",
                                                   "id1", "id2",
                                                   "mag1", "mag2",
                                                   "mag1_est", "mag2_est",
                                                   *self.angle_estimates, *self.gamma_estimates],
                             log_type=LogData)

        # camera setup
        self.st = Satellite(dt=0.25).star_tracker
        self.img_center = np.array([self.st.camera.width / 2, self.st.camera.height / 2])
        self.principal_axis_abc = self.st.camera.part2abc().rotation.vec_mult(np.array([0, 0, 1]))

        # voting algorithm
        self.voting = VotingAlgorithm()

        # filter for FoV/magnitude
        self.df_in_fov = self.df_pairs[self.df_pairs.angle < self.st.camera.fov]
        self.mag1_discrete = self.df_in_fov.mag1.apply(self.voting._mag2class)
        self.mag2_discrete = self.df_in_fov.mag2.apply(self.voting._mag2class)
        self.cos_angle = np.cos(self.df_in_fov.angle)

    @property
    def focal_length(self):
        return self.st.camera.camera_mat.matrix[0, 0]

    def z_angle(self, noise_std=0.1, eps=3e-4):
        return np.array([len(self.df_in_fov.loc[(np.abs(self.df_in_fov.angle - angle) < eps)]) for angle in
                         self.logger.__dict__[f"angle_est_std_{noise_std}"].data])

    def z_old(self, noise_std=0.1, eps=3e-4):
        return np.array([len(self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps)]) for angle in
                         self.logger.__dict__[f"angle_est_std_{noise_std}"].data])

    def z_old_filtered(self, noise_std=0.1, eps=3e-4):
        return np.array([len(self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps)])
                         if len(self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps) &
                                                   (self.df_in_fov.id1 == star1) &
                                                   (self.df_in_fov.id2 == star2)
                                                   ]) != 0
                         else 0
                         for angle, star1, star2 in
                         zip(self.logger.__dict__[f"angle_est_std_{noise_std}"].data, self.logger.id1.data,
                             self.logger.id2.data)])

    def z_old_mag(self, noise_std=0.1, eps=3e-4):
        return np.array([len(self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps) &
                                                (self.mag1_discrete == m1) & (self.mag2_discrete == m2)
                                                ])
                         for angle, m1, m2 in
                         zip(self.logger.__dict__[f"angle_est_std_{noise_std}"].data, self.logger.mag1_d,
                             self.logger.mag2_d)])

    def z_new2(self, noise_std=0.1, k=1e-4, w=6, offset=3.3e-4):
        return np.array([len(self.df_in_fov.loc[
                                 (np.abs(self.df_in_fov.distance_ratio - dist) < (k * np.abs(
                                     np.sqrt(2) - dist) ** w + offset))
                             ]) for dist in self.logger.__dict__[f"gamma_est_std_{noise_std}"].data])

    def z_new2_filtered(self, noise_std=0.1, k=1e-4, w=6, offset=3.3e-4):
        return np.array([len(self.df_in_fov.loc[(
                np.abs(self.df_in_fov.distance_ratio - dist) < (k * np.abs(np.sqrt(2) - dist) ** w + offset))])
                         if len(self.df_in_fov.loc[
                                    (self.df_in_fov.id1 == star1) &
                                    (self.df_in_fov.id2 == star2) &
                                    (np.abs(self.df_in_fov.distance_ratio - dist) < (k * np.abs(
                                        np.sqrt(2) - dist) ** w + offset))
                                    ]) != 0
                         else 0
                         for dist, star1, star2 in
                         zip(self.logger.__dict__[f"gamma_est_std_{noise_std}"].data, self.logger.id1.data,
                             self.logger.id2.data)])

    def z_new2_mag(self, noise_std=0.1, k=1e-4, w=6, offset=3.3e-4):
        return np.array([len(self.df_in_fov.loc[
                                 (np.abs(self.df_in_fov.distance_ratio - dist) < (k * np.abs(
                                     np.sqrt(2) - dist) ** w + offset)) &
                                 (self.mag1_discrete == m1) & (self.mag2_discrete == m2)
                                 ])
                         for dist, m1, m2 in
                         zip(self.logger.__dict__[f"gamma_est_std_{noise_std}"].data, self.logger.mag1_d,
                             self.logger.mag2_d)])

    def z2(self, noise_std=0.1, k=1e-4, w=6, offset=3.3e-4, eps=3e-4):
        return np.array([len(
            self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps) &
                               (np.abs(self.df_in_fov.distance_ratio - dist) < (k * np.abs(
                                   np.sqrt(2) - dist) ** w + offset))
                               ]
        )
            for angle, dist in zip(self.logger.__dict__[f"angle_est_std_{noise_std}"].data,
                                   self.logger.__dict__[f"gamma_est_std_{noise_std}"].data)])

    def z2_mag(self, noise_std=0.1, k=1e-4, w=6, offset=3.3e-4, eps=3e-4):
        return np.array([len(
            self.df_in_fov.loc[(np.abs(self.cos_angle - np.cos(angle)) < eps) &
                               (np.abs(self.df_in_fov.distance_ratio - dist) < (
                                       k * np.abs(np.sqrt(2) - dist) ** w + offset)) &
                               (self.mag1_discrete == m1) & (self.mag2_discrete == m2)
                               ]
        )
            for angle, dist, m1, m2 in
            zip(self.logger.__dict__[f"angle_est_std_{noise_std}"].data,
                self.logger.__dict__[f"gamma_est_std_{noise_std}"].data, self.logger.mag1_d, self.logger.mag2_d)])

    def discretize_logger_magnitude(self):
        if len(self.logger.mag1.data):
            self.logger.mag1_d = [self.voting._mag2class(m) for m in self.logger.mag1.data]
            self.logger.mag2_d = [self.voting._mag2class(m) for m in self.logger.mag2.data]

    def gain_grid(self, data):

        # init data structures
        gain = np.zeros((4, 4), dtype=np.int)
        num = np.zeros((4, 4), dtype=np.int)

        # fill the grid
        for (s1, s2, val) in zip(self.logger.mag1_d, self.logger.mag2_d, data):
            gain[s1 - 1, s2 - 1] += val
            num[s1 - 1, s2 - 1] += 1

        # correct items with 0 item
        num[num == 0] = 1
        return gain, num

    def calc_ground_truth(self, star1: np.ndarray, star2: np.ndarray) -> float:

        # 1. calculate the bisector
        bisector = (star1 + star2) / 2.0
        bisector /= np.linalg.norm(bisector)

        # 2. determine the rotation axis and angle
        rot_axis = np.cross(self.principal_axis_abc, bisector)
        rot_angle = vector_angle(self.principal_axis_abc, bisector)
        q_rot = Quaternion.from_axis_angle(axis=rot_axis, angle=rot_angle)

        # 3. rotate the stars to simulate the configuration with
        # the camera axis equal to the bisector, normlaize teh vectors
        # (they are anyway of identical length, but this way, the method is more robust)
        s1_rot = q_rot.conj.rotate_vector(star1)
        s1_rot /= np.linalg.norm(s1_rot)
        s2_rot = q_rot.conj.rotate_vector(star2)
        s2_rot /= np.linalg.norm(s2_rot)

        # 4. project the stars onto the sensor
        stars_projected = self.st.camera.project(np.array([s1_rot, s2_rot]), clip2sensor=False)

        # 5. calculate distance
        r_norm = np.linalg.norm(np.append(stars_projected[0] - self.img_center, self.focal_length))

        # 6. calculate ||2c|| - in this case 2c lies on the sensor
        two_c_norm = np.linalg.norm(stars_projected[0] - stars_projected[1])

        # 7. calculate gamma
        gamma = two_c_norm / r_norm

        return gamma

    def calc_gamma(self, star1: np.ndarray, star2: np.ndarray, rot: Rotation,
                   noise1: np.ndarray = np.zeros(2), noise2: np.ndarray = np.zeros(2)) -> float:
        """
        Calculates the gamma metric as in the paper
        "Efficient Candidate Selection for Star TrackerAlgorithms without Magnitude Information"
        :param star1: numpy array of the first star in 2D
        :param star2: numpy array of the second star in 2D
        :param rot: Rotation object representing the orientation of the star tracker
        :param noise1: numpy array of the first star noise in 2D
        :param noise2: numpy array of the second star noise in 2D
        :return:
        """

        # project stars
        stars_projected = self.st.camera.project(np.array([star1, star2]), corr_mat=rot, clip2sensor=True)

        # 1. convert to 3D
        # 2. add noise
        s0 = np.append(stars_projected[0] + noise1 - self.img_center, self.focal_length)
        s1 = np.append(stars_projected[1] + noise2 - self.img_center, self.focal_length)

        # 3. calculate norms
        s0_norm = np.linalg.norm(s0)
        s1_norm = np.linalg.norm(s1)

        # 4. select the vector with bigger norm
        if s0_norm < s1_norm:
            r_b = s1
            r_a = s0

            r_b_norm = s1_norm
            r_a_norm = s0_norm
        else:
            r_b = s0
            r_a = s1

            r_b_norm = s0_norm
            r_a_norm = s1_norm

        # 5. calculate the estimate of ||2c||
        two_c_norm = np.linalg.norm(r_b_norm / r_a_norm * r_a - r_b)

        # 6. return with the estimate of gamma
        gamma = two_c_norm / r_b_norm

        return gamma

    def calc_distance_ratio(self):
        """
        Use for recalculating the distance_ratio (a.k.a. gamma)
        :return:
        """
        distance_ratios = []

        for i in range(len(self.df_pairs)):
            # extract star coordinates of the pair
            star1 = self.df.loc[int(self.df_pairs.loc[i].id1)][["x", "y", "z"]].to_numpy()
            star2 = self.df.loc[int(self.df_pairs.loc[i].id2)][["x", "y", "z"]].to_numpy()

            # calculate sin-like quantity
            distance_ratios.append(self.calc_ground_truth(star1, star2))

        # append column
        self.df_pairs["distance_ratio"] = distance_ratios

        # save file
        self.df_pairs.to_csv(join(self.data_dir, "star_pairs_max_90_deg_with_distance.tsv"), sep="\t")

    def run_monte_carlo(self, mag_noise_std: float = 0, seed: int = 42):
        np.random.seed(seed)

        # shorthand for deciding whether to synthesize the image
        synthesize = int(mag_noise_std) == 0

        for experiment in range(self.num_experiments):

            # generate random orientation
            axis = np.random.rand(3)
            angle = 0
            rot = Rotation.from_axis_angle(axis, angle)

            num_stars = 0
            while True:

                # samle the star pair
                i = np.random.randint(len(self.df_pairs) - 1)

                # extract star coordinates of the pair
                star1 = self.df.loc[int(self.df_pairs.loc[i].id1)][["x", "y", "z"]].to_numpy()
                star2 = self.df.loc[int(self.df_pairs.loc[i].id2)][["x", "y", "z"]].to_numpy()

                stars_projected = self.st.camera.project(np.array([star1, star2]), corr_mat=rot, clip2sensor=True)

                # proceed only if both stars are on the sensor
                if len(stars_projected) == 2:
                    num_stars += 1

                    if synthesize:

                        # 1. synthesize
                        # add magnitude noise:
                        # - salt and pepper (masked by not zero pixels)
                        # - magnitude mismatch for rendering the ellipse
                        img = self.st.camera.synthesize(
                            self.df.loc[[int(self.df_pairs.loc[i].id1), int(self.df_pairs.loc[i].id2)]],
                            mag_noise_std=mag_noise_std)
                        # 2. calculate centroids
                        centroids = self.voting.weighted_centroids(img)

                        if len(centroids) != 2:
                            raise ValueError(f"The number of centroids is {len(centroids)} instead of 2")

                    # create temporary placeholder dicts for the estimates
                    angle_dict = {a: None for a in self.angle_estimates}
                    gamma_dict = {g: None for g in self.gamma_estimates}

                    for n_std, gamma_key, angle_key in zip(self.noise_std_scales, self.gamma_estimates,
                                                           self.angle_estimates):

                        # generate noises
                        n1 = n_std * np.random.randn(2, )
                        n2 = n_std * np.random.randn(2, )

                        if not synthesize:
                            s0 = np.append(stars_projected[0][0:2] + n1 - self.img_center, self.focal_length)
                            s1 = np.append(stars_projected[1][0:2] + n2 - self.img_center, self.focal_length)
                        else:
                            # no img_center adjustment needed
                            s0 = np.array([centroids[0][0:2] + n1, self.focal_length])
                            s1 = np.array([centroids[1][0:2] + n2, self.focal_length])

                        # calculate angle
                        angle_dict[angle_key] = vector_angle(s0, s1)

                        # calculate gamma
                        gamma_dict[gamma_key] = self.calc_gamma(star1, star2, rot, n1, n2)

                    # log data
                    self.logger.log(angle_gt=self.df_pairs.loc[i].angle,
                                    gamma_gt=self.df_pairs.loc[i].distance_ratio,
                                    id1=int(self.df_pairs.loc[i].id1),
                                    id2=int(self.df_pairs.loc[i].id2),
                                    mag1=self.df_pairs.loc[i].mag1,
                                    mag2=self.df_pairs.loc[i].mag2,
                                    mag1_est=0 if not synthesize else centroids[0][2],
                                    mag2_est=0 if not synthesize else centroids[1][2],
                                    **angle_dict,
                                    **gamma_dict
                                    )

                    if num_stars == self.num_stars:
                        break

        self.logger.save()


from args import get_args

if __name__ == "__main__":
    # parse arguments
    args = get_args()

    # set seed
    np.random.seed(args.seed)

    # run MC experiments
    self = StarMetric(num_experiments=args.num_experiments, num_stars=args.num_stars)
    self.calc_distance_ratio()
    # self.run_monte_carlo(args.mag_noise_std)
