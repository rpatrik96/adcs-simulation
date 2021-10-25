from datetime import datetime
from datetime import timedelta
from enum import Enum

import astropy
import numpy as np
from geomag import WorldMagneticModel
from pyorbital import astronomy
from pyorbital.orbital import Orbital
from sunpy.coordinates import sun

from quaternion import Quaternion
from transforms import Rotation, HomogeneousTransform


class DisturbanceModels(Enum):
    REALISTIC = 0
    PERIODIC = 1


class FrameVector(object):

    def __init__(self, vec_ref: np.ndarray, vec_abc: np.ndarray, noise_abc: np.ndarray) -> None:
        super().__init__()

        self.ref = vec_ref
        self.abc = vec_abc
        self.noise_abc = noise_abc

    @classmethod
    def from_ref(cls, vec: np.ndarray, q: Quaternion, noise=None):
        return cls(vec, q.rotate_vector(vec), noise)

    @property
    def abc_noisy(self):
        return self.abc + self.noise_abc

    def update_abc(self, q: Quaternion):
        self.abc = q.rotate_vector(self.ref)


class GravityModel(object):
    def __init__(self):
        # WGS84 latitude model constants
        # https://en.wikipedia.org/wiki/Theoretical_gravity
        self.g_equator = 9.7803253359  # m/s^2
        self.k = 0.00193185265241  # formula constant
        self.squared_eccentricity = 0.006694379999013

        self.mean_radius = 6371.0087714  # km

    def calc_lat_model(self, lat):
        sin_lat_squared = np.sin(lat) ** 2
        return self.g_equator * (1. + self.k * sin_lat_squared) / (
            np.sqrt(1. - self.squared_eccentricity * sin_lat_squared))

    def calc_alt_lat_model(self, alt, lat):
        g_lat = self.calc_lat_model(lat)

        return g_lat * (self.mean_radius / (alt + self.mean_radius)) ** 2


class DisturbanceTorques(object):

    def __init__(self, mode: DisturbanceModels = DisturbanceModels.REALISTIC) -> None:
        super().__init__()

        self.mode = mode

        """Atmospheric drag"""
        self.rho = 6.967e-13  # kg/m^3, from 407.p. (Markley&Crassidis)

        # from O:\Projects\C3S - RADCUBE\4. Engineering\1. Space Segment\04. ADCS\01. Documentation\
        # ADCS Pointing Analysis Report\RADCUBE-KUL-ADCS_Disturbance_Torques.pdf
        self.drag_coeff = 2.6
        self.aero_surface = 0.06  # m^2
        self.pressure_lever = np.array(
            [0., 0., .05])  # m, distance between CoG and CoP #todo: should be a vector in ECI frame
        self.earth_omega = np.array([0., 0., .000072921158553])  # rad/s

        """Magnetic disturbances"""
        # used as in RADCUBE-KUL-ADCS_Disturbance_Torques.pdf, Section 3
        self.sat_dipole = 0.05 * np.array([1., 1., 1.]) / np.sqrt(3)

        """Solar radiation pressure"""
        self.solar_const = 1367  # W/m^2, maximum of the solar cycle value [1361;1363], from 421.p. (Markley&Crassidis)
        self.reflection_factor = 1.0  # 0 - transparent, 2 - totally reflective ([1;2] is advised at 390.p. in Markley&Crassidis) --> should be between 0 and 1
        assert self.reflection_factor <= 1.0 and self.reflection_factor >= 0.0, ValueError(
            "reflection_factor shall be in [0;1]")
        # from RADCUBE-KUL-ADCS_Disturbance_Torques.pdf
        self.solar_surface = 0.06

        self.step = 0

    def calc_torques(self, sat_vel: np.ndarray, B_vec: np.ndarray, sat2sun: np.ndarray, sat_pos: np.ndarray, inertia,
                     in_eclipse: bool) -> np.ndarray:
        """

        :param inertia: inertia matrix of the satellite
        :param sat_pos: position vector of the satellite in km
        :param sat2sun: vector between the satellite and the Sun in km
        :param sat_vel: velocity vector of the satellite in km/s
        :param B_vec: magnetic field vector in T
        :param in_eclipse: indicator of being in the shadow of the Earth
        :return: sum of the disturbance torques as a vector
        """
        if self.mode is DisturbanceModels.REALISTIC:

            """Conversion to SI"""
            # km to m
            sat_pos *= 1000
            sat_pos_norm = np.linalg.norm(sat_pos)
            sat_pos_unit = sat_pos / sat_pos_norm
            # km to m
            sat2sun *= 1000
            sat2sun_norm = np.linalg.norm(sat2sun)
            sat2sun_unit = sat2sun / sat2sun_norm
            # km/s to m/s
            sat_vel *= 1000

            """Atmospheric drag"""
            # from Markley&Crassidis, p.109.
            sat_vel_rel = sat_vel + np.cross(self.earth_omega, sat_pos)
            sat_vel_rel_norm = np.linalg.norm(sat_vel_rel)
            f_drag = -.5 * self.rho * self.drag_coeff * self.aero_surface * sat_vel_rel_norm * sat_vel_rel
            tau_drag = np.cross(f_drag, self.pressure_lever)

            """Magnetic disturbances"""
            tau_dipole = np.cross(B_vec, self.sat_dipole)

            """Solar radiation pressure"""
            # from Space mission analysis and design, p.366
            f_solar = self.solar_const / astropy.constants.c.value * self.solar_surface * (
                    1 + self.reflection_factor) * sat2sun_unit  # incidence angle is assumed to be 0 (in case of Sun pointing it will be approximately so)
            tau_solar = np.cross(f_solar, self.pressure_lever)

            """Gravity gradient"""
            # from Markley&Crassidis, p.104.
            tau_gg = 3 * astropy.constants.GM_earth.value / sat_pos_norm ** 3 * np.cross(sat_pos_unit,
                                                                                         inertia @ sat_pos_unit)

            tau_dist = tau_drag + tau_dipole + tau_solar * in_eclipse + tau_gg
        elif self.mode is DisturbanceModels.PERIODIC:

            """TEST AREA"""
            # as in https://homepages.laas.fr/arzelier/drupal/sites/homepages.laas.fr.arzelier/files/u117/Version_Finale.pdf
            T0 = 1e-7
            T1 = T2 = 2.5e-6
            phi_T1x = phi_T2y = np.pi / 4
            phi_T2x = phi_T1y = -np.pi / 4
            phi_T1z = 0
            phi_T2z = np.pi / 2

            omega = 2 * np.pi / self.period

            tau_test_x = T0 + T1 * np.sin(omega * self.step + phi_T1x) + T2 * np.sin(2 * omega * self.step + phi_T2x)
            tau_test_y = T0 + T1 * np.sin(omega * self.step + phi_T1y) + T2 * np.sin(2 * omega * self.step + phi_T2y)
            tau_test_z = T0 + T1 * np.sin(omega * self.step + phi_T1z) + T2 * np.sin(2 * omega * self.step + phi_T2z)

            tau_dist = np.array([tau_test_x, tau_test_y, tau_test_z])
        else:
            raise NotImplementedError

        self.step += 1

        return tau_dist


class EnvModelWrapper(object):
    def __init__(self, timestep=500, start_time=datetime(2019, 9, 1, 0, 0, 0, 0)) -> None:
        """

        :param timestep: timestep in MILLISECONDS
        :param start_time:
        """

        # temporal
        self.last_vel = np.array([0., 0., 0.])
        self.timestep = timedelta(microseconds=1000 * timestep)
        self.time = start_time

        # magnetic model
        self.wmm = WorldMagneticModel()

        # gravity
        self.grav_model = GravityModel()

        # disturbance torques
        self.dist_torques = DisturbanceTorques()
        self.dist_torques.period = 5000. / self.timestep.total_seconds()  # todo: only needed to scale the test torques

        # orbit propagator
        self.orb = Orbital("LANDSAT-7")

    def _calc_eclipse_condition(self, sun_pos: np.ndarray, sat_pos: np.ndarray) -> bool:
        # Markley&Crassidis, p.422.
        sun_pos /= np.linalg.norm(sun_pos)
        return sat_pos.dot(sun_pos) < -np.sqrt(
            np.linalg.norm(sat_pos) ** 2 + np.linalg.norm(self.grav_model.mean_radius) ** 2)

    def step(self, inertia: np.ndarray):
        """
        Performs one update step in the environment models
        :param inertia:
        :return: - acceleration vector
                 - magnetic field vector
                 - vector from satellite frame to the Sun (both frames are assumed to be as ECI but translated)
        """

        """orbit propagation"""
        # sat_pos in km
        # sat_vel in km/s
        sat_pos, sat_vel = self.orb.get_position(self.time, normalize=False)  # returns ECI coordinates of the satellite
        lon, lat, alt = self.orb.get_lonlatalt(self.time)

        ned2eci_rot = Rotation.from_axis_angle(np.array([0., 0., 1.]), np.deg2rad(lon)) @ Rotation.from_axis_angle(
            np.array([0., 1., 0.]), -(np.pi / 2 + np.deg2rad(lat)))  # .transpose()

        obs_pos, _ = astronomy.observer_position(self.time, lon, lat, 0)
        obs_pos = np.array(obs_pos)

        ned2eci = HomogeneousTransform(ned2eci_rot, -obs_pos)

        abc2ned_tr = obs_pos - sat_pos

        """sun angle"""
        # check whether the ECI x direction is the vernal equinox
        # according to https://en.wikipedia.org/wiki/Equatorial_coordinate_system,
        # x is in the direction of the vernal equinox
        # right ascension in hours (according to tests, hours = RADIANS)
        # declination in radians
        # todo: for C++ the algorithm on 420.p. (Markley&Crassidis) can be used
        ra_sun, dec_sun = astronomy.sun_ra_dec(self.time)

        """# project position onto the xy-plane
        pos_xy_plane = deepcopy(sat_pos)
        pos_xy_plane[2] = 0.

        ra_sat = np.arctan2(sat_pos[1], sat_pos[0])
        dec_sat = np.arccos(
            np.clip(np.dot(sat_pos, pos_xy_plane) / (np.linalg.norm(sat_pos) * np.linalg.norm(pos_xy_plane)), -1., 1.))

        # RA + declination
        rel_sun_angles = (ra_sat-ra_sun, dec_sat-dec_sun)"""

        sun_earth_dist = sun.earth_distance(self.time)

        dec_sun_cos = np.cos(dec_sun)
        sun_pos_x = dec_sun_cos * np.cos(ra_sun) * sun_earth_dist.km
        sun_pos_y = dec_sun_cos * np.sin(ra_sun) * sun_earth_dist.km
        sun_pos_z = np.sin(dec_sun) * sun_earth_dist.km
        sun_pos = np.array([sun_pos_x, sun_pos_y, sun_pos_z])

        sat2sun = sun_pos - sat_pos

        """gravity"""
        # calculate gravity direction
        # sanity check done as part of a dicussion with Norbert Tarcai, it seems to be OK
        g_dir = sat_pos - obs_pos
        g_dir /= np.linalg.norm(g_dir)

        # scale gravity
        g = g_dir * self.grav_model.calc_alt_lat_model(alt, lat)  # m/s^2

        """acceleration"""
        sat_vel = np.array(sat_vel)
        a_dv = (sat_vel - self.last_vel) / self.timestep.total_seconds() * 1000  # m/s^2

        a = g + a_dv

        """magnetic field"""
        self.wmm.calc_mag_field(lat, lon, alt, self.time.date(), 'km')
        B_vec = np.array([self.wmm.Bx, self.wmm.By, self.wmm.Bz]) * 1e-9  # convert to T

        """eclipse condition"""
        in_eclipse = self._calc_eclipse_condition(sun_pos, sat_pos)

        """disturbance torques"""
        tau_dist = self.dist_torques.calc_torques(sat_vel, B_vec, sat2sun, sat_pos, inertia, in_eclipse)

        # store last velocity
        self.last_vel = np.array(sat_vel)

        # update time
        self.time += self.timestep

        return a, B_vec, sat2sun, ned2eci, abc2ned_tr, sat_vel, sat_pos, sun_pos, in_eclipse, tau_dist


if __name__ == "__main__":
    print("\n-----Noise process test-----")
    inertia = np.array([
        [6.313E-02, 7.776E-06, 3.399E-04],
        [7.776E-06, 7.356E-03, 7.018E-05],
        [3.399E-04, 7.018E-05, 3.511E-03]])

    env = EnvModelWrapper()
    a, B_vec, sat2sun, ned2eci, abc2ned_tr, sat_vel, sat_pos, sun_pos, in_eclipse, tau_dist = env.step(inertia)

    print("\n-----Disturbance torque test-----")
    dist = DisturbanceTorques()
    print(dist.calc_torques(sat_vel, B_vec, sat2sun, sat_pos, inertia, in_eclipse))
