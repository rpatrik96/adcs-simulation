import numpy as np
from datetime import datetime
from pyorbital import astronomy
from pyorbital.orbital import Orbital
from sunpy.coordinates import sun
from transforms import Rotation
if __name__ == "__main__":
    orb = Orbital("Suomi NPP")
    now = datetime.utcnow()
    orb.get_orbit_number(now)

    sat_pos, _ = orb.get_position(now, False)  # returns ECI coordinates of the satellite
    lon, lat, alt = orb.get_lonlatalt(now)

    print(type(sat_pos))

    print(lon, lat, alt)
    obs_pos, _ = astronomy.observer_position(now, lon, lat, 0)
    print(sat_pos)
    print(obs_pos)
    print(np.array(sat_pos) - np.array(obs_pos))

    print("-----ECI2NED test-----")
    ned2eci_rot = Rotation.from_axis_angle(np.array([0., 0., 1.]), np.deg2rad(lon)).rotate(
        Rotation.from_axis_angle(np.array([0., 1., 0.]), -(np.pi / 2 + np.deg2rad(lat))))  # .transpose()

    print(ned2eci_rot.rot_mat)

    """NED calculation from vectors - FAILS (not the same as the matrix approach)"""
    # normal vector of the N-E plane of NED
    sat2obs = obs_pos - sat_pos
    D = sat2obs / np.linalg.norm(sat2obs)

    # intersection (z coordinate) of the N vector of NED and the z axis of ECI

    obs2eci = np.array([0., 0., np.dot(D, obs_pos) / D[2]]) - obs_pos
    N = obs2eci / np.linalg.norm(obs2eci)

    E = np.cross(D, N)

    # as the N, E, D vector are the rows of the matrix
    # it transforms from ECI to NED
    eci2ned = Rotation.from_rot_mat(np.array([N, E, D]))

    print(np.array([N, E, D]).transpose())
    # print(eci2ned.rot_mat)

    """Sun"""

    astronomy.sun_zenith_angle(now, lon, lat)
    # todo: unit disambiguation + azimuth is given wrt what?
    sun_alt, sun_az = astronomy.get_alt_az(now, lon, lat)

    """right ascension check"""
    vernal_eq_2019 = datetime(2019,9,23,9,50)

    print(astronomy.sun_ra_dec(vernal_eq_2019))

    # check whether the ECI x direction is the vernal equinox
    # according to https://en.wikipedia.org/wiki/Equatorial_coordinate_system,
    # x is in the direction of the vernal equinox
    # right ascension in hours (according to tests, hours = RADIANS)
    # declination in degrees
    right_asc_h, decl = astronomy.sun_ra_dec(now)

    # right_asc_rad = right_asc_h / 24. * 2 * np.pi

    print("-----sun-earth distance test-----")
    dist = sun.earth_distance(vernal_eq_2019)
    print(dist)
    pass
