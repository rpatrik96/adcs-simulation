from os.path import join, dirname, abspath
from typing import List

import numpy as np
import pandas as pd

from alg_utils import vector_angle


# todo: a tracking módhoz kellene valami hatékonyabb keresési módszer/ adatstruktúra
class StarCatalog(object):
    def __init__(self) -> None:
        super().__init__()

        data_dir = join(dirname(dirname(abspath(__file__))), "data")
        self.stars = pd.read_csv(join(data_dir, "stars_mag5_5.tsv"), sep='\t').set_index('Unnamed: 0')
        self.star_pairs = pd.read_csv(join(data_dir, "star_pairs_max_90_deg.tsv"), sep='\t')

    def filter_stars(self, orientation: np.ndarray, fov: float) -> pd.DataFrame:
        """
        Filters the stars from the star catalog which are in the FoV given an orientation
        :param orientation: principal axis of vision (3d orientation vector of the observer)
        :param fov: FoV of the observer
        :return: stars within the FoV as a DataFrame
        """

        if fov < 0.:
            raise ValueError("FoV should be positive, got: ", fov)
        if fov > np.pi / 2:
            raise ValueError("FoV should be less than 90 degrees, got: ", fov)

        # select the stars which are within the FoV
        # query their indices (=IDs) as a set
        filtered_stars = self.stars[self._stars_within_fov(orientation, fov)]
        filtered_star_ids = set(filtered_stars.index.get_values())

        # filter pairs
        # 1: preserve only pairs where id1 belongs to star within FoV
        id1s = set(self.star_pairs.id1)
        # set_index is used to enable indexing with the ids
        # reset_index is needed to preserve the id columns (otherwise, the second set_index removes the id1 column)
        pairs_id1_filtered = self.star_pairs.set_index('id1').loc[filtered_star_ids.intersection(id1s)].reset_index()
        # 2: preserve only pairs where id1 and id2 belongs to star within FoV
        id2s = set(pairs_id1_filtered.id2)
        pairs_id1_2_filtered = pairs_id1_filtered.set_index('id2').loc[
            filtered_star_ids.intersection(id2s)].reset_index()

        # collect the indices of the stars witih FoV
        stars_in_fov_ids = set(pairs_id1_2_filtered.id1).union(set(pairs_id1_2_filtered.id2))

        return self.stars.loc[stars_in_fov_ids]

    def _stars_within_fov(self, orientation: np.ndarray, fov: float) -> List[bool]:
        """
        Calculates whether a star in the catalog is within the FoV of the star tracker given a specific orientation
        :param orientation: normal vector of the camera sensor plane
        :param fov: Field of View of the camera
        :return: a list of bools indicating whether a star is within FoV
        """
        # constants
        num_stars = len(self.stars)
        fov_2 = fov / 2

        # placeholder for the index
        within_fov = num_stars * [False]

        # list of star positions
        star_pos = self.stars[["x", "y", "z"]].values.tolist()

        # selecting stars within FoV
        for i, pos in enumerate(star_pos):
            if vector_angle(orientation, pos, True) < fov_2:
                within_fov[i] = True

        return within_fov
