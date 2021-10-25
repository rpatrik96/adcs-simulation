from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from alg_utils import vector_angle

Centroid = Tuple[float, float, float]


class VotingAlgorithm(object):
    def __init__(self, uncertainty=np.deg2rad(.03), max_stars=20):
        """

        :param uncertainty: uncertainty of the centroid calculation
        """
        super().__init__()
        self.uncertainty = uncertainty
        self.max_stars = max_stars

    def weighted_centroids(self, img: np.ndarray) -> List[Centroid]:
        """
        Calculates the centroids of the stars on the input image

        :param img: grayscale image
        :return: centroid coordinates on the image plane (negative coordinates can be present, as the principal point has coordinates (0,0)
                + pixel density (for classifying the stars)
        """

        """Preprocessing"""
        height, width = img.shape[0], img.shape[1]
        # thresholding (mainly for real images)
        img = (img > 25) * img

        # remove salt&pepper noise
        img = cv2.medianBlur(img, 3)

        """Contour calculation"""
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        """Centroid+density calculation"""
        centroids = []
        for idx, _ in enumerate(contours):
            # create mask + mask original image
            # this way only one object is included
            cont_mask = cv2.drawContours(np.zeros((height, width)), contours, idx, 1, -1).astype(np.uint8)
            masked_img = cont_mask * img

            # CoM calculation
            moments = cv2.moments(masked_img, False)
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # density calculation
            num_pix = (masked_img > 0).sum()
            density = masked_img.sum() / num_pix * 1.9

            # centroid coordinates should be centered around the principal axis,
            # i.e. the offset added for imaging purposes should be removed
            centroids.append((cx - width / 2., cy - height / 2., density))

        return centroids[:self.max_stars]

    def predict_voting(self, img: np.ndarray, star_pairs: pd.DataFrame, fov: float) -> Dict[int, np.ndarray]:
        """
        Implements the voting algorithm

        :param img: grayscale image
        :param star_pairs: catalogue containing star pairs
        :param fov: Field of View of the camera in radians
        :return: list of the identified stars
        """
        # calculate focal length
        height, width = img.shape[0], img.shape[1]
        focal_length = np.sqrt(width ** 2 + height ** 2) / (2 * np.tan(fov / 2))

        # calculate centroids
        centroids = self.weighted_centroids(img)
        print(len(centroids))

        # initialize the structure for votes
        self.votes = []
        num_stars = len(centroids)
        for i in range(num_stars):
            self.votes.append(dict())

        """vote based on the star pairs lut"""
        for i in range(num_stars - 1):
            for j in range(i + 1, num_stars):
                # calculate angular distance
                angle = self._star_angle(centroids[i], centroids[j], focal_length)

                """filter possible pairs"""
                # proceed only if angle within FoV
                if angle < fov:
                    # filter by angle
                    filtered_pairs = star_pairs[np.abs(star_pairs.angle - angle) < self.uncertainty]

                    # and by magnitude class
                    mag_class1 = filtered_pairs.mag1.apply(self._mag2class)
                    mag_class2 = filtered_pairs.mag2.apply(self._mag2class)
                    den_class_i = self._density2class(centroids[i][2])
                    den_class_j = self._density2class(centroids[j][2])
                    filtered_pairs = filtered_pairs[((mag_class1 == den_class_i) & (mag_class2 == den_class_j)) |
                                                    ((mag_class1 == den_class_j) & (mag_class2 == den_class_i))]
                    ids = filtered_pairs.id1.to_list() + filtered_pairs.id2.to_list()

                    self._vote(i, j, ids)
                else:
                    # should not be raised
                    raise NotImplementedError

        """assign the most probable star id"""
        assigned_id = []
        centroids2remove = []
        for i in range(num_stars):
            max_votes = -1
            for star, num_votes in self.votes[i].items():
                if num_votes > max_votes:
                    max_votes = num_votes
                    star_decision = star

            # save the assigned id
            if max_votes != -1:
                assigned_id.append(star_decision)
            else:
                centroids2remove.append(i)

        # remove centroids which did not get any votes
        num_stars -= len(centroids2remove)
        for c2r in centroids2remove:
            centroids.remove(centroids[c2r])

        """Validation"""
        # reset variable
        self.votes = []
        for i in range(num_stars):
            self.votes.append(dict())

        for i in range(num_stars - 1):
            for j in range(i + 1, num_stars):
                # as the catalog is built in a way that id1 is always smaller,
                # the query can be made simpler
                min_id = min(assigned_id[i], assigned_id[j])
                max_id = max(assigned_id[i], assigned_id[j])

                pair = star_pairs[(star_pairs.id1 == min_id) & (star_pairs.id2 == max_id)]
                if len(pair) and np.abs(pair.angle.item() - self._star_angle(centroids[i], centroids[j],
                                                                             focal_length)) < self.uncertainty:
                    self._vote(i, j, assigned_id[i], assigned_id[j])

        print(self.votes)
        # calculate maximum vote number
        max_valid_votes = -1
        for i in range(num_stars):
            for num_votes in self.votes[i].values():
                if num_votes > max_valid_votes:
                    max_valid_votes = num_votes

        # keep only that stars which have the maximum number of validation votes
        stars_found: dict = dict()
        for i in range(num_stars):
            for star, num_votes in self.votes[i].items():
                if num_votes == max_valid_votes and star not in stars_found:
                    star_vec = np.array([centroids[i][0], centroids[i][1], focal_length])
                    stars_found[star] = star_vec / np.linalg.norm(star_vec)
                    break

        return stars_found

    def _vote(self, i: int, j: int, id1, id2: Optional[int] = None) -> None:
        """

        :param i: first star index
        :param j: second star index
        :param id1: if id2==None, a list of ids, otherwise the id for star1
        :param id2: the id for star1
        :return:
        """
        if id2 is None:
            for star in id1:
                if star in self.votes[i].keys():
                    self.votes[i][star] += 1
                else:
                    self.votes[i][star] = 1

                if star in self.votes[j].keys():
                    self.votes[j][star] += 1
                else:
                    self.votes[j][star] = 1
        else:
            if id1 in self.votes[i].keys():
                self.votes[i][id1] += 1
            else:
                self.votes[i][id1] = 1

            if id2 in self.votes[j].keys():
                self.votes[j][id2] += 1
            else:
                self.votes[j][id2] = 1

    def _star_angle(self, c1: Centroid, c2: Centroid, focal_length: float) -> float:
        """
        Calculates the angle between two stars specified by their centroids

        :param c1: first centroid
        :param c2: second centroid
        :param focal_length: focal length of the camera in pixels
        :return: angle between the two star vectors in rad
        """
        star1 = np.array([c1[0], c1[1], focal_length])
        star2 = np.array([c2[0], c2[1], focal_length])
        angle = vector_angle(star1, star2, False)
        return angle

    def _density2class(self, density: float) -> int:
        """
        Converts pixel density to magnitude class
        :param density: pixel density
        :return: magnitude class
        """
        cl = 4
        if density > 236:
            cl = 1
        elif density > 176:
            cl = 2
        elif density > 100:
            cl = 3

        return cl

    def _mag2class(self, mag: float) -> int:
        """
        Converts magnitude to magnitude class
        :param mag: magnitude value
        :return: magnitude class
        """
        cl = 4
        if mag < 1.38:
            cl = 1
        elif mag < 2.76:
            cl = 2
        elif mag < 4.14:
            cl = 3

        return cl
