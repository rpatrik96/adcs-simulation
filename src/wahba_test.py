from quaternion import Quaternion
import numpy as np
from esoq2 import ESOQ2
from transforms import Rotation

def create_k(b:np.array, q:Quaternion):

    b = normalize(b)

    r = q.rotate_vector(b)


    B = np.outer(r,b)

    S = B+B.T
    z = np.cross(r,b)
    K = np.zeros((4,4))

    K[0:3,0:3] = S-B.trace()*np.eye(3)
    K[-1, :3] = z.T
    K[:3, -1] = z
    K[-1, -1] = B.trace()

    return K


def normalize(b):
    b = b / np.linalg.norm(b)
    return b



q = Quaternion(np.array([0.5, 2, -3, -1.02]))
b1 = normalize(np.array([1, 2, -.5]))


K1 = create_k(b1, q)
b2 = normalize(np.array([3, -2, -1.5]))
K2 = create_k(b2, q)

b3 = normalize(np.array([-2.5364,0.7654, -0.85465]))

esoq = ESOQ2()

esoq.step(np.array([b1, b2]), np.array([q.rotate_vector(b1), q.rotate_vector(b2)]))

def rot24d(r):
    R4 = np.eye(4)
    R4[:3,:3] = r

    return R4