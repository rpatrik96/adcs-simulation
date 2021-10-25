from os import makedirs
from os.path import isdir


def make_dir(dirname):
    if not isdir(dirname):
        makedirs(dirname)