from os.path import abspath, dirname, join
from time import gmtime, strftime

import h5py
import matplotlib.pyplot as plt
import numpy as np

from utils import make_dir


class LogData(object):
    def __init__(self, name: str):
        self.name = name
        self.data = []

    def log(self, sample):
        """
        :param sample: data for logging specified as a numpy.array
        :return:
        """
        self.data.append(sample)

    def save(self, group):
        """
        :param group: the reference to the group level hierarchy of a .hdf5 file to save the data
        :return:
        """
        for key, val in self.__dict__.items():
            group.create_dataset(key, data=val)

    def load(self, group, decimate_step=1):
        """
        :param decimate_step:
        :param group: the reference to the group level hierarchy of a .hdf5 file to load
        :return:
        """
        # read in parameters
        # [()] is needed to read in the whole array if you don't do that,
        #  it doesn't read the whole data but instead gives you lazy access to sub-parts
        #  (very useful when the array is huge but you only need a small part of it).
        # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
        self.data = group["data"][()][::decimate_step]

    def plot(self):
        plt.plot(self.data, label=self.name)


class LogVec(LogData):

    def plot(self):
        data = np.array(self.data)
        for idx, coord in enumerate(["x", "y", "z"]):
            plt.plot(data[:, idx], label=f"{self.name}_{coord}")


class LogQuat(LogData):
    def plot(self):
        data = np.array(self.data)
        for idx, coord in enumerate(["w", "x", "y", "z"]):
            plt.plot(data[:, idx], label=f"{self.name}_{coord}")


class Logger(object):
    def __init__(self, sim_name, timestamp=None, log_dir=None,
                 data_items=(
                         "omega",
                         "omega_pred",

                         "angles",
                         "angles_pred",
                         "angles_target",
                         "angles_ref_pred",

                         "angles_esoq",
                         "esoq_cond_fro",
                         "esoq_cond_inf",
                         "esoq_cond_inf_neg",
                         "esoq_lambda_max",
                         "esoq_q_norm",
                         "angles_st",

                         "ads_error_angles",
                         "ads_ref_error_angles",
                         "acs_error_angles",

                         "mtq_m",
                         "mtq_torques",

                         "hybrid_torques",

                         "rw_torques",
                         "h_rw",

                         "magnet",
                         "magnet_abc",

                         "dist_torques",
                         "dist_torques_abc",

                         "sun",
                         "sun_abc"),
                        log_type=LogVec
                 ):
        """
        Creates a TemporalLogger object. If the folder structure is nonexistent, it will also be created
        :param sim_name: name of the simulation
        :param timestamp: timestamp as a string
        :param log_dir: logging directory, if it is None, then logging will be at the same hierarchy level as src/
        :param log_type: type of the data to log, defaults to LogVec
        """
        super().__init__()
        self.timestamp = strftime("%Y-%m-%d %H_%M_%S", gmtime()) if timestamp is None else timestamp
        # file structure
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log" if log_dir is None else log_dir)
        self.data_dir = join(self.base_dir, sim_name)
        make_dir(self.base_dir)
        make_dir(self.data_dir)

        # data
        self.data_items = data_items
        for data in self.data_items:
            self.__dict__[data] = log_type(data)

    def log(self, **kwargs):
        """
        Function for storing the new values of the given attribute
        :param **kwargs:
        :return:
        """
        for key, value in kwargs.items():
            self.__dict__[key].log(value)

    def save(self):
        """
        Saves the temporal statistics into a .hdf5 file
        :param **kwargs:
        :return:
        """
        with h5py.File(join(self.data_dir, 'time_log_' + self.timestamp + '.hdf5'), 'w') as f:
            for arg in self.data_items:
                self.__dict__[arg].save(f.create_group(arg))

    def load(self, filename, decimate_step=1):
        """
        Loads the temporal statistics and fills the attributes of the class
        :param decimate_step:
        :param filename: name of the .hdf5 file to load
        :return:
        """
        if not filename.endswith('.hdf5'):
            filename = filename + '.hdf5'

        with h5py.File(join(self.data_dir, filename), 'r') as f:
            for key, value in self.__dict__.items():
                if isinstance(value, LogData):
                    value.load(f[key], decimate_step)

    def plot(self, *args):
        for arg in args:
            if arg in self.__dict__.keys():  # and isinstance(self.__dict__[arg], LogData):
                self.__dict__[arg].plot()
        plt.title("Temporal evolution of time series")


