"""
Load data in the .npy file
"""
# Import libraries
import glob
import numpy as np


def load_npy(filename):
    return np.load(filename)


class LoadFile:
    def __init__(self, training_path, validation_path, testing_path):
        self.training_path = training_path
        self.validation_path = validation_path
        self.testing_path = testing_path

    # Get all the files in the Training_set folder into a list
    def get_train_coor(self):
        train_lst = glob.glob(self.training_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_coor.npy"), train_lst)))

    def get_train_confi(self):
        train_lst = glob.glob(self.training_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_confi.npy"), train_lst)))

    def get_val_coor(self):
        val_lst = glob.glob(self.validation_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_coor.npy"), val_lst)))

    def get_val_confi(self):
        val_lst = glob.glob(self.validation_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_confi.npy"), val_lst)))

    def get_test_coor(self):
        test_lst = glob.glob(self.testing_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_coor.npy"), test_lst)))

    def get_test_confi(self):
        test_lst = glob.glob(self.testing_path + "/*")
        return sorted(list(filter(lambda x: x.endswith("_confi.npy"), test_lst)))
