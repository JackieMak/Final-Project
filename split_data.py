"""
Split data into three sets, 60% of data is training set, 20% of data is validation set and 20% of data in testing set.
"""

# Import libraries
import os
import glob
import math
import random
import shutil


def split_data(path):
    # path = "./Voxel_Result"
    file_lst = glob.glob(path + "/*")

    # Get all the files in the folder
    filename_lst = [os.path.basename(i) for i in file_lst]
    filename_lst_coor = sorted(list(filter(lambda x: x.endswith("_coor.npy"), filename_lst)))
    filename_lst_confi = sorted(list(filter(lambda x: x.endswith("_confi.npy"), filename_lst)))

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(filename_lst_coor)
    random.seed(randnum)
    random.shuffle(filename_lst_confi)

    for i in range(len(filename_lst_coor)):
        if i < math.floor(0.6 * len(filename_lst_coor)):
            new_path = os.path.join(path, "Training_set")
        elif i < math.floor(0.8 * len(filename_lst_coor)):
            new_path = os.path.join(path, "Validation_set")
        else:
            new_path = os.path.join(path, "Testing_set")

        # If the file is not exist, create the file
        if os.path.exists(new_path) == 0:
            os.makedirs(new_path)

        shutil.move(os.path.join(path, filename_lst_coor[i]), os.path.join(new_path, filename_lst_coor[i]))
        shutil.move(os.path.join(path, filename_lst_confi[i]), os.path.join(new_path, filename_lst_confi[i]))
