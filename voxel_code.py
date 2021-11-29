"""
This code is input a .pdb file, calculate the distance between all atoms to voxel centre atom.
If the distance between the atom and centre atom is in the voxel, the corresponding place of x, y, z-axis and the atom
index would be increased by 1.
The first function would return the voxel result, and the last process would save the result to a .npy file for further.
"""

# Import libraries
import os
import glob
import numpy as np
from biopandas.pdb import PandasPdb

from split_data import split_data


def input_file(file):
    # Read .pdb file
    ppdb = PandasPdb()
    ppdb.read_pdb(file)

    # Load data
    atom = ppdb.df["ATOM"]

    coordinates = atom[["atom_name", "x_coord", "y_coord", "z_coord", "element_symbol", "b_factor"]]
    coordinates_df = coordinates.copy()
    # Adding a new column for change the element symbol to index
    coordinates_df["element_index"] = coordinates_df["element_symbol"]
    # Replace the element N, C, O, S to corresponding index
    coordinates_df["element_index"].replace({"N": 0,
                                             "C": 1,
                                             "O": 2,
                                             "S": 3}, inplace=True)

    centre_element_name = "CA"  # Set CA be the centre element
    centre_element_df = coordinates[coordinates["atom_name"].isin([centre_element_name])].reset_index().drop(["index"],
                                                                                                             axis=1)
    # Set the centre element corresponding index to 0, and won't affect the distance calculation
    centre_element_df["centre_element_index"] = 0

    # Get coordinates of all element and centre element and their index
    all_element_coord = coordinates_df[["x_coord", "y_coord", "z_coord", "element_index"]].values
    centre_coord = centre_element_df[["x_coord", "y_coord", "z_coord", "centre_element_index"]].values
    confident_val = centre_element_df[["b_factor"]].values
    confident_value = confident_val.reshape(len(confident_val))

    def inside_test(points, centre_points):
        L = 20  # Length of the voxel
        l = int(L / 2)

        a = [0, 0, 0, 0]
        a = a * L * L * L  # Create a 20x20x20 voxel
        atom_count = np.array(a, dtype=np.uint8).reshape((L, L, L, 4))

        # Distance calculation
        distance = np.array(np.floor(points - centre_points), dtype=np.uint8) + np.array(
            [(l, l, l, 0)] * len(coordinates), dtype=np.uint8)

        # Check is in the voxel
        for v in distance:
            if (0 > v[0] or v[0] > L - 1) or (0 > v[1] or v[1] > L - 1) or (0 > v[2] or v[2] > L - 1):
                continue

            # v[0], v[1], v[2] represent the coordinate of x, y, z; v[3] represent the element index
            atom_count[v[0], v[1], v[2], v[3]] += 1

        return atom_count

    def check_for_all_elements(points, centre_coords):
        _res = np.array([inside_test(points, i) for i in centre_coords])

        return _res

    check_elements = check_for_all_elements(all_element_coord, centre_coord)

    return check_elements, confident_value


def output_file(file_name, result):
    # Save numpy.arrays into .npy format
    return np.save(file_name, result)


def main():
    path_raw_data = "./UP000005640_9606_HUMAN"
    path_result = "./Voxel_Result"
    file_lst = glob.glob(path_raw_data + "/*")

    # Check is the folder exist, if not, create the folder
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    filename_lst = [os.path.basename(i) for i in file_lst]
    new_filename_lst_coor = []
    new_filename_lst_confi = []

    for filename in filename_lst[:50]:  # Save the first 50 .pdb files inside_test result of each residue
        f_name = os.path.splitext(filename)[0] + "_coor.npy"
        new_filename_lst_coor.append(f_name)

    for filename in filename_lst[:50]:  # Save the first 50 .pdb files confident result of each residue
        f_name = os.path.splitext(filename)[0] + "_confi.npy"
        new_filename_lst_confi.append(f_name)

    for filename in filename_lst[:50]:  # Test for the first 50 .pdb file
        new_file_path_coor = os.path.join(path_result, new_filename_lst_coor[filename_lst.index(filename)])
        new_file_path_confi = os.path.join(path_result, new_filename_lst_confi[filename_lst.index(filename)])

        result = input_file(os.path.join(path_raw_data, filename))

        if not os.path.exists(new_file_path_coor):
            output_file(new_file_path_coor, result[0])

        if not os.path.exists(new_file_path_confi):
            output_file(new_file_path_confi, result[1])

        print(f"{filename_lst.index(filename) + 1}/{50}")

    split_data(path_result)


main()
