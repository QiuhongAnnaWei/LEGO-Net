import h5py
import os
import numpy as np
from ConDor_torch.datasets import dataset_from_3D_FRONT as dataset_utils
import argparse

def get_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file_path", type = str, required=True)
    parser.add_argument("--output_directory", type = str, required=False, default = None)
    parser.add_argument("--train_split", type = float, required=False, default = 80)
    parser.add_argument("--val_split", type = float, required=False, default = 10)


    args = parser.parse_args()

    return args

def create_splits(args):

    data = h5py.File(args.file_path, "r")

    if args.output_directory is None:
        output_directory = os.path.dirname(args.file_path)
    else:
        output_directory = args.output_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_directory = os.path.join(output_directory, "")

    filename_save = os.path.basename(args.file_path)

    train_dataset, val_dataset, test_dataset = {}, {}, {}

    for key in data.keys():
        print(key)
        array = np.array(data[key])
        train_idx, val_idx = int(args.train_split/100.0 * array.shape[0]), int((args.train_split/100.0 + args.val_split/100.0) * array.shape[0])

        train_split = data[key][:train_idx]
        val_split = data[key][train_idx:val_idx]
        test_split = data[key][val_idx: ]


        print(train_idx, val_idx, array.shape[0] - val_idx)
        # print(train_split.shape, val_split.shape, test_split.shape)
        train_dataset[key] = train_split
        val_dataset[key] = val_split
        test_dataset[key] = test_split

    dataset_utils.write_dict_to_h5(train_dataset, output_directory + "train_" + filename_save)
    dataset_utils.write_dict_to_h5(val_dataset, output_directory + "val_" + filename_save)
    dataset_utils.write_dict_to_h5(test_dataset, output_directory + "test_" + filename_save)


if __name__=="__main__":

    args = get_arguments()
    create_splits(args)