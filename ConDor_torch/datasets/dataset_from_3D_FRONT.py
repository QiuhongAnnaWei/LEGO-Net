import numpy as np
import h5py
import argparse
import open3d as o3d
import os
import glob
import tqdm

def read_category_file(category_file):
    txt_file = open(category_file, "r")
    file_content = txt_file.read()
    file_content = file_content.split("\n")
    # print(file_content)
    return file_content

def get_model_files(dataset_root, category_file, parts = 100, model_file_name = "normalized_model.obj"):

    model_files = []

    model_folders = read_category_file(category_file)
    # print(model_folders)

    part_directories = []
    #for i in range(parts):
    #    directory = os.path.join(dataset_root, "") + "3D-FUTURE-model-part%d" % i
        # print(directory)
    #    if os.path.exists(directory):
            # print(directory)
    #        part_directories.append(directory)
    

    part_directories = [os.path.join(dataset_root, "")]# + "3D-FUTURE-model-part%d" % i

    for model_directory in model_folders:
        for part_directory in part_directories:
            process_directory = os.path.join(part_directory, "") + model_directory
            # print(process_directory)
            if os.path.exists(process_directory):
                # print(process_directory)
                model_files.append(os.path.join(process_directory, "") + model_file_name)
                
    return model_files
    

def convert_mesh_to_pointcloud(mesh_file, num_points):

    try:
        model_mesh = o3d.io.read_triangle_mesh(mesh_file)
        model_pointcloud = model_mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=1)
        return model_pointcloud
    except:
        return None


def generate_directory(args):
    
    model_files = get_model_files(args.dataset_root, args.category_file)

    model_pcds = []

    if args.num_models == -1:
        total_models = len(model_files)
    else:
        total_models = args.num_models

    for mesh_idx in tqdm.tqdm(range(min(len(model_files), total_models))):
        mesh_file = model_files[mesh_idx]
        model_pointcloud = convert_mesh_to_pointcloud(mesh_file, args.num_points)
        if model_pointcloud is None:
            continue
        np_pcd = np.array(model_pointcloud.points)
        model_pcds.append(np_pcd)


    model_pcds = np.stack(model_pcds, axis = 0)
    print("Final dataset size: ", model_pcds.shape)
    write_dict_to_h5({"data": model_pcds}, args.output_h5_file)

def write_dict_to_h5(dictionary, path):

    output_h5_file = path
    output_directory = os.path.dirname(path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dataset_file = h5py.File(output_h5_file, "w")    
    for key in dictionary.keys():
        dataset_file.create_dataset(key, data = dictionary[key])
    dataset_file.close()

def get_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset_root", type = str, required=True)
    parser.add_argument("--category_file", type=str, required=True)
    parser.add_argument("--num_models", type=int, default = -1)
    parser.add_argument("--num_points", type=int, default = 2048)
    parser.add_argument("--output_h5_file", type=str, required=True)
    
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_arguments()

    generate_directory(args)
    pass
