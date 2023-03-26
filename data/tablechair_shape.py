import os, sys
sys.path.insert(1, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt

from data.distance import *
from data.utils import *


# porportion based on utils's 3d chair and rectable models
table_width, table_height = 0.625, 0.275  # in [-1,1], 1/4 of original # complete length (not half)
chair_width, chair_height = 0.1112, 0.1310 # 1/5 of original
chair2_width, chair2_height = chair_width, chair_height
table_distance_min, table_distance_max = table_width+0.1, table_width+0.3
x_off, y_off = max(chair_width, chair2_width)+0.05, table_height/2+max(chair_height, chair2_height)/2+0.01  # using the larger of the 2 chairs

chairs_condor = h5_to_dictionary(os.path.join(os.getcwd(), "ConDor_torch/preprocessed_condor_output/Chair.h5"))
chair1_sha = np.mean(chairs_condor[chair_jid]['points_code'], axis=0) # (1024, 128) -> (128,) numpy array
chair2_sha = np.mean(chairs_condor[chair2_jid]['points_code'], axis=0) 

tables_condor = h5_to_dictionary(os.path.join(os.getcwd(), "ConDor_torch/preprocessed_condor_output/Table.h5"))
table_sha = np.mean(tables_condor[rectable_jid]['points_code'], axis=0)

    

## DATA GENERATION: utils
def table_to_chair(table):
    """ Data is to be in range [-1, 1] for x and y, and there are 2 tables of table_width and table_height.
    
        table: table center coordinates, numpy array of shape (2,) 
        Return: chair positions and chir angles, both as numpy array of shape (6, 2)
    """
    chair_pos = [ [table[0]-x_off, table[1]+y_off], [table[0], table[1]+y_off], [table[0]+x_off, table[1]+y_off],   # 1st row
                  [table[0]-x_off, table[1]-y_off], [table[0], table[1]-y_off], [table[0]+x_off, table[1]-y_off] ]  # 2nd row
    chair_pos = np.array(chair_pos) 
    # orientation
    chair_ang = np.zeros((6, 2))
    chair_ang[0:3,:] = np.array([np.cos(np.pi), np.sin(np.pi)]) # 1st row [-1, 0] - 180 degree from pos y
    chair_ang[3:6,:] = np.array([np.cos(0), np.sin(0)])         # 2nd row [1, 0]  - 0 degree faces pos y

    return chair_pos, chair_ang


def _gen_tablechair_shape_batch(batch_size):
    """ Generate scene of 2 tables, each with 6 chairs. 

        batch_pos: position has size [batch_size, 7, 2]=[x, y], where [:,0:2,:] are the 2 tables,
                   and the rest are the chairs.
        batch_ang: orientation has size [batch_size, 7, 2]=[cos(th), sin(th)]
        batch_siz: [batch_size, 7, 2] = [width, height]
        batch_cla: shape has size [batch_size, 7, 2], represents type of object with one hot encoding,
                   specifically [1,0] refers to table and [0,1] refers to chairs
    """
    batch_pos = np.zeros((batch_size, 7, 2))
    batch_ang = np.zeros((batch_size, 7, 2))
    batch_siz = np.zeros((batch_size, 7, 2))
    batch_cla = np.zeros((batch_size, 7, 2))
    batch_invsha = np.zeros((batch_size, 7, 128))

    # table (1 per scene)
    ## position coordinates
    table_x = np.random.uniform(-1+table_width/2, 1-table_width/2, size=(batch_size,1,1))
    table_y = np.random.uniform(-1+table_height,  1-table_height,  size=(batch_size,1,1)) # also need space for chairs
    table = np.concatenate([table_x, table_y], axis=2) # batch_size, 1, 2
    batch_pos[:,0:1,:] = table # in [-1, 1]
    ## orientation
    ang_rad = np.random.choice([0,np.pi], size=(batch_size,1,1)) # 0 degree is horizontal (facing pos y)
    batch_ang[:,0:1,0:1] = np.cos(ang_rad)
    batch_ang[:,0:1,1:2] = np.sin(ang_rad)
    ## size, class, shape
    batch_siz[:,0:1,:] = np.array([table_width, table_height]) # in [-1, 1]
    batch_cla[:,0:1,:] = np.array([1,0])
    batch_invsha[:,0:1,:] = table_sha
    
    # chairs
    for batch_i in range(batch_size):
        chair_pos, chair_ang = table_to_chair(batch_pos[batch_i,0,0:2])
        batch_pos[batch_i,1:7,:] = chair_pos #[6, 2]
        batch_ang[batch_i,1:7,:] = chair_ang #[6, 2]

    batch_siz[:,1:4,:] = np.array([chair_width, chair_height])  # in [-1, 1]
    batch_siz[:,4:7,:] = np.array([chair2_width, chair2_height])  # in [-1, 1]
    batch_cla[:,1:7,:] = np.array([0,1])
    if np.random.rand()<0.5:
        batch_invsha[:,1:4,:] = chair1_sha # top row
        batch_invsha[:,4:7,:] = chair2_sha # bottom row
    else:
        batch_invsha[:,1:4,:] = chair2_sha # top row
        batch_invsha[:,4:7,:] = chair1_sha # bottom row  

    batch_sha = np.concatenate([batch_siz, batch_cla, batch_invsha], axis=2)  # [batch_size, 7, 2+2+128=132]

    return batch_pos, batch_ang, batch_sha



## DATA GENERATION: public API
def gen_data_tablechair_shape_bimodal(half_batch_size, noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4,
                                     clean_noise_level_stddev=0.01, clean_angle_noise_level_stddev=np.pi/90):
    """ Generates a scene of 1 table with 2 sets of chairs, one on each side of the table. The 2 sets of chairs have different shapes.
    """
    clean_orig_pos, clean_orig_ang, clean_orig_sha = _gen_tablechair_shape_batch(half_batch_size)
    clean_pos = np_add_gaussian_gaussian_noise(clean_orig_pos, noise_level_stddev=clean_noise_level_stddev)
    clean_ang = np_add_gaussian_gaussian_angle_noise(clean_orig_ang, noise_level_stddev=clean_angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    noisy_orig_pos, noisy_orig_ang, noisy_orig_sha = _gen_tablechair_shape_batch(half_batch_size)
    noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
    noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    clean_input = np.concatenate([clean_pos, clean_ang, clean_orig_sha], axis=2) # [half_batch_size, 7, 136]
    noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, 7, 136]
    input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, 7, 136]

    clean_labels =  np.concatenate([clean_orig_pos, clean_orig_ang], axis=2)  # disregard shape NOTE: perturbation small enough to do direct correspondence
    noisy_labels = np.zeros((half_batch_size, 7, 4))
    for scene_idx in range(half_batch_size):
        # only 1 table
        noisy_labels[scene_idx, 0:1, 0:2] = noisy_orig_pos[scene_idx, 0:1, :]
        noisy_labels[scene_idx, 0:1, 2:4] = noisy_orig_ang[scene_idx, 0:1, :]

        # earthmover distance assignment for the 2 sets of chairs
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 1:4, :]] # array of tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 1:4, :]] # array of tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 1:4, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 1:4, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, 1:4, 2:4] = np.array(chair_assign_ang)

        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 4:7, :]] # array of tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 4:7, :]] # array of tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 4:7, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 4:7, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, 4:7, 2:4] = np.array(chair_assign_ang)

    labels = np.concatenate([clean_labels, noisy_labels], axis=0)

    return input, labels



## VISUALIZATION: 2D

def visualize_tablechair_shape(scene, fp="tablechair_shape.jpg", title="table_horizontal"):
    """ scene has shape [nobj, pos_d+ang_d+sha_d+cla_dim], where first 2 rows are tables.
        traj, if given, has shape [iter, nobj, pos_d+ang_d]
        Difference from visualize functinos in utils.py: more coherent with the visualization
        function for 3d front, mainly the 90-degree rotation of angles, and combine both one-shot
        visualization and denoising visualization in one function through the argument traj.
    """
    siz = scene[:, 4:6]  # bbox lengths
    arrows = np_rotate(scene[:,2:4], np.zeros((scene.shape[0],1))+np.pi/2) # rot about origin as these are directional vectors
        # NOTE: add 90 deg because of offset between 0 degree angle visualization (1,0) and 0 degree obj, which starts off facing pos y/(0, 1)

    fig = plt.figure(dpi=300, figsize=(5,5))
    # initial & final
    table = plt.scatter(x=scene[:1,0], y=scene[:1,1],  c="#2ca02c", label="table") # green
    chair = plt.scatter(x=scene[1:4,0], y=scene[1:4,1], c="#1f77b4", label=f"chair 1") # blue
    chair2 = plt.scatter(x=scene[4:7,0], y=scene[4:7,1], c="#1fafb4", label=f"chair 2") # cyan

    # orientation
    plt.quiver(scene[:1,0], scene[:1,1], arrows[:1,0], arrows[:1,1], color="#2ca02c", label="table", width=0.005)
    plt.quiver(scene[1:4,0], scene[1:4,1], arrows[1:4,0], arrows[1:4,1], color="#1f77b4", label="chair 1", width=0.005)
    plt.quiver(scene[4:7,0], scene[4:7,1], arrows[4:7,0], arrows[4:7,1], color="#1fafb4", label="chair 2", width=0.005)
   
    for o_i in range(7):
        if o_i < 1:
            c="#2ca02c" 
        elif o_i < 4:
            c= "#1f77b4"
        else:
            c = "#1fafb4"
        plt.gca().add_patch(plt.Rectangle(
                                (scene[o_i,0]-siz[o_i,0]/2, scene[o_i,1]-siz[o_i,1]/2), 
                                siz[o_i,0], siz[o_i,1], linewidth=1, edgecolor=c, facecolor='none',
                                transform = mt.Affine2D().rotate_around(scene[o_i,0], scene[o_i,1], np.arctan2(scene[o_i,3], scene[o_i,2])) +plt.gca().transData 
                            )) # original orietantion unaffected (table 30 deg = rotate horizontal table by 30 degree + arrow points to 120)


    plt.legend(handles=[table, chair, chair2], fontsize=7)
    plt.gca().set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2])
    plt.title(f"{title}", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(fp)
    plt.close(fig)



if __name__ == '__main__':
    input, labels = gen_data_tablechair_shape_bimodal(2)  # [batch_size, 7, 136]  
    for scene_i in range(input.shape[0]):
        visualize_tablechair_shape(input[scene_i],  fp=f"tablechair_shape_{scene_i}.jpg", title=f"tablechair_shape: {scene_i}")
