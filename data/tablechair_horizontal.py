import os, sys
sys.path.insert(1, os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt

from data.distance import *
from data.utils import *


# porportion based on utils's 3d chair and rectable models
table_width, table_height = 0.625, 0.275  # in [-1,1], 1/4 of original # complete length (not half)
chair_width, chair_height = 0.1112, 0.1310 # 1/5 of original
table_distance_min, table_distance_max = table_width+0.1, table_width+0.3
x_off, y_off = chair_width+0.05, table_height/2+chair_height/2+0.01 

penalty_headers = ["dist moved", "tab y", "tab dist", "rel EMD(L)", "rel EMD(R)", "too close", "clean CD", "clean EMD", "penalty"]


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


def _gen_tablechair_horizontal_batch(batch_size, horizontal_align):
    """ Generate scene of 2 tables, each with 6 chairs. 

        batch_pos: position has size [batch_size, 14, 2]=[x, y], where [:,0:2,:] are the 2 tables,
                   and the rest are the chairs.
        batch_ang: orientation has size [batch_size, 14, 2]=[cos(th), sin(th)]
        batch_siz: [batch_size, 14, 2] = [width, height]
        batch_cla: shape has size [batch_size, 14, 2], represents type of object with one hot encoding,
                   specifically [1,0] refers to table and [0,1] refers to chairs
    """
    batch_pos = np.zeros((batch_size, 14, 2))
    batch_ang = np.zeros((batch_size, 14, 2))
    batch_siz = np.zeros((batch_size, 14, 2))
    batch_cla = np.zeros((batch_size, 14, 2))

    # table
    ## position coordinates
    left_table_x = np.random.uniform(-1+table_width/2, 0-table_width/2, size=(batch_size,1,1))
    left_table_y = np.random.uniform(-1+table_height,  1-table_height,  size=(batch_size,1,1)) # also need space for chairs
    left_table = np.concatenate([left_table_x, left_table_y], axis=2) # batch_size, 1, 2
    batch_pos[:,0:1,:] = left_table # in [-1, 1]
    if horizontal_align:
        table_distance = np.random.uniform(table_distance_min, table_distance_max, size=(batch_size,1,1))
        right_table = np.concatenate([left_table[:,:,0:1]+table_distance, left_table[:,:,1:2]], axis=2) # batch_size, 1, 2
    else:
        right_table_x = np.random.uniform(0+table_width/2, 1-table_width/2, size=(batch_size,1,1))
        right_table_y = np.random.uniform(-1+table_height, 1-table_height, size=(batch_size,1,1))  # also need space for chairs
        right_table = np.concatenate([right_table_x, right_table_y], axis=2) # batch_size, 1, 2
    batch_pos[:,1:2,:] = right_table # in [-1,1]
    ## orientation
    ang_rad = np.random.choice([0,np.pi], size=(batch_size,2,1)) # 0 degree is horizontal (facing pos y)
    batch_ang[:,0:2,0:1] = np.cos(ang_rad)
    batch_ang[:,0:2,1:2] = np.sin(ang_rad)
    ## size
    batch_siz[:,0:2,:] = np.array([table_width, table_height]) # in [-1, 1]
    ## class
    batch_cla[:,0:2,:] = np.array([1,0])
    
    # chairs
    for batch_i in range(batch_size):
        chair_pos, chair_ang = table_to_chair(batch_pos[batch_i,0,0:2])
        batch_pos[batch_i,2:8,:] = chair_pos #[6, 2]
        batch_ang[batch_i,2:8,:] = chair_ang #[6, 2]
        chair_pos, chair_ang = table_to_chair(batch_pos[batch_i,1,0:2])
        batch_pos[batch_i,8:14,:] = chair_pos #[6, 2]
        batch_ang[batch_i,8:14,:] = chair_ang #[6, 2]

    batch_siz[:,2:14,:] = np.array([chair_width, chair_height])  # in [-1, 1]
    batch_cla[:,2:14,:] = np.array([0,1])

    batch_sha = np.concatenate([batch_siz, batch_cla], axis=2)  # [batch_size, 14, 2+2=4]

    return batch_pos, batch_ang, batch_sha


def compute_scene_penalty(traj):
    """ For analytics purposes.

        traj: np array of [num_snapshot, 14, 2] representing trajectory of objects in a scene of 14 obj. traj[i] is a scene. 
        scene: [14, 2] array of xy coords of all objects in a scene, where [0:2, :] are the tables, [2:8, :] corresponds
              to chairs surrounding table scene[0] and [8:14, ] corresponds to chairs surrounding table scene[1].

        Returns: 1-d numpy array of scalar metrics representing penalty of the scene (the higher the worse), each with
                description in comments. The last element represents the overall penalty (weighted sum of metrics).
    """
    orig_scene, scene = traj[0], traj[-1] # scene refers to final scene (result of denoising)
    left_table, right_table = scene[0], scene[1]
    left_chairs, right_chairs = scene[2:8, :], scene[8:14, :]
    
    # 1. distance moved
    distance_moved = 0
    for obj_i in range(scene.shape[0]):
        distance_moved+=euclidean_distance(orig_scene[obj_i], scene[obj_i]) 

    # 2. horizontal alignment of table (in units of abs dist)
    table_y_offset = abs(left_table[1]-right_table[1])
    
    # 3. horizontal distance between tables (in units of abs dist)
    table_x_offset = abs(left_table[0]-right_table[0])
    table_distance_offset = 0
    if table_x_offset < table_distance_min and table_x_offset > table_distance_max:
        table_distance_offset = min(abs(table_x_offset-table_distance_min), abs(table_x_offset-table_distance_max))

    # 4. earthmover distance assignment for each set of 6 chairs (in units of abs dist)
    left_t_chairgt, _ = table_to_chair(left_table)
    right_t_chairgt, _ = table_to_chair(right_table) #[6, 2]
    p1 = [tuple(chair) for chair in left_chairs] # array of 6 tuples
    p2 = [tuple(chair) for chair in left_t_chairgt]
    _, _, _, relative_emd_left = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    p1 = [tuple(chair) for chair in right_chairs]
    p2 = [tuple(chair) for chair in right_t_chairgt]
    _, _, _, relative_emd_right = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 

    # 5. penalize if multiple objects are mapped to the same position (exp(-abs dist))
    too_close = 0
    for obj_i in range(scene.shape[0]):
        for other_obj in scene[obj_i+1:]: # [num_other_obj, 2]
            obj = scene[obj_i] #(2, )
            dist = euclidean_distance(obj, other_obj) 
            if dist < 0.125: too_close+= math.exp(-10 * dist)/2 # 0->0.5, 0.125 -> 0.143

    # 6. chamfer distance from a clean scene constructed based on table positions (avg of table x coords + gap 
    #    distance evenly distrubted/shrunk, avg of table y corod)
    table_gt_y = left_table[1]+right_table[1]/2
    if table_x_offset >= table_distance_min and table_x_offset <= table_distance_max:
        left_table_gt = [left_table[0], table_gt_y]
        right_table_gt = [right_table[0], table_gt_y]
    elif table_x_offset < table_distance_min:
        shortage = table_distance_min-table_x_offset
        left_table_gt = [left_table[0]-shortage/2, table_gt_y]
        right_table_gt = [right_table[0]+shortage/2, table_gt_y]
    elif table_x_offset > table_distance_max:
        excess = table_x_offset-table_distance_max
        left_table_gt = [left_table[0]+excess/2, table_gt_y]
        right_table_gt = [right_table[0]-excess/2, table_gt_y]
    scene_gt = np.zeros(scene.shape) #[14,2]
    scene_gt[0] = np.array(left_table_gt)
    scene_gt[1] = np.array(right_table_gt)
    scene_gt[2:8,:], _ = table_to_chair(left_table_gt)
    scene_gt[8:14,:], _ = table_to_chair(right_table_gt)
    clean_CD = chamfer_distance(scene, scene_gt)

    # 7. total cost of free-for-all (disregarding types) emd assignment to the clean scene in 6
    p1 = [tuple(o) for o in scene]  # array of 14 tuples
    p2 = [tuple(o) for o in scene_gt]
    _, _, _, clean_EMD = earthmover_distance(p1, p2)

    penalty_list = [distance_moved, table_y_offset, table_distance_offset, relative_emd_left, relative_emd_right, 
                    too_close, clean_CD, clean_EMD]
    coeff_list = [1/10, 1, 1, 1, 1, 1, 1, 1]
    penalty_list.append(np.dot(np.array(penalty_list), np.array(coeff_list)))
    return penalty_list




## DATA GENERATION: public API
## 3 variations: bimodal (preferred), allnoisy, halfsplit
def gen_data_tablechair_horizontal_bimodal(half_batch_size, horizontal_align=True, scene_rot=False, abs_pos=True, abs_ang=True, noise_level_stddev=0.25, 
                                            angle_noise_level_stddev=np.pi/4, clean_noise_level_stddev=0.01, clean_angle_noise_level_stddev=np.pi/90):
    """ Preferred way of data generation. 
        
        Generates a scene of 2 tables, each with 6 chairs in rectangular formation. The tables may be horizontally aligned or not 
        (see more below). The chairs are at a fixed relative position from the table (at 0.25 from each other and from the table, 
        see the table_to_chair function for more).

        half_batch_size: number of clean scenes; number of noisy scenes 
        horizontal_align: if True, forces tables to have same y and at x-distance sampled from a fixed range; if False,
                          the two tables are independently generated, one for left half and one for right half (no collision).
        scene_rot: if True, apply scene-level rotation to all inputs and labels. Default range for rotation is [-pi/2, pi/2].
        abs_pos, abs_ang: if True, the returned labels are (final ordered assigned position/angles); otherwise, use relative positions
                          (=final ordered assigned position - initial position) or angles

        Intermediary: {clean, noisy}_{pos, ang, sha} have shape [batch_size, 14, 2]

        Returns: input: [batch_size, 14, 8] = [x, y, cos(th), sin(th), sizx, sizy, 1(0), 0(1)]
                 label: [batch_size, 14, 4] = [x, y, cos(th), sin(th)]
        
        Clean input's labels are identical to the input. Noisy input's labels are ordered assignment to clean position for each 
        object (through earthmover distance assignment), using the original clean sample (noisy_orig_pos) as target, and angle labels
        are the corresponding angles from the assignment.
       
        Note that this implementation takes a non-hierarhical, non-scene-graph, free-for-all approach: noise is added randomly to
        each object regardless of type, and emd assignment is done freely among all objects of a class. 
    
        bimodal-specific:
        input:  {clean scene + small gaussian guassian nosie, clean scene + large gaussian gaussian nosie}.
        labels: {clean sene without noise, clean scene without noise + earthmover distance assignemnt}
    """
    clean_orig_pos, clean_orig_ang, clean_orig_sha = _gen_tablechair_horizontal_batch(half_batch_size, horizontal_align)
    clean_pos = np_add_gaussian_gaussian_noise(clean_orig_pos, noise_level_stddev=clean_noise_level_stddev)
    clean_ang = np_add_gaussian_gaussian_angle_noise(clean_orig_ang, noise_level_stddev=clean_angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    noisy_orig_pos, noisy_orig_ang, noisy_orig_sha = _gen_tablechair_horizontal_batch(half_batch_size, horizontal_align)
    noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
    noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    clean_input = np.concatenate([clean_pos, clean_ang, clean_orig_sha], axis=2) # [half_batch_size, 14, 6]
    noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, 14, 6]
    input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, 14, 6]


    clean_labels =  np.concatenate([clean_orig_pos, clean_orig_ang], axis=2)  # disregard shape NOTE: perturbation small enough to do direct correspondence
    if not abs_pos: # relative
        clean_labels[:, :, 0:2] = clean_orig_pos-clean_pos
    if not abs_ang: # relative
        angles = np.squeeze(np_angle_between( clean_ang, clean_orig_ang )) # (B, 14, 1) from clean to clean_orig_ang
        clean_labels[:, :, 2] = np.cos(angles) # 1d
        clean_labels[:, :, 3] = np.sin(angles)

    noisy_labels = np.zeros((half_batch_size, 14, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:2, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 0:2, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 0:2, :]) # 2x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 0:2, 0:2] = np.array(table_assignment)
        noisy_labels[scene_idx, 0:2, 2:4] = np.array(table_assign_ang)
        if not abs_pos: # relative:
            noisy_labels[scene_idx, 0:2, 0:2] = np.array(table_assignment) - noisy_pos[scene_idx, 0:2, :]
        if not abs_ang: # relative
            angles = np.squeeze(np_angle_between( noisy_ang[scene_idx:scene_idx+1, 0:2, :], np.array([table_assign_ang]) )) # (2,) from current ang -> assigned ang
            noisy_labels[scene_idx, 0:2, 2] = np.cos(angles) # 1d
            noisy_labels[scene_idx, 0:2, 3] = np.sin(angles)

        # earthmover distance assignment for chairs
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 2:14, :]] # array of 12 tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 2:14, :]] # array of 12 tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 2:14, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 2:14, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, 2:14, 2:4] = np.array(chair_assign_ang)
        if not abs_pos: # relative
            noisy_labels[scene_idx, 2:14, 0:2] = np.array(chair_assignment) - noisy_pos[scene_idx, 2:14, :]
        if not abs_ang: # relative
            angles = np.squeeze(np_angle_between( noisy_ang[scene_idx:scene_idx+1, 2:14, :], np.array([chair_assign_ang]) )) # (12,), from current ang -> assigned ang
            noisy_labels[scene_idx, 2:14, 2] = np.cos(angles) # 1d
            noisy_labels[scene_idx, 2:14, 3] = np.sin(angles)

    labels = np.concatenate([clean_labels, noisy_labels], axis=0)

    if scene_rot: input, labels = apply_rot(input, labels, ang_min=-np.pi/2, ang_max=np.pi/2)
    
    return input, labels


def gen_data_tablechair_horizontal_allnoisy( half_batch_size, horizontal_align=True, scene_rot=False, noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4):
    """ allnoisy-specific:
        input: {clean scene + gaussian gaussian noise}
        label: {corresponding clean scenes})
    """
    clean_pos, clean_ang, clean_sha = _gen_tablechair_horizontal_batch(half_batch_size*2, horizontal_align)
    noisy_pos = np_add_gaussian_gaussian_noise(clean_pos, noise_level_stddev=noise_level_stddev)
    noisy_ang = np_add_gaussian_gaussian_angle_noise(clean_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    input = np.concatenate([noisy_pos, noisy_ang, clean_sha], axis=2) # [batch_size, 14, 8], concatenate creates copies

    labels = np.zeros((half_batch_size*2, 14, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:2, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in clean_pos[scene_idx, 0:2, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_idx, 0:2, :]) # 2x2: assigned position for each pt in p1 (in that order)
        labels[scene_idx, 0:2, 0:2] = np.array(table_assignment)
        labels[scene_idx, 0:2, 2:4] = np.array(table_assign_ang)

        # earthmover distance assignment for chairs
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 2:14, :]] # array of 12 tuples
        p2 = [tuple(pt) for pt in clean_pos[scene_idx, 2:14, :]] # array of 12 tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_idx, 2:14, :]) # 12x2: assigned position for each pt in p1 (in that order)
        labels[scene_idx, 2:14, 0:2] = np.array(chair_assignment)
        labels[scene_idx, 2:14, 2:4] = np.array(chair_assign_ang)

    if scene_rot: input, labels = apply_rot(input, labels, ang_min=-np.pi/2, ang_max=np.pi/2)
    
    return input, labels


def gen_data_tablechair_horizontal_halfsplit(half_batch_size, horizontal_align=True, scene_rot=False,
                                             noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4):
    """ halfsplit-specific:
        Clean pos and ang in input has no noise. 
    """

    clean_pos, clean_ang, clean_sha = _gen_tablechair_horizontal_batch(half_batch_size, horizontal_align) # NOTE: no noise for clean input and label
    noisy_orig_pos, noisy_orig_ang, noisy_orig_sha = _gen_tablechair_horizontal_batch(half_batch_size, horizontal_align)

    noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
    noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized

    clean_input = np.concatenate([clean_pos, clean_ang, clean_sha], axis=2) # [half_batch_size, 14, 6]
    noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, 14, 6]
    input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, 14, 6]

  
    clean_labels = np.copy(clean_input[:,:,0:4]) # disregard shape
    noisy_labels = np.zeros((half_batch_size, 14, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:2, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 0:2, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 0:2, :]) # 2x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 0:2, 0:2] = np.array(table_assignment)
        noisy_labels[scene_idx, 0:2, 2:4] = np.array(table_assign_ang)

        # earthmover distance assignment for chairs
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 2:14, :]] # array of 12 tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 2:14, :]] # array of 12 tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 2:14, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 2:14, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, 2:14, 2:4] = np.array(chair_assign_ang)

    labels = np.concatenate([clean_labels, noisy_labels], axis=0)

    if scene_rot: input, labels = apply_rot(input, labels, ang_min=-np.pi/2, ang_max=np.pi/2)
    
    return input, labels




## VISUALIZATION: 2D

def visualize_tablechair_horizontal(scene, fp="tablechair_horizontal.jpg", title="table_horizontal"):
    """ scene: [nobj, pos_d+ang_d+siz_d+cla_d], where first 2 rows are tables.

        Difference from visualize functinos in utils.py: more coherent with the visualization function for
        3d front, mainly has 90-degree rotation for angles
    """
    siz = scene[:, 4:6]  # bbox lengths
    arrows = np_rotate(scene[:,2:4], np.zeros((scene.shape[0],1))+np.pi/2) # rot about origin as these are directional vectors
        # NOTE: add 90 deg because of offset between 0 degree angle visualization (1,0) and 0 degree obj, which starts off facing pos y/(0, 1)

    fig = plt.figure(dpi=300, figsize=(5,5))
    # initial & final
    table = plt.scatter(x=scene[:2,0], y=scene[:2,1],  c="#2ca02c", label="table") # green
    chair = plt.scatter(x=scene[2:14,0], y=scene[2:14,1], c="#1f77b4", label=f"chair") # blue
    # orientation
    plt.quiver(scene[:2,0], scene[:2,1], arrows[:2,0], arrows[:2,1], color="#2ca02c", label="table", width=0.005)
    plt.quiver(scene[2:14,0], scene[2:14,1], arrows[2:14,0], arrows[2:14,1], color="#1f77b4", label="chair", width=0.005)
   
    for o_i in range(14):
        c="#2ca02c" if o_i < 2 else "#1f77b4"
        plt.gca().add_patch(plt.Rectangle(
                                (scene[o_i,0]-siz[o_i,0]/2, scene[o_i,1]-siz[o_i,1]/2), 
                                siz[o_i,0], siz[o_i,1], linewidth=1, edgecolor=c, facecolor='none',
                                transform = mt.Affine2D().rotate_around(scene[o_i,0], scene[o_i,1], np.arctan2(scene[o_i,3], scene[o_i,2])) +plt.gca().transData 
                            )) # original orietantion unaffected (table 30 deg = rotate horizontal table by 30 degree + arrow points to 120)

    plt.legend(handles=[table, chair], fontsize=7)
    plt.gca().set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2])
    plt.title(f"{title}", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(fp)
    plt.close(fig)



if __name__ == '__main__':
    input, labels = gen_data_tablechair_horizontal_bimodal(2, horizontal_align=True, scene_rot=False) # [batch_size, 14, 8]  
    for scene_i in range(input.shape[0]):
        visualize_tablechair_horizontal(input[scene_i], fp=f"tablechair_horizontal_{scene_i}.jpg", title=f"tablechair_horizontal: {scene_i}")
