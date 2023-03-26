import os, sys
sys.path.insert(1, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt

from data.distance import *
from data.utils import *


ntable=2
nchair_min = 2
nchair_max = 7 # exclusive
maxnumobj=ntable+ntable*(nchair_max-1) # 14
gen_no_table = False

# porportion based on utils's 3d chair and roundtable models
table_width, table_height = 0.2, 0.2  # in [-1,1], 1/4 of original # radius
chair_width, chair_height = 0.1112, 0.1310 # 1/5 of original

offset_from_table = 0.02


## DATA GENERATION: utils
def reset_padding(numchairs, toreset, toreset2):
    """ Needed as the number of chairs is variable.
        toreset(2): [batch_size, maxnumobj, 2]
        numchairs : [batch_size, 2]
    """
    for scene_idx in range(toreset.shape[0]):
        scene_numobj = 2+numchairs[scene_idx,0]+numchairs[scene_idx,1] 
        toreset[scene_idx, scene_numobj:,:]=0
        toreset2[scene_idx, scene_numobj:,:]=0
    return toreset, toreset2


def _gen_tablechair_circle_batch(batch_size):
    """ Generate scene of 2 tables, each with 2-6 chairs in circular formation.

        Variable data length is delt with through padding with 0s at the end.

        batch_pos: position has size [batch_size, maxnumobj, 2]=[x, y], where [:,0:2,:] are the 2 tables,
                   and the rest are chairs.
        batch_ang: orientation has size [batch_size, maxnumobj, 2]=[cos(th), sin(th)]
        batch_siz: [batch_size, 14, 2] = [width, height]
        batch_cla: [batch_size, maxnumobj, 2], represents type of object with one hot encoding,
                   specifically [1,0] refers to table and [0,1] refers to chair
        batch_numchairs: [batch_size, 2], contains numbers of chairs for left table and for right table
                            
    """
    batch_pos = np.zeros((batch_size, maxnumobj, 2))
    batch_ang = np.zeros((batch_size, maxnumobj, 2))
    batch_siz = np.zeros((batch_size, maxnumobj, 2))
    batch_cla = np.zeros((batch_size, maxnumobj, 2))

    # table
    ## position coordinates
    left_table_x = np.random.uniform(-1+table_width+offset_from_table, -0.1-table_width-offset_from_table, size=(batch_size,1,1))
    left_table_y = np.random.uniform(-1+table_height+offset_from_table, 1-table_height-offset_from_table,  size=(batch_size,1,1)) # also need space for chairs
    left_table = np.concatenate([left_table_x, left_table_y], axis=2) # batch_size, 1, 2
    batch_pos[:,0:1,:] = left_table # in [-1, 1]
  
    right_table_x = np.random.uniform(0.1+table_width+offset_from_table, 1-table_width-offset_from_table, size=(batch_size,1,1))
    right_table_y = np.random.uniform(-1+table_height+offset_from_table, 1-table_height-offset_from_table, size=(batch_size,1,1))  # also need space for chairs
    right_table = np.concatenate([right_table_x, right_table_y], axis=2) # batch_size, 1, 2
    batch_pos[:,1:2,:] = right_table # in [-1, 1]

    ## orientation
    batch_ang[:,0:ntable,0:1] = np.cos(0)
    batch_ang[:,0:ntable,1:2] = np.sin(0)
    ## size
    batch_siz[:,0:2,:] = np.array([table_width, table_height]) # in [-1, 1]
    ## class
    batch_cla[:,0:ntable,:] = np.array([1,0])
    

    # chairs
    batch_numchairs = np.random.randint(nchair_min, nchair_max, size=(batch_size, ntable)) # 2-6 chairs per table
    for batch_i in range(batch_size):
        starting_obj_i = ntable
        for table_i in range(ntable):
            num_c = batch_numchairs[batch_i, table_i]
            angles = np.linspace(0,2*np.pi,num=num_c+1)[:-1] + np.random.uniform(0, 2*np.pi/(num_c)) # [num_c]
            angles = np.expand_dims(angles, axis=1) # [num_c, 1]
            pos, _ = angles_to_circle_scene(angles, batch_pos[batch_i,table_i,:], table_width+offset_from_table) # angles, center, radius
            batch_pos[batch_i, starting_obj_i:starting_obj_i+num_c, :] = pos
             # NOTE: angle recalcualated because subtract 90 degrees to be coherent with 3d front, pos y axis is 0 degree
            batch_ang[batch_i, starting_obj_i:starting_obj_i+num_c, :] = np.around(np.concatenate([-np.cos(angles-np.pi/2) , -np.sin(angles-np.pi/2)], axis=1), decimals=3)
            starting_obj_i+= num_c
        batch_siz[batch_i,ntable:starting_obj_i, :] = np.array([chair_width, chair_height])  # in [-1, 1]
        batch_cla[batch_i,ntable:starting_obj_i, :] = np.array([0,1])
        # NOTE: there may be left-over rows in pos, ang, sha for each scene - left as 0

    batch_sha = np.concatenate([batch_siz, batch_cla], axis=2)  # [batch_size, maxnobj, 2+2=4]
    return batch_pos, batch_ang, batch_sha, batch_numchairs


## DATA GENERATION: public API
## 3 variations: bimodal (preferred), allnoisy, halfsplit
def gen_data_tablechair_circle_bimodal(half_batch_size, noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4,
                                       clean_noise_level_stddev=0.01, clean_angle_noise_level_stddev=np.pi/90, no_table=None):
    """ Preferred way of data generation. 

        Generates scene of 2 'round' tables, each with a variable number of chairs in circular formation at a fixed radius.
        The two tables are independently generated, each in half of the domain, and the chairs are generated subsequently with
        respect to the table. The chairs are equi-angle in the circle, at any starting rotation offset with respect to the table.

        half_batch_size: number of clean scenes; number of noisy scenes 
        noise_level_min, noise_level_max: for adding position noise
        angle_range: for adding angle noise
        no_table: defaults to gen_no_table, overwritten if provided
        
        Intermediary: {clean, noisy}_{pos, ang, sha} have shape [batch_size, maxnumobj, 2]

        Returns: input: [batch_size, maxnumobj, 8] = [x, y, cos(th), sin(th), sizx, sizy, 1(0), 0(1)]
                 label: [batch_size, maxnumobj, 4] = [x, y, cos(th), sin(th)]
                 
        Clean input's labels are identical to the input. Noisy input's labels are ordered assignment to clean position for each 
        object (through earthmover distance assignment), using the original clean sample (noisy_orig_pos) as target, and angle labels
        are the corresponding angles from the assignment.
       
        Note that this implementation takes a non-hierarhical, non-scene-graph, free-for-all approach: emd assignment is done freely 
        among all objects of a class. 

        bimodal-specific:
        input:  {clean scene + small gaussian guassian nosie, clean scene + large gaussian gaussian nosie}.
        labels: {clean sene without noise, clean scene without noise + earthmover distance assignemnt}

    """
    clean_orig_pos, clean_orig_ang, clean_orig_sha, clean_orig_numchairs = _gen_tablechair_circle_batch(half_batch_size)
    clean_pos = np_add_gaussian_gaussian_noise(clean_orig_pos, noise_level_stddev=clean_noise_level_stddev)
    clean_ang = np_add_gaussian_gaussian_angle_noise(clean_orig_ang, noise_level_stddev=clean_angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
    clean_pos, clean_ang = reset_padding(clean_orig_numchairs, clean_pos, clean_ang) # [batch_size, maxnumobj 2]

    noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_numchairs = _gen_tablechair_circle_batch(half_batch_size)
    noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)# [batch_size, 2+12, 2] # NOTE: Changed noise distribution to be the same for all objs in a scene
    noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
    noisy_pos, noisy_ang = reset_padding(noisy_orig_numchairs, noisy_pos, noisy_ang) # [batch_size, maxnumobj 2]

    clean_input = np.concatenate([clean_pos, clean_ang, clean_orig_sha], axis=2) # [half_batch_size, maxnumobj, 6]
    noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, maxnumobj, 6]
    input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, maxnumobj, 6]
    
    clean_labels =  np.concatenate([clean_orig_pos, clean_orig_ang], axis=2)  # disregard shape NOTE: perturbation small enough to do direct correspondence
    noisy_labels = np.zeros((half_batch_size, maxnumobj, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:ntable, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 0:ntable, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 0:ntable, :]) # 2x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 0:ntable, 0:2] = np.array(table_assignment)
        noisy_labels[scene_idx, 0:ntable, 2:4] = np.array(table_assign_ang)

        # earthmover distance assignment for chairs
        scene_numobj = ntable+noisy_orig_numchairs[scene_idx,0]+noisy_orig_numchairs[scene_idx,1]
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, ntable:scene_numobj, :]] # array of tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, ntable:scene_numobj, :]] # array of tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, ntable:scene_numobj, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, ntable:scene_numobj, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, ntable:scene_numobj, 2:4] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_idx, scene_numobj:, :] left as 0
    
    labels = np.concatenate([clean_labels, noisy_labels], axis=0)

    if no_table is None: no_table = gen_no_table
    if no_table:
        input = input[:,ntable:,:]
        labels = input[:,ntable:,:]

    return input, labels #, firstscene_nopad


def gen_data_tablechair_circle_allnoisy(half_batch_size, noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4, no_table=None):
    """  allnoisy-specific:
        In generating the noisy positions, chairs are given greater noise while tables are  given less. This is to teach the network 
        to move certain types less.

        input: {clean scene + gaussian gaussian noise}
        label: {corresponding clean scenes})
    """
    clean_pos, clean_ang, clean_sha, clean_numchairs = _gen_tablechair_circle_batch(half_batch_size*2) # NOTE: no noise for clean input and label. Note the *2
    noisy_pos_tables = np_add_gaussian_gaussian_noise(clean_pos[:,0:ntable,:], noise_level_stddev=noise_level_stddev/2)
    noisy_pos_chairs = np_add_gaussian_gaussian_noise(clean_pos[:,ntable:maxnumobj,:], noise_level_stddev=noise_level_stddev)
    noisy_pos = np.concatenate([noisy_pos_tables, noisy_pos_chairs], axis=1) # [batch_size, 2+12, 2]
    noisy_ang = np_add_gaussian_gaussian_angle_noise(clean_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
    noisy_pos, noisy_ang = reset_padding(clean_numchairs, noisy_pos, noisy_ang) # [batch_size, maxnumobj 2]

    input = np.concatenate([noisy_pos, noisy_ang, clean_sha], axis=2) # [batch_size, maxnumobj, 6]

    labels = np.zeros((half_batch_size*2, maxnumobj, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:ntable, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in clean_pos[scene_idx, 0:ntable, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_idx, 0:ntable, :]) # 2x2: assigned position for each pt in p1 (in that order)
        labels[scene_idx, 0:ntable, 0:2] = np.array(table_assignment)
        labels[scene_idx, 0:ntable, 2:4] = np.array(table_assign_ang)

        # earthmover distance assignment for chairs
        scene_numobj = ntable+clean_numchairs[scene_idx,0]+clean_numchairs[scene_idx,1]
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, ntable:scene_numobj, :]] # array of nchair tuples
        p2 = [tuple(pt) for pt in clean_pos[scene_idx, ntable:scene_numobj, :]] # array of nchair tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_idx, ntable:scene_numobj, :]) # 12x2: assigned position for each pt in p1 (in that order)
        labels[scene_idx, ntable:scene_numobj, 0:2] = np.array(chair_assignment)
        labels[scene_idx, ntable:scene_numobj, 2:4] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_idx, scene_numobj:, :] left as 0

    if no_table is None: no_table = gen_no_table
    if no_table:
        input = input[:,ntable:,:]
        labels = labels[:,ntable:,:]

    return input, labels


def gen_data_tablechair_circle_halfsplit(half_batch_size, noise_level_stddev=0.25, angle_noise_level_stddev=np.pi/4, no_table=None):
    """ halfsplit-specific:
        Clean pos and ang in input has no noise. 
        Different noise levels for tables and chairs.
    """
    clean_pos, clean_ang, clean_sha, clean_numchairs = _gen_tablechair_circle_batch(half_batch_size) # NOTE: no noise for clean input and label
    noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_numchairs = _gen_tablechair_circle_batch(half_batch_size)

    noisy_pos_tables = np_add_gaussian_gaussian_noise(noisy_orig_pos[:,0:ntable,:],noise_level_stddev=noise_level_stddev/2)
    noisy_pos_chairs = np_add_gaussian_gaussian_noise(noisy_orig_pos[:,ntable:maxnumobj,:], noise_level_stddev=noise_level_stddev)
    noisy_pos = np.concatenate([noisy_pos_tables, noisy_pos_chairs], axis=1) # [batch_size, 2+12, 2]
    noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
    # noisy_ang = np_add_angle_noise(noisy_orig_ang, low=-np.pi/2, high=np.pi/2)
    noisy_pos, noisy_ang = reset_padding(noisy_orig_numchairs, noisy_pos, noisy_ang) # [batch_size, maxnumobj 2]

    clean_input = np.concatenate([clean_pos, clean_ang, clean_sha], axis=2) # [half_batch_size, maxnumobj, 6]
    noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, maxnumobj, 6]
    input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, maxnumobj, 6]
    

    clean_labels = np.copy(clean_input[:,:,0:4]) # disregard shape
    noisy_labels = np.zeros((half_batch_size, maxnumobj, 4))
    for scene_idx in range(half_batch_size):
        # earthmover distance assignment for tables
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, 0:ntable, :]] # array of 2 tuples: [tuple(x, y), tuple(x, y)]
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, 0:ntable, :]] # array of 2 tuples
        table_assignment, table_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, 0:ntable, :]) # 2x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, 0:ntable, 0:2] = np.array(table_assignment)
        noisy_labels[scene_idx, 0:ntable, 2:4] = np.array(table_assign_ang)

        # earthmover distance assignment for chairs
        scene_numobj = ntable+noisy_orig_numchairs[scene_idx,0]+noisy_orig_numchairs[scene_idx,1]
        p1 = [tuple(pt) for pt in noisy_pos[scene_idx, ntable:scene_numobj, :]] # array of tuples
        p2 = [tuple(pt) for pt in noisy_orig_pos[scene_idx, ntable:scene_numobj, :]] # array of tuples
        chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, noisy_orig_ang[scene_idx, ntable:scene_numobj, :]) # 12x2: assigned position for each pt in p1 (in that order)
        noisy_labels[scene_idx, ntable:scene_numobj, 0:2] = np.array(chair_assignment)
        noisy_labels[scene_idx, ntable:scene_numobj, 2:4] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_idx, scene_numobj:, :] left as 0
    
    labels = np.concatenate([clean_labels, noisy_labels], axis=0)

    if no_table is None: no_table = gen_no_table
    if no_table:
        input = input[:,ntable:,:]
        labels = input[:,ntable:,:]

    return input, labels #, firstscene_nopad






## VISUALIZATION

def visualize_tablechair_circle(scene, fp="tablechair_horizontal.jpg", title="table_horizontal"):
    """ scene: shape [nobj, pos_d+ang_d+siz_d+cla_d], where first 2 rows are tables.
        
        Difference from visualize functinos in utils.py: more coherent with the visualization function for
        3d front, mainly has 90-degree rotation for angles
    """
    numobj=find_numobj(scene[:,6:8]) # initial=generated input ("tablechair_circle")

    siz = scene[:, 4:6]  # bbox lengths
    arrows = np_rotate(scene[:,2:4], np.zeros((scene.shape[0],1))+np.pi/2) # rot about origin as these are directional vectors
            # NOTE: add 90 deg because of offset between 0 degree angle visualization (1,0) and 0 degree obj, which starts off facing pos y/(0, 1)
            
    fig = plt.figure(dpi=300, figsize=(5,5))
    # initial & final
    table = plt.scatter(x=scene[:ntable,0], y=scene[:ntable,1],  c="#2ca02c", label="table") # green
    chair = plt.scatter(x=scene[ntable:numobj,0], y=scene[ntable:numobj,1], c="#1f77b4", label=f"chair") # blue
    # orientation
    plt.quiver(scene[:2,0], scene[:2,1], arrows[:ntable,0], arrows[:ntable,1], color="#2ca02c", label="table", width=0.005)
    plt.quiver(scene[ntable:numobj,0], scene[ntable:numobj,1], arrows[ntable:numobj,0], arrows[ntable:numobj,1], color="#1f77b4", label="chair", width=0.005)
   
    for o_i in range(ntable):
        plt.gca().add_patch(plt.Circle((scene[o_i,0], scene[o_i,1]), radius=scene[o_i,4], fill = False, alpha=0.3, edgecolor="#2ca02c" ))
    for o_i in range(ntable, numobj):
        plt.gca().add_patch(plt.Rectangle(
                                (scene[o_i,0]-siz[o_i,0]/2, scene[o_i,1]-siz[o_i,1]/2), 
                                siz[o_i,0], siz[o_i,1], linewidth=1, edgecolor="#1f77b4", facecolor='none',
                                transform = mt.Affine2D().rotate_around(scene[o_i,0], scene[o_i,1], np.arctan2(scene[o_i,3], scene[o_i,2])) +plt.gca().transData 
                            ))

    plt.legend(handles=[table, chair], fontsize=7)
    plt.gca().set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2])
    plt.title(f"{title}", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(fp)
    plt.close(fig)




if __name__ == '__main__':
    input, labels = gen_data_tablechair_circle_bimodal(2) # [batch_size, 14, 8]  
    for scene_i in range(input.shape[0]):
        visualize_tablechair_circle(input[scene_i], fp=f"tablechair_circle_{scene_i}.jpg", title=f"tablechair_circle: {scene_i}")

