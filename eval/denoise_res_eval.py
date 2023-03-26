import os, sys
sys.path.insert(1, os.getcwd())

import argparse
import numpy as np

from data.utils import *
from data.distance import *
from data.TDFront import TDFDataset

from data.tablechair_horizontal import table_to_chair
from data.tablechair_circle import table_width, offset_from_table
position_radius = table_width+offset_from_table

from filepath import eval_dir



## 3DFRONT
def dist2gt_from_npz(data_type, fp, use_emd=True):
    tdf = TDFDataset(data_type, use_augment=True, print_info=False)
    print(f"\ndist2gt_from_npz:\n{fp}")
    data =np.load(fp, allow_pickle=True)
    for method in ["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"]:
        res = dist_2_gt(data[f"{method}_trajs"], data["scenepaths"], tdf, use_emd=use_emd)
        print(f"{method}: dist_2_gt={res}")


def dist_2_gt(trajs, scenepaths, tdf, use_emd=True):
    """ trajs: [nscene, niter, nobj, pos+ang+siz+cla], in [-1,1]
        returns: how far an obj is from (emd) ground truth, in [-1,1], averaged across scenes
    """
    P, A, S = tdf.pos_dim, tdf.ang_dim, tdf.siz_dim

    scenes_mean_perobjdists = []
    for scene_i in range(trajs.shape[0]):
        final = trajs[scene_i][-1] # (maxnobj, pos+ang+siz+cla)
        nobj, cla_idx = TDFDataset.parse_cla(final[:,P+A+S:])
        groundtruth, _ = tdf.read_one_scene(os.path.split(scenepaths[scene_i])[1], normalize=True) # [nobj (not maxnobj), pos+ang+siz+cla], c1d1893e-86a3-4a1f-a0bb-1b5104461b5a_MasterBedroom-21047_3

        if use_emd:
            gt_assignment = tdf.emd_by_class(np.expand_dims(final[:nobj, :P], axis=0), np.expand_dims(groundtruth[:nobj, :P],axis=0),
                                             np.expand_dims(groundtruth[:nobj, P:P+A],axis=0), np.expand_dims(groundtruth[:nobj, P+A:], axis=0), np.array([nobj]))[0] # [1->None, maxnobj, P+A]
        else:
            gt_assignment = groundtruth # for non-emd
        
        final_pos = final[:nobj, :tdf.pos_dim] # [nobj, P]
        gt_pos = gt_assignment[:nobj, :tdf.pos_dim] #  [nobj, P]
        per_obj_dists = np.linalg.norm(final_pos-gt_pos, axis=1) #[nobj,]
        scene_mean_perobjdist = np.mean(per_obj_dists) # for this scene, on avg, how far is each obj from ground truth
        scenes_mean_perobjdists.append(scene_mean_perobjdist)

    # NOTE: each scene is weighed the same. For rooms with more num of furnitures, many of them share 1 slice of weight -> have less impact
    return np.mean(scenes_mean_perobjdists)/(tdf.room_size[0]/2) # x and y/z have same length


def tdfront_success(data_type, fp, thresholds=None, within_floorplan=True, no_penetration=False):
    """ Checks the porportion of objects that intersect with the floor plan boundaries in final predicted result.
        For comparing two methods of floor plan encoding (resnet vs pointnet).
        
        fp: leads to a npz file saved from denoise_meta() in train.py. Example: 'pos0.1_ang15_train50000.npz'.
    """
    if thresholds is None:
        thresholds = {
            "pen_siz_scale": 0.92,
            "validobj_porportion": 0.9
        }

    tdf = TDFDataset(data_type, use_augment=True, print_info=False)
    P, A, S = tdf.pos_dim, tdf.ang_dim, tdf.siz_dim

    print(f"\ntdfront_success:\n{fp}")
    print("thresholds: ", thresholds)
    data = np.load(fp, allow_pickle=True)
    numscene = data['direct_map_once_trajs'].shape[0] # 500
    for method in ["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"]:
        trajs = data[f"{method}_trajs"] # (nscene, niter, maxnobj, pos+ang+siz+cla)
        success_scene_ct = 0
        for scene_i in range(numscene):
            final = trajs[scene_i][-1] # [maxnobj, pos+ang+siz+cla]
            final[:,:P] /= (tdf.room_size[0]/2)
            final[:, P+A:P+A+S] /= (tdf.room_size[0]/2)

            nobj, cla_idx = TDFDataset.parse_cla(final[:,P+A+S:]) 
            scene_data_path = os.path.join(tdf.scene_dir, os.path.split(data['scenepaths'][scene_i])[-1], "boxes.npz")
            scene_data = np.load(scene_data_path)
            scene_fpoc = scene_data["floor_plan_ordered_corners"] # [nfpc, pos_dim=2]

            success_o_ct = 0
            for o_i in range(nobj):
                if tdf.is_valid(o_i, final[o_i:o_i+1,:P], final[o_i:o_i+1,P:P+A], final[:nobj, :P], final[:nobj, P:P+A], final[:nobj, P+A:],
                                scene_fpoc, within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=thresholds['pen_siz_scale']):
                    success_o_ct+=1
            # print(f"scene {scene_i}: {success_o_ct}/{nobj} = {round(success_o_ct/nobj, 4)}")
            if (success_o_ct/nobj) > thresholds['validobj_porportion']: success_scene_ct +=1
        
        
        print(f"{method}: {success_scene_ct}/{numscene} = {round(success_scene_ct/numscene, 4)}")





## TABLECHAIR

def tablechair_horizontal_stats(scene, perobj_distmoved):
    """ Adapted from tablechair_horizontal_success, used to report numerical results for further analysis."""
    left_table, right_table = scene[0], scene[1]
    chairs_dist2lefttable = [ np.linalg.norm(scene[i, :2]-left_table[:2]) for i in range(2, 14)]
    idx = np.argsort(chairs_dist2lefttable) # ex: array([3, 1, 2, 0]) for [40, 20, 30, 10]
    left_chair_idx = np.sort(idx[:6]) # [6,] sorted idx of the 6 chairs to the left, 
    right_chair_idx = np.delete(np.arange(12), left_chair_idx) # [6,], sorted
    
    left_chairs = scene[left_chair_idx+2] # [6, pos+ang+siz+cla], +2 because first 2 are table
    right_chairs = scene[right_chair_idx+2]

    # 2. realang: penalty for when where chairs are not facing the table (in top rows).
    down, up = np.array([np.cos(np.pi), np.sin(np.pi)]), np.array([np.cos(0), np.sin(0)])
    dir_offsets = []
    for left_chair in left_chairs:
        if left_chair[1] >= left_table[1]: # top row
            offset_from_table_facing_dir = np_single_angle_between(left_chair[2:4], down) # [0, pi] 
        else:
            offset_from_table_facing_dir = np_single_angle_between(left_chair[2:4], up)
        dir_offsets.append(offset_from_table_facing_dir)

    for right_chair in right_chairs:
        if right_chair[1] >= right_table[1]: # top row
            offset_from_table_facing_dir = np_single_angle_between(right_chair[2:4], down)
        else:
            offset_from_table_facing_dir = np_single_angle_between(right_chair[2:4], up)
        dir_offsets.append(offset_from_table_facing_dir)
        
    # 3. tablechair_relpos: EMD between positions of the chairs based on final table prediction & prediction
    leftT_chairposgt, _ = table_to_chair(left_table[:2])
    p1 = [tuple(chair[:2]) for chair in left_chairs] # array of 6 tuples
    p2 = [tuple(chair) for chair in leftT_chairposgt]
    _, _, _, relative_emd_left = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    
    rightT_chairposgt, _ = table_to_chair(right_table[:2]) #[6, 2]
    p1 = [tuple(chair[:2]) for chair in right_chairs]
    p2 = [tuple(chair) for chair in rightT_chairposgt]
    _, _, _, relative_emd_right = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    
    dir_offsets = np.array(dir_offsets) # averaged across chairs -> each object's angle between correct one
    return perobj_distmoved, np.mean(dir_offsets), relative_emd_left+relative_emd_right


def tablechair_horizontal_success(scene, perobj_distmoved, thresholds=None):
    """ Returns True if scene passes validity check, False otherwise.
        scene:     [nobj, pos+ang+siz+cla=8]
        dist_moved: scalar
    """
    if thresholds is None:
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/30,
            'relative_emd': 0.1
        }

    # 1. distance moved
    if perobj_distmoved > thresholds["perobj_distmoved"]: return False # noise level distribution's stddev was 0.25

    left_table, right_table = scene[0], scene[1]
    chairs_dist2lefttable = [ np.linalg.norm(scene[i, :2]-left_table[:2]) for i in range(2, 14)]
    idx = np.argsort(chairs_dist2lefttable) # ex: array([3, 1, 2, 0]) for  [40, 20, 30, 10]
    left_chair_idx = np.sort(idx[:6]) # [6,] sorted idx of the 6 chairs to the left, 
    right_chair_idx = np.delete(np.arange(12), left_chair_idx) # [6,], sorted
    
    left_chairs = scene[left_chair_idx+2] # [6, pos+ang+siz+cla], +2 because first 2 are table
    right_chairs = scene[right_chair_idx+2]

    # 2. realang: penalty for when where chairs are not facing the table (in top rows).
    down, up = np.array([np.cos(np.pi), np.sin(np.pi)]), np.array([np.cos(0), np.sin(0)])
    for left_chair in left_chairs:
        if left_chair[1] >= left_table[1]: # top row
            offset_from_table_facing_dir = np_single_angle_between(left_chair[2:4], down) # [0, pi] 
        else:
            offset_from_table_facing_dir = np_single_angle_between(left_chair[2:4], up)
        if offset_from_table_facing_dir > thresholds['offset_from_table_facing_dir']: return False

    for right_chair in right_chairs:
        if right_chair[1] >= right_table[1]: # top row
            offset_from_table_facing_dir = np_single_angle_between(right_chair[2:4], down)
        else:
            offset_from_table_facing_dir = np_single_angle_between(right_chair[2:4], up)
        if offset_from_table_facing_dir > thresholds['offset_from_table_facing_dir']: return False # [0, pi] 

    # 3. tablechair_relpos: EMD between positions of the chairs based on final table prediction & prediction
    leftT_chairposgt, _ = table_to_chair(left_table[:2])
    p1 = [tuple(chair[:2]) for chair in left_chairs] # array of 6 tuples
    p2 = [tuple(chair) for chair in leftT_chairposgt]
    _, _, _, relative_emd_left = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    
    rightT_chairposgt, _ = table_to_chair(right_table[:2]) #[6, 2]
    p1 = [tuple(chair[:2]) for chair in right_chairs]
    p2 = [tuple(chair) for chair in rightT_chairposgt]
    _, _, _, relative_emd_right = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    
    if relative_emd_left+relative_emd_right > thresholds['relative_emd']: return False # in scla [-1,1]

    return True


def tablechair_circle_success(scene, perobj_distmoved, thresholds=None):
    """ Returns True if scene passes validity check, False otherwise.
        scene:     [nobj, pos+ang+siz+cla=8]
        dist_moved: scalar
    """
    if thresholds is None:
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/30,
            "uniform_ang": 0.009, # around 0.5 degree
            "radial_pos": 0.02
        }
    # 1. distance moved
    if perobj_distmoved > thresholds["perobj_distmoved"]: return False # noise level distribution's stddev was 0.25
    
    nobj = find_numobj(scene[:,6:8])
    scene = scene[:nobj]

    left_table, right_table = scene[0], scene[1]
    left_chair_idx, right_chair_idx = [], []
    for i in range(2,nobj): # chairs
        if np.linalg.norm(scene[i, :2]-left_table[:2]) <= np.linalg.norm(scene[i, :2]-right_table[:2]):
            left_chair_idx.append(i)
        else:
            right_chair_idx.append(i)
    left_chairs = scene[left_chair_idx] # [n, pos+ang+siz+cla]
    right_chairs = scene[right_chair_idx]

    # 2. realang: penalty for when where chairs are not facing the table (in top rows).
    comparison_ang = np_rotate(left_chairs[:,2:4], np.zeros((left_chairs.shape[0], 1))+np.pi/2) # [n,2], 0 faces pos y
    for i in range(left_chairs.shape[0]):
        left_chair = left_chairs[i]
        offset_from_table_facing_dir = np_single_angle_between(comparison_ang[i], left_table[:2]-left_chair[:2]) # normalize in function, [0, pi] 
        if offset_from_table_facing_dir > thresholds['offset_from_table_facing_dir']: return False
    
    comparison_ang = np_rotate(right_chairs[:,2:4], np.zeros((right_chairs.shape[0], 1))+np.pi/2) # [n,2], 0 faces pos y
    for i in range(right_chairs.shape[0]):
        right_chair = right_chairs[i]
        offset_from_table_facing_dir = np_single_angle_between(comparison_ang[i], right_table[:2]-right_chair[:2])
        if offset_from_table_facing_dir > thresholds['offset_from_table_facing_dir']: return False # [0, pi] 


    # 3. uniform_ang: variance of the angular distances between each pair of adjacent chairs.
    left_chairs_ang = np.sort(trig2ang(left_chairs[:,2:4]).reshape(-1)) # -1.85613356 -0.27810768  1.28576114  2.85007502]
    left_chairs_ang = np.append(left_chairs_ang, left_chairs_ang[0]+2*np.pi)
    var = np.var( np.ediff1d(left_chairs_ang) )
    if var > thresholds['uniform_ang']: return False
    right_chairs_ang = np.sort(trig2ang(right_chairs[:,2:4]).reshape(-1))
    right_chairs_ang = np.append(right_chairs_ang, right_chairs_ang[0]+2*np.pi)
    var = np.var( np.ediff1d(right_chairs_ang) )
    if var > thresholds['uniform_ang']: return False

    # 4. radial_pos: checks if mean per-chair distance to the closest table is far from the pre-designated radius employed in clean scene synthesis.
    diff = abs(np.mean(np.linalg.norm(left_chairs[:,:2]-left_table[:2], axis=1)) - position_radius)  # [nleftchair, ]
    if diff > thresholds['radial_pos']: return False 
    diff = abs(np.mean(np.linalg.norm(right_chairs[:,:2]-right_table[:2], axis=1)) - position_radius) 
    if diff > thresholds['radial_pos']: return False

    return True


def tablechair_shape_success(scene, perobj_distmoved, thresholds=None):
    """ Returns True if scene passes validity check, False otherwise.
        scene:     [nobj, pos+ang+siz+cla=8]
        dist_moved: scalar
    """
    if thresholds is None:
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/30,
            'relative_emd': 0.1
        }

    # 1. distance moved
    if perobj_distmoved > thresholds["perobj_distmoved"]: return False # noise level distribution's stddev was 0.25

    table = scene[0,] # pos+ang+siz+cla+sha=8+128=136
    chairs = scene[1:] # 6, pos+ang+siz+cla+sha=8+128=136

    # 2. realang: penalty for when where chairs are not facing the table (in top rows).
    down, up = np.array([np.cos(np.pi), np.sin(np.pi)]), np.array([np.cos(0), np.sin(0)])
    top_row_idx, bottom_row_idx = [], []
    for i in range(chairs.shape[0]):
        chair = chairs[i]
        if chair[1] >= table[1]: # top row
            top_row_idx.append(i)
            offset_from_table_facing_dir = np_single_angle_between(chair[2:4], down) # [0, pi] 
        else:
            bottom_row_idx.append(i)
            offset_from_table_facing_dir = np_single_angle_between(chair[2:4], up)
        if offset_from_table_facing_dir > thresholds['offset_from_table_facing_dir']: return False
    if len(top_row_idx) != len(bottom_row_idx): return False # not 3 and 3


    # 3. tablechair_relpos: EMD between positions of the chairs based on final table prediction & prediction
    chairposgt, _ = table_to_chair(table[:2])
    p1 = [tuple(chair[:2]) for chair in chairs] # array of 6 tuples
    p2 = [tuple(chair) for chair in chairposgt]
    _, _, _, relative_emd = earthmover_distance(p1, p2) # sum of cost(euclidean dist) of each chair's assignment 
    if relative_emd > thresholds['relative_emd']: return False # in scla [-1,1]

    # 4. group_by_sha: binary indicator of if a row only contains chairs of one shape.
    sha = chairs[top_row_idx[0]][8:] # [128,]
    for chair_i in top_row_idx[1:]:
        if not np.array_equal(sha, chairs[chair_i][8:]): return False

    return True


def tablechair_success(data_type, fp):
    """ fp: fp: leads to a npz file saved from denoise_meta() in train.py. Example: 'pos0.1_ang15_train50000.npz'. """
    print(f"\ntablechair_success:\n{fp}")
    data =np.load(fp, allow_pickle=True)

    if data_type == "tablechair_horizontal":
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/60,
            'relative_emd': 0.25
        }
    elif data_type == "tablechair_circle":
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/60,
            "uniform_ang": 0.009, # around 0.5 degree
            "radial_pos": 0.01
        }
    elif data_type == "tablechair_shape":
        thresholds = {
            "perobj_distmoved": 0.5,
            "offset_from_table_facing_dir": np.pi/60,
            'relative_emd': 0.05,
        }
    print("thresholds: ", thresholds)

    numscene = data['direct_map_once_trajs'].shape[0] # 500
    for method in ["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"]:
        trajs = data[f"{method}_trajs"] # (nscene, niter, maxnobj, pos+ang+siz+cla)
        perobj_distmoveds = data[f"{method}_perobj_distmoveds"] # [500,]
        success_ct = 0
        # objdistmoveds, dir_offsets, emds=[], [], []
        for scene_i in range(numscene):
            if data_type == "tablechair_horizontal":
                if tablechair_horizontal_success(trajs[scene_i][-1], perobj_distmoveds[scene_i], thresholds=thresholds): success_ct +=1
                # objdistmoved, dir_offset, emd = tablechair_horizontal_stats(trajs[scene_i][-1], perobj_distmoveds[scene_i])
                # objdistmoveds.append(objdistmoved)
                # dir_offsets.append(dir_offset)
                # emds.append(emd)
            elif data_type == "tablechair_circle":
                if tablechair_circle_success(trajs[scene_i][-1], perobj_distmoveds[scene_i], thresholds=thresholds): success_ct +=1
            elif data_type == "tablechair_shape":
                if tablechair_shape_success(trajs[scene_i][-1], perobj_distmoveds[scene_i], thresholds=thresholds): success_ct +=1
        
        # objdistmoveds, dir_offsets, emds = np.array(objdistmoveds), np.array(dir_offsets), np.array(emds)
        # print(f"{method}: objdistmoveds mean = {np.mean(objdistmoveds)}, variance = {np.var(objdistmoveds)}")
        # print(f"\t dir_offsets mean = {np.mean(dir_offsets)}, variance = {np.var(dir_offsets)}")
        # print(f"\t emds mean = {np.mean(emds)}, variance = {np.var(emds)}")

        print(f"{method}: {success_ct}/{numscene} = {round(success_ct/numscene, 4)}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_type", type = str, default="3dfront",
                        choices=["tablechair_horizontal", "tablechair_circle", "tablechair_shape", "3dfront"],
                        help = '''
                                -tablechair_horizontal: 2 tables with 6 chairs each in rectangular formation (fixed relative distance).
                                -tablechair_circle: 2 tables with 2-6 chairs each in circular formation.
                                -tablechair_shape: 1 table with 6 chairs of 2 types, one type on each side.
                                -3dfront: 3D-FRONT dataset (professionally designed indoor scenes); currently support bedroom and livingroom.
                                ''')
    parser.add_argument("--res_filepath", type = str, help="filepath of npz file saved from denoise_meta() in train.py.")
    # ['random_idx', 'scenepaths', 'noise_level', 'direct_map_once_trajs', 'direct_map_once_perobj_distmoveds', 'direct_map_trajs', 'direct_map_perobj_distmoveds',
    #  'grad_nonoise_trajs', 'grad_nonoise_perobj_distmoveds', 'grad_noise_trajs', 'grad_noise_perobj_distmoveds']
    parser.add_argument("--room_type", type = str, default="livingroom", choices = ["bedroom", "livingroom"], help="3D-FRONT specific." )
    args = vars(parser.parse_args()) #'dict' 



    if "tablechair" in args['data_type']:
        tablechair_success(args['data_type'], args['res_filepath'])
    else: # 3dfront
        dist2gt_from_npz(args['room_type'], args['res_filepath'], use_emd=True)
        tdfront_success( args['room_type'], args['res_filepath'], thresholds=None, within_floorplan=True, no_penetration=False)

