import os, sys
sys.path.insert(1, os.getcwd()) # canonical-arrangement

import json
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import cv2 as cv

from data.utils import *
from data.distance import *

from filepath import *


TDF_room_types = [  "bedroom",      # 4041
                    "livingroom",   # 813
                    "diningroom",   # 900
                    "library"]      # 285

room_info = {
    "nobj": { # inclusive, from preprocessing the dataset
        "bedroom": [3, 12],
        "diningroom": [3, 21],
        "library": [3, 11],
        "livingroom": [3, 21]
    },
    "room_size": { # y axis: height
        "bedroom": [6,4,6],
        "diningroom": [6,4,6],
        "library": [12,4,12],
        "livingroom": [12,4,12],
    },
    "maxnfpoc":{ # maximum number of floor plan ordered corners
        "bedroom": [4,25],
        "livingroom": [1,51] # 39 rooms above 35
    }
}


class TDFDataset:
    '''3D-Front Dataset'''

    def __init__(self, room_type, use_augment=True, livingroom_only=False, print_info=True):
        """ Currently all scenes_tv (test+validation) are used in training, and validation uses test.
            bedroom:                       len(self.scenes_tv)=5668,  len(self.scenes_test)=224
                augmened:                  len(self.scenes_tv)=22672, len(self.scenes_test)=896
            livingroom:                    len(self.scenes_tv)=660,   len(self.scenes_test)=163 
                augmented:                 len(self.scenes_tv)=2640,  len(self.scenes_test)=652
            livingroom + livingdiningroom: len(self.scenes_tv)=2338,  len(self.scenes_test)=587
                augmented:                 len(self.scenes_tv)=9352,  len(self.scenes_test)=2348
        """
        self.room_type = room_type if room_type in TDF_room_types else "bedroom"
        self.livingroom_only = (room_type=="livingroom") and livingroom_only 

        # train-test-val splits (70-20-10)
        split2room = {} # { "test": ['MasterBedroom-147840', 'MasterBedroom-49298', ...] }
        with open(os.path.join(data_dir, f"{self.room_type}_threed_front_splits.csv"), "r") as csv_file:
            data = [row for row in csv.reader(csv_file, delimiter=',')]  # 6286 in list, 5668tv/224test for bedroom
        for s in ["train", "test", "val"]: 
            if self.livingroom_only:  # discard LivingDiningRoom 
                split2room[s] = [room[0] for room in data if (room[1] == s and room[0].split('-')[0]=="LivingRoom")] # LivingRoom-4353
            else:
                split2room[s] = [room[0] for room in data if room[1] == s]
                
        # scene paths (based off of split2room)
        folder_name = f"processed_{self.room_type}_augmented" if use_augment else f"processed_{self.room_type}"
        self.scene_dir = os.path.join(data_dir, folder_name)
       
        trainvalrooms = split2room["train"] + split2room["val"]
        self.scenes_tv = sorted([ os.path.join(self.scene_dir, e) 
                                  for e in list(os.listdir(self.scene_dir)) 
                                  if e.split("_")[1] in trainvalrooms ]) # NOTE: selecting the 2nd argument works for augmented as well
                                  # Example: <scene_dir>/ff7b42d9-0e58-4847-8d47-f793a11cd3bd_MasterBedroom-83934(_0)
                          
        self.scenes_test = sorted([ os.path.join(self.scene_dir, e) 
                                    for e in list(os.listdir(self.scene_dir)) 
                                    if e.split("_")[1] in split2room["test"] ])
        if print_info: print(f"TDFDataset: len(self.scenes_tv)={len(self.scenes_tv)}, len(self.scenes_test)={len(self.scenes_test)}\n")

        # preload data into RAM
        if self.livingroom_only:
            if os.path.exists(os.path.join(self.scene_dir, "data_tv_ctr_livingroomonly.npz")):
                self.data_tv = np.load( os.path.join(self.scene_dir, "data_tv_ctr_livingroomonly.npz"), allow_pickle=True)
            if os.path.exists(os.path.join(self.scene_dir, "data_test_ctr_livingroomonly.npz")):
                self.data_test = np.load( os.path.join(self.scene_dir, "data_test_ctr_livingroomonly.npz"), allow_pickle=True)
        else:
            if os.path.exists(os.path.join(self.scene_dir, "data_tv_ctr.npz")):
                self.data_tv = np.load( os.path.join(self.scene_dir, "data_tv_ctr.npz"), allow_pickle=True)
            if os.path.exists(os.path.join(self.scene_dir, "data_test_ctr.npz")):
                self.data_test = np.load( os.path.join(self.scene_dir, "data_test_ctr.npz"), allow_pickle=True)

        with open(os.path.join(self.scene_dir, "dataset_stats_all.txt")) as f:
            # Same regardless of if only living room (ctr.npz processed from boxes.npz, generated in one go from ATISS for all living+livingdiningrooms)
            # NOTE: data prepared with splits test+train+val (ATISS's preprocess_data.py)
            ds_js_all= json.loads(f.read()) 
        self.object_types = ds_js_all["object_types"] # class labels in order from 0 to self.cla_dim

        self.maxnobj = room_info["nobj"][self.room_type][1] # based on ATISS Suplementary Material
        self.maxnfpoc = room_info["maxnfpoc"][self.room_type][1] # bedroom: 25 (based on preprocessing data)
        self.nfpbpn = 250
        
        self.pos_dim = 2 # coord in x, y (disregard z); normalized to [-1,1]
        self.ang_dim = 2 # cos(theta), sin(theta), where theta is in [-pi, pi]
        self.siz_dim = 2 # length of bounding box in x, y; normalized to [-1, 1]
        self.cla_dim = len(self.object_types) # number of classes (19 for bedroom, 22 for all else)
        self.sha_dim = self.siz_dim+self.cla_dim

        self.cla_colors = list(plt.cm.rainbow(np.linspace(0, 1, self.cla_dim)))

        self.room_size = room_info["room_size"][self.room_type] #[rs_x, rs_y, rs_z]


    ## HELPER FUNCTION: agnostic of specific dataset configs
    @staticmethod
    def parse_cla(cla):
        """ cla: [nobj, cla_dim]

            nobj: scalar, number of objects in the scene
            cla_idx: [nobj,], each object's class type index. 
        """ 
        nobj = cla.shape[0]
        for o_i in range(cla.shape[0]):
            if np.sum(cla[o_i]) == 0: 
                nobj = o_i
                break
        cla_idx = np.argmax(cla[:nobj,:], axis=1) #[nobj,cla_dim] -> [nobj,] (each obj's class index)
        return nobj, cla_idx

    @staticmethod
    def reset_padding(nobjs, toreset):
        """ nobjs: [batch_size]
            toreset(2): [batch_size, maxnumobj, 2]
        """
        for scene_idx in range(toreset.shape[0]):
            toreset[scene_idx, nobjs[scene_idx]:,:]=0
        return toreset

    @staticmethod
    def get_objbbox_corneredge(pos, ang_rad, siz):
        """ pos: [pos_dim,]
            ang_rad: [1,1], rotation from (1,0) in radians
            siz: [siz_dim,], full bbox length

            corners: corner points (4x2 numpy array) of the rotated bounding box centered at pos and with bbox len siz,
            bboxedge: array of 2-tuples of numpy arrays [x,y]
        """
        siz = (siz.astype(np.float32))/2
        corners = np.array([[pos[0]-siz[0], pos[1]+siz[1]],  # top left (origin: bottom left)
                            [pos[0]+siz[0], pos[1]+siz[1]],  # top right
                            [pos[0]-siz[0], pos[1]-siz[1]],  # bottom left 
                            [pos[0]+siz[0], pos[1]-siz[1]]]) # bottom right #(4, 2)
        corners =  np_rotate_center(corners, np.repeat((ang_rad), repeats=4, axis=0), pos) # (4, 2/1/2) , # +np.pi/2, because our 0 degree means 90
            # NOTE: no need to add pi/2: obj already starts facing pos y, we directly rotate from that
        bboxedge = [(corners[2], corners[0]), (corners[0], corners[1]), (corners[1], corners[3]), (corners[3], corners[2])]
                    # starting from bottom left corner
        return corners, bboxedge

    @staticmethod
    def get_xyminmax(ptxy):
        """ptxy: [numpt, 2] numpy array"""
        return np.amin(ptxy[:,0]), np.amax(ptxy[:,0]), np.amin(ptxy[:,1]), np.amax(ptxy[:,1])



    ## HELPER FUNCTION: not agnostic of specific dataset configs
    def emd_by_class(self, noisy_pos, clean_pos, clean_ang, clean_sha, nobjs):
        """ For each scene, for each object, assign it a target object of the same class based on its position.
            Performs earthmover distance assignment based on pos from noisy to target, for instances of one class, 
            and assign ang correspondingly.

            pos/ang/sha: [batch_size, maxnumobj, ang/pos/sha(siz+cla)_dim]
            nobjs: [batch_size]
            clean_sha: noisy and target (clean) should share same shape data (size and class don't change)
        """
        numscene = noisy_pos.shape[0]
        noisy_labels = np.zeros((numscene, self.maxnobj, self.pos_dim+self.ang_dim))
        for scene_i in range(numscene):
            nobj = nobjs[scene_i]
            cla_idx = np.argmax(clean_sha[scene_i, :nobj, self.siz_dim:], axis=1) # (nobj, cla_dim) -> (nobj,) # example: array([ 9, 11, 15,  7,  2, 18, 15])
            for c in np.unique(cla_idx) : # example unique out: array([ 2,  7,  9, 11, 15, 18]) (indices of the 1 in one-hot encoding)
                objs = np.where(cla_idx==c)[0] # 1d array of obj indices whose class is c # example: array([2, 6]) for c=15
                p1 = [tuple(pt) for pt in noisy_pos[scene_i, objs, :]] # array of len(objs) tuples
                p2 = [tuple(pt) for pt in clean_pos[scene_i, objs, :]] # array of len(objs) tuples
                chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_i, objs, :]) # len(objs)x2: assigned pos for each pt in p1 (in that order)
                noisy_labels[scene_i, objs, 0:self.pos_dim] = np.array(chair_assignment)
                noisy_labels[scene_i, objs, self.pos_dim:self.pos_dim+self.ang_dim] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_i, nobj:, :] left as 0; each obj assigned exactly once
        return noisy_labels


    def add_gaussian_gaussian_noise_by_class(self, classname, noisy_orig_pos, noisy_orig_sha, noise_level_stddev=0.1):
        noisy_pos = np.copy(noisy_orig_pos) # [batch_size, maxnobj, pos]
        
        for scene_i in range(noisy_orig_pos.shape[0]):
            nobj, cla_idx = TDFDataset.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
            for o_i in range(nobj):
                if classname in self.object_types[cla_idx[o_i]]: # only add noise for chairs
                    noisy_pos[scene_i:scene_i+1, o_i:o_i+1, :] = np_add_gaussian_gaussian_noise(noisy_orig_pos[scene_i:scene_i+1, o_i:o_i+1, :], noise_level_stddev=noise_level_stddev)

        return noisy_pos


    def clever_add_noise(self, noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_nobj, noisy_orig_fpoc, noisy_orig_nfpc, noisy_orig_vol, 
                         noise_level_stddev, angle_noise_level_stddev, weigh_by_class=False, within_floorplan=False, no_penetration=False, max_try=None, pen_siz_scale=0.92):
        """ noisy_orig_pos/ang/sha: [batch_size, maxnobj, pos_dim/ang_dim/sha_dim]
            noisy_orig_fpoc:        [batch_size, maxnfpoc, pos_dim]
            noisy_orig_vol:         used only if weigh_by_class
        """
        if not weigh_by_class and not within_floorplan and not no_penetration:
            # NOTE: each scene has zero-mean gaussian distributions for noise
            # noisy_pos = add_gaussian_gaussian_noise_by_class("chair', noisy_orig_pos, noisy_orig_sha, noise_level_stddev=noise_level_stddev)
            noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
            noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
            noisy_pos, noisy_ang = TDFDataset.reset_padding(noisy_orig_nobj, noisy_pos), TDFDataset.reset_padding(noisy_orig_nobj, noisy_ang) # [batch_size, maxnumobj, dim]
            return noisy_pos, noisy_ang
                
        if max_try==None: 
            max_try=1 # weigh_by_class only, exactly 1 iteration through while loop (always reach break in first iter)
            if within_floorplan: max_try+= 1000
            if no_penetration: max_try+= 2000 # very few in range 300-1500 for bedroom

        noisy_pos = np.copy(noisy_orig_pos) # the most up to date arrangement, with 0 padded
        noisy_ang = np.copy(noisy_orig_ang)
        obj_noise_factor = 1 # overriden if weighing by class
        
        if weigh_by_class: obj_noise_factors = 1/np.sqrt((noisy_orig_vol+0.00001)*2) # [batch_size, maxnobj, 1], intuition: <1 for large objects, >1 for small objects
            # Purpose of *2: so not as extreme: vol<2, factor < 1/vol (not too large); > 2, factor > 1/vol (not too small)
        # Ending value are inf, but not used as we only consider up until nobj
        
        for scene_i in range(noisy_orig_pos.shape[0]): # each scene has its own noise level
            # NOTE: each scene has zero-mean gaussian distributions for noise and noise level
            scene_noise_level = abs(np.random.normal(loc=0.0, scale=noise_level_stddev)) # 68% in one stddev
            scene_angle_noise_level = abs(np.random.normal(loc=0.0, scale=angle_noise_level_stddev))

            parse_nobj, cla_idx = TDFDataset.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
            obj_indices = list(range(noisy_orig_nobj[scene_i]))
            random.shuffle(obj_indices) # shuffle in place
            for obj_i in obj_indices: # 0 padding unchanged
                # if "chair" not in self.object_types[cla_idx[obj_i]]: continue # only add noise for chairs

                if weigh_by_class: obj_noise_factor = obj_noise_factors[scene_i, obj_i, 0] # larger objects have smaller noise
                # print(f"\n--- obj_i={obj_i}: nf={nf}")
                try_count = -1
                while True:
                    try_count += 1
                    if try_count >= max_try: 
                        # print(f"while loop counter={try_count}")
                        break
                    obj_noise = obj_noise_factor * np.random.normal(size=(noisy_orig_pos[scene_i, obj_i:obj_i+1, :]).shape, loc=0.0, scale=scene_noise_level) # [1, pos_dim]
                    new_o_pos = noisy_orig_pos[scene_i, obj_i:obj_i+1, :] + obj_noise # [1, pos_dim]

                    obj_angle_noise = obj_noise_factor * np.random.normal(size=(1,1), loc=0.0, scale=scene_angle_noise_level) 
                    new_o_ang = np_rotate(noisy_orig_ang[scene_i, obj_i:obj_i+1, :], obj_angle_noise) # [1, ang_dim=2]

                    if within_floorplan or no_penetration: # if can skip, will always break
                        if not self.is_valid( obj_i, np.copy(new_o_pos), np.copy(new_o_ang),
                                            np.copy(noisy_pos[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_ang[scene_i, :noisy_orig_nobj[scene_i], :]),  # latest state
                                            np.copy(noisy_orig_sha[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_orig_fpoc[scene_i, :noisy_orig_nfpc[scene_i], :]), 
                                            within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=pen_siz_scale):
                            continue # try regenerating noise

                    # reached here means passed checks
                    noisy_pos[scene_i, obj_i:obj_i+1, :] = new_o_pos
                    noisy_ang[scene_i, obj_i:obj_i+1, :] = new_o_ang

                    # print(f"break @ try_count={try_count}")
                    break # continue to next object
      
        return noisy_pos, noisy_ang


    def is_valid(self, o_i, o_pos, o_ang, scene_pos, scene_ang, scene_sha, scene_fpoc, within_floorplan=True, no_penetration=True, pen_siz_scale=0.92):
        """ A object's pos + ang is valid if the object's bounding box does not intersect with any floor plan wall or other object's bounding box edge.
            Note this function modifies the input arguments in place.

            o_i: scalar, the index of the object of interest in the scene
            o_{pos, ang}: [1, dim], information about the object being repositioned
            scene_{pos, ang, sha}: [nobj, dim], info about the rest of the scene, without padding, the obj_ith entry of pos and ang is skipped
            scene_fpoc: [nfpoc, pos_dim], without padding, ordered (consecutive points form lines).
            pen_siz_scale: to allow for some minor intersection (respecting ground truth dataset)
        """
        # Dnormalize data to the same scale: [-3, 3] (meters) for both x and y(z) axes.
        room_size = np.array(self.room_size) #[x, y, z]
        o_pos = o_pos*room_size[[0,2]]/2  #[1, dim]
        scene_fpoc = scene_fpoc* room_size[[0,2]]/2 # from [-1,1] to [-3,3]
        scene_pos = scene_pos*room_size[[0,2]]/2 # [nobj, pos_dim]
        scene_ang = trig2ang(scene_ang) #[nobj, 1], in [-pi, pi]
        scene_siz = (scene_sha[:,:self.siz_dim] +1) * (room_size[[0,2]]/2)  # [nobj, siz_dim], bbox len 
    
        # check intersection with floor plan
        if within_floorplan:
            # check if obj o's pos and corners is outside floor plan's convex bounds
            fp_x_min, fp_x_max, fp_y_min, fp_y_max = TDFDataset.get_xyminmax(scene_fpoc)
            if ((o_pos[0,0]<fp_x_min) or (o_pos[0,0]>fp_x_max)or (o_pos[0,1]<fp_y_min) or (o_pos[0,1]>fp_y_max)): return False
            o_corners, o_bboxedge = TDFDataset.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]) # cor=(4,2)
            o_cor_x_min, o_cor_x_max, o_cor_y_min, o_cor_y_max = TDFDataset.get_xyminmax(o_corners)
            if ((o_cor_x_min<fp_x_min) or (o_cor_x_max>fp_x_max)or(o_cor_y_min)<fp_y_min) or (o_cor_y_max>fp_y_max): return False

            # check for intersection of boundaries
            for wall_i in range (len(scene_fpoc)):
                fp_pt1, fp_pt2 = scene_fpoc[wall_i], scene_fpoc[(wall_i+1)%len(scene_fpoc)]

                # Entirely out of bounds for each line (especially for concave shapes): scene_fpoc ordered counterclockwisely starting from bottom left corner 
                if(fp_pt1[0] == fp_pt2[0]): # vertical
                    if (fp_pt1[1]>=fp_pt2[1]): # top to bottom, right edge
                        if o_cor_x_min >= fp_pt1[0]: return False
                    else: # bottom to top, left edge
                        if o_cor_x_max <= fp_pt1[0]: return False
                if(fp_pt1[1] == fp_pt2[1]): # horizontal
                    if (fp_pt1[0]>=fp_pt2[0]): # from right to left, bottom edge
                        if o_cor_y_max <= fp_pt1[1]: return False
                    else: # from left to right top edge
                        if o_cor_y_min >= fp_pt1[1]: return False

                for edge_i in range(4): # obj is rectangular bounding box
                    if do_intersect(o_bboxedge[edge_i][0], o_bboxedge[edge_i][1], fp_pt1, fp_pt2):
                        return False

        # check intersection with each of the other objects
        if no_penetration:
            o_scale_corners, o_scale_bboxedge = TDFDataset.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]*pen_siz_scale)
            o_scale_cor_x_min, o_scale_cor_x_max, o_scale_cor_y_min, o_scale_cor_y_max = TDFDataset.get_xyminmax(o_scale_corners)

            for other_o_i in range (scene_pos.shape[0]):
                if other_o_i == o_i: continue # do not compare against itself
                other_scale_cor, other_scale_edg = TDFDataset.get_objbbox_corneredge(scene_pos[other_o_i], scene_ang[other_o_i:other_o_i+1,:], scene_siz[other_o_i]*pen_siz_scale)
                other_scale_cor_x_min, other_scale_cor_x_max, other_scale_cor_y_min, other_scale_cor_y_max = TDFDataset.get_xyminmax(other_scale_cor)
                
                # check entire outside one another
                if ((o_scale_cor_x_max<=other_scale_cor_x_min) or (o_scale_cor_x_min>=other_scale_cor_x_max) or
                    (o_scale_cor_y_max<=other_scale_cor_y_min) or (o_scale_cor_y_min>=other_scale_cor_y_max)):
                   continue # go check next obj

                # check if one is inside the other:
                if ((other_scale_cor_x_min <= o_scale_cor_x_min <= other_scale_cor_x_max) and (other_scale_cor_x_min <= o_scale_cor_x_max <= other_scale_cor_x_max) and
                    (other_scale_cor_y_min <= o_scale_cor_y_min <= other_scale_cor_y_max) and (other_scale_cor_y_min <= o_scale_cor_y_max <= other_scale_cor_y_max)):
                    return False
                if ((o_scale_cor_x_min <= other_scale_cor_x_min <= o_scale_cor_x_max) and (o_scale_cor_x_min <= other_scale_cor_x_max <= o_scale_cor_x_max) and
                    (o_scale_cor_y_min <= other_scale_cor_y_min <= o_scale_cor_y_max) and (o_scale_cor_y_min <= other_scale_cor_y_max <= o_scale_cor_y_max)):
                    return False
                # check if edges intersect
                for edge_i in range(4):
                    for other_edge_i in range(4):
                        if do_intersect(o_scale_bboxedge[edge_i][0], o_scale_bboxedge[edge_i][1], 
                                        other_scale_edg[other_edge_i][0], other_scale_edg[other_edge_i][1]):
                            return False

        return True

    def gen_random_selection(self, batch_size, data_partition='trainval'):
        if data_partition=='trainval':
            total_data_count = self.data_tv['pos'].shape[0]
        elif data_partition=='test':
            total_data_count = self.data_test['pos'].shape[0]
        elif data_partition=='all':
            total_data_count = self.data_tv['pos'].shape[0] + self.data_test['pos'].shape[0]
        return np.random.choice(total_data_count, size=batch_size, replace=False)   # False: each data can be selected once # (batch_size,)
    
    def gen_stratified_selection(self, n_to_select, data_partition='test'):
        """ Only makes sense for augmented dataset. Select at least 1 from each original scene, and the (n_to_select-n_original_scene) scenes
            are selected randomly from the remaining unselected scenes.
        """
        if data_partition=='trainval':
            data = self.data_tv
        elif data_partition=='test':
            data = self.data_test
        elif data_partition=='all':
            pass # TODO

        # first 'stratify the dataset'
        originalscene2id = {} # {scenedir: [10,2381,103,800]}
        for i , scenedir in enumerate(data['scenedirs']): # 00ecd5d3-d369-459f-8300-38fc159823dc_SecondBedroom-6249_0
            if scenedir[:-2] in originalscene2id:
                originalscene2id[scenedir[:-2]].append(i)
            else:
                originalscene2id[scenedir[:-2]]= [i] # originalscene2id[data['scenedirs'][24][:-2]]) = [24, 25, 26, 27]
        
        grouped_ids = []
        for originalscene in originalscene2id:
            grouped_ids.append(originalscene2id[originalscene])
        grouped_ids = np.array(grouped_ids)  # (224, 4) for bedroom, if not augmented, then (224, 1)
        
        selection=np.random.randint(grouped_ids.shape[1], size=[grouped_ids.shape[0]]) # for each row, pick 1 element
        guaranteed_selection = grouped_ids[range(grouped_ids.shape[0]), selection] # [grouped_ids.shape[0], ]
        if n_to_select < guaranteed_selection.shape[0]:
            return np.random.choice(guaranteed_selection, size=n_to_select, replace=False)  # False: each data can be selected once 

        remaining_unselected = np.delete(np.arange(data['scenedirs'].shape[0]), guaranteed_selection) # total nscene - guaranteed_selection.shape[0]
        remaining_selection = np.random.choice(remaining_unselected, size=n_to_select-guaranteed_selection.shape[0], replace=False)
        return np.append(guaranteed_selection, remaining_selection) # (n_to_select, )

        
    def get_conDor_feat():
        # instantiate a condor trainer
        # load all the category specific weights
        # For each furniture, find its type's super category, pass through the right model
        # have gen_3dfront return a batch_invsha
        pass




    ## DATA GENERATION: helper function for generating clean/ground truth data from 3DFRONT dataset
    def _gen_3dfront_batch_preload(self, batch_size, data_partition='trainval', use_floorplan=True, random_idx=None):
        """ Reads from preprocessed data npz files (already normalized) to return data for batch_size number of scenes.
            Variable data length is dealt with through padding with 0s at the end.

            random_idx: if given, selects these designated scenes from the set of all trainval or test data.
            
            Returns:
            batch_scenepaths: [batch_size], contains full path to the directory named as the scenepath (example
                             scenepath = '<scenedir>/004f900c-468a-4f70-83cc-aa2c98875264_SecondBedroom-27399')
            batch_nbj: [batch_size], contains numbers of objects for each scene/room.

            batch_pos: position has size [batch_size, maxnumobj, pos_dim]=[x, y], where [:,0:2,:] are the 2 tables,
                       and the rest are the chairs.
            batch_ang: [batch_size, maxnumobj, ang_dim=[cos(th), sin(th)] ]
            batch_sha: [batch_size, maxnumobj, siz_dim+cla_dim], represents bounding box lengths and
                       class of object/furniture with one hot encoding.
            
            batch_vol: [batch_size, maxnumobj], volume of each object's bounding box (in global absolute scale).

            floor plan representations:
               batch_fpoc:   [batch_size, maxnfpoc, pos_dim]. Floor plan ordered corners. For each scene, have a list of
                             ordered (clockwise starting at bottom left [-x, -y]) corner points of floor plan contour,
                             where consecutive points form a line. Padded with 0 at the end. Normalized to [-1, 1] in 
                             write_all_data_summary_npz.
               batch_nfpc:   [batch_size], number of floor plan corners for each scene
               batch_fpmask: [batch_size, 256, 256, 3]
               batch_fpbpn:  [batch_size, self.nfpbp=250, 4]. floor plan boundary points and normals, including corners.
                             [:,:,0:2] normalized in write_all_data_summary_npz                   
        """
        random_idx = self.gen_random_selection(batch_size, data_partition) if random_idx is None else random_idx
        data = self.data_tv 
        if data_partition=='test': data = self.data_test
        if data_partition=='all':  
            data = dict(self.data_tv)
            data_test = dict(self.data_test)
            for key in data: data[key] = np.concatenate([data[key], data_test[key]], axis=0)

        batch_scenepaths = []  
        for data_i in range(batch_size): 
            s = data['scenedirs'][random_idx[data_i]] # a6704fd9-02c2-42a6-875c-723b26a8048a_MasterBedroom-45545
            batch_scenepaths.append(os.path.join(self.scene_dir, s))
        batch_scenepaths = np.array(batch_scenepaths) # [] # numpy array of strings

        batch_nbj = data['nbj'][random_idx] # [] -> (batch_size,)

        batch_pos = data['pos'][random_idx] # np.zeros((batch_size, self.maxnobj, self.pos_dim)) # batch_size, maxnobj, pos_dim
        batch_ang = data['ang'][random_idx] # np.zeros((batch_size, self.maxnobj, self.ang_dim))
        batch_siz = data['siz'][random_idx] # np.zeros((batch_size, self.maxnobj, self.siz_dim)) # shape
        batch_cla = data['cla'][random_idx] # np.zeros((batch_size, self.maxnobj, self.cla_dim)) # shape
        
        batch_vol = data['vol'][random_idx] # np.zeros((batch_size, self.maxnobj, 1)) # shape

        batch_fpoc, batch_nfpc, batch_fpmask, batch_fpbpn = None, [], None, None
        if use_floorplan:
            # floor plan representation 1: floor plan ordered corners
            batch_fpoc = data['fpoc'][random_idx] # np.zeros((batch_size, self.maxnfpoc, self.pos_dim))
            batch_nfpc = data['nfpc'][random_idx] # [] (batch_size,)
            
            # floor plan representation 2: binary mask (1st channel is remapped_room_layout=drawing ctr/rescaled fpoc on empty mask)
            batch_fpmask = np.zeros((batch_size, 256, 256, 3))  # 1 3-channel mask per scene
            xy = generate_pixel_centers(256,256) / 128 -1 # [0 (0.5), 256 (255.5)] -> [0,2] -> [-1,1] # in the same coord system as vertices 
            for data_i in range(batch_size): 
                random_i = random_idx[data_i]
                new_contour_mask = np.zeros((256,256,1))
                ctr = data['ctr'][random_i] # (51, 1, 2)
                ctr = np.expand_dims(ctr[np.any(ctr, axis=2)], axis=1) # (numpt, none->1, 2), kept non-zero rows
                cv.drawContours(new_contour_mask, [ctr.astype(np.int32)], -1 , (255,255,255), thickness=-1) # thickness < 0 : fill
                batch_fpmask[data_i] = np.concatenate([new_contour_mask, np.copy(xy)], axis=2) #(256,256,1+2=3)

            # floor plan representation 3: floor plan boundary points & their normals
            batch_fpbpn = data['fpbpn'][random_idx] # (batch_size, self.nfpbp=250, 4)

        batch_sha = np.concatenate([batch_siz, batch_cla], axis=2)  # [batch_size, maxnumobj, 2+22=24]
        return batch_scenepaths, batch_nbj, batch_pos, batch_ang, batch_sha, batch_vol, batch_fpoc, batch_nfpc, batch_fpmask, batch_fpbpn


    def _gen_3dfront_batch_onthefly(self, batch_size, data_partition='trainval', use_floorplan=True):
        """ Individually generate (and normalize) data for each of batch_size number of scenes.
            Outputs the same as _gen_3dfront_batch_preload.
        """
        batch_pos = np.zeros((batch_size, self.maxnobj, self.pos_dim))
        batch_ang = np.zeros((batch_size, self.maxnobj, self.ang_dim))
        batch_siz = np.zeros((batch_size, self.maxnobj, self.siz_dim)) # shape
        batch_cla = np.zeros((batch_size, self.maxnobj, self.cla_dim)) # shape
        batch_vol = np.zeros((batch_size, self.maxnobj, 1)) # shape
        batch_nbj = []
        batch_fpoc = np.zeros((batch_size, self.maxnfpoc, self.pos_dim)) # Pad with other values? # floor plan ordered corners (neighboring pt form a line)
        batch_nfpc = []
        batch_scenepaths = [] # list of strings
        batch_fpmask = np.zeros((batch_size, 256, 256, 3))  # 1 3-channel mask per scene

        for i in range(batch_size):
            scenepath = random.choice(self.scenes_tv) if data_partition=='trainval' else random.choice(self.scenes_test)
            scene_data = np.load(os.path.join(scenepath, "boxes.npz"), allow_pickle=True)
            # print("\n", scenepath)

            batch_scenepaths.append(scenepath)

            nobj = scene_data['jids'].shape[0]
            batch_nbj.append(nobj)

            # normalize data
            batch_pos[i,:nobj,0] = scene_data['translations'][:,0]/(self.room_size[0]/2) #[-3, 3] -> [-1,1]
            batch_pos[i,:nobj,1] = scene_data['translations'][:,2]/(self.room_size[2]/2) #[-3, 3] -> [-1,1] # use z as y
            batch_ang[i,:nobj,0:1] = np.cos(scene_data['angles']*-1) # since pos z (out of screen) -> neg y (vertical flip)
            batch_ang[i,:nobj,1:2] = np.sin(scene_data['angles']*-1)
            batch_siz[i,:nobj,0] = scene_data['sizes'][:,0]*2 /(self.room_size[0]/2)-1  #[0,3]*2 (only half of box len) -> [0,6]/3 -> [0,2]-1 -> [-1,1]
            batch_siz[i,:nobj,1] = scene_data['sizes'][:,2]*2 /(self.room_size[2]/2)-1  #[0,3]*2 (only half of box len) -> [0,6]/3 -> [0,2]-1 -> [-1,1]# use z as y
            batch_cla[i,:nobj,:] = scene_data['class_labels'][:,:self.cla_dim] # subsequent columns are 0 (start, end tokens)
            
            batch_vol[i,:nobj,0] = scene_data['sizes'][:,0] * scene_data['sizes'][:,1] * scene_data['sizes'][:,2] * 8 # absolute scale volume
            
            batch_fpoc, batch_nfpc, batch_fpmask, batch_fpbpn = None, [], None, None
            if use_floorplan:
                ## floor plan representation 1: floor plan ordered corners
                corners = scene_data["floor_plan_ordered_corners"] # (numpt, 2), in [-3, 3]
                batch_fpoc[i, :corners.shape[0], 0] = corners[:,0] /(self.room_size[0]/2) # scale to [-1, 1]
                batch_fpoc[i, :corners.shape[0], 1] = corners[:,1] /(self.room_size[2]/2) # scale to [-1, 1]
                batch_nfpc.append(corners.shape[0])

                ## floor plan representation 2: binary mask (1st channel is remapped_room_layout=drawing ctr/rescaled fpoc on empty mask)
                fpmask = scene_data["remapped_room_layout"]  #(256,256,1), binary occupancy
                xy = generate_pixel_centers(256,256) / 128 -1 # [0 (0.5), 256 (255.5)] -> [0,2] -> [-1,1] # in the same coord system as vertices 
                batch_fpmask[i] = np.concatenate([fpmask, xy], axis=2) #(256,256,1+2=3)
                    
                ## floor plan representation 3: floor plan boundary points & their normals
                # TODO

        batch_sha = np.concatenate([batch_siz, batch_cla], axis=2)  # [batch_size, maxnumobj, 2+22=24]
        return batch_scenepaths, np.array(batch_nbj), batch_pos, batch_ang, batch_sha, batch_vol, batch_fpoc, np.array(batch_nfpc), batch_fpmask, batch_fpbpn




    ## DATA GENERATION: public API
    def gen_3dfront(self, batch_size, random_idx=None, data_partition='trainval', use_emd=True, 
                    abs_pos=True, abs_ang=True, use_floorplan=True, noise_level_stddev=0.1, angle_noise_level_stddev=np.pi/12,
                    weigh_by_class = False, within_floorplan = False, no_penetration = False, pen_siz_scale=0.92, 
                    is_classification=False, replica= ""):
        """ Main entry point for generating data form the 3D-FRONT dataset.

            batch_size: number of scenes
            abs_pos/ang: if True, the returned labels are (final ordered assigned pos/ang); otherwise, use relative pos/ang
                         = (final ordered assigned pos - initial pos, angle between final and initial angles)
            noise_level_stddev, angle_noise_level_stddev: for adding position and angle noise. 
                                                          input: {clean scenes + gaussian gaussian noise}, labels: {clean scene without noise + emd}
            pen_siz_scale: value between 0 and 1, the lower the value, the more intersection among objects it allows when adding noise
            replica: if not "room_0”, no effect; if "room_0", use room 0 (living room) from replica dataset as clean scene - for additional testing
            
            Intermediary: {clean, noisy}_{pos, ang, sha} have shape [batch_size, maxnumobj, 2]

            Returns: 
                input:  [batch_size, maxnumobj, pos+ang+siz+cla] = [x, y, cos(th), sin(th), bbox_lenx, bbox_leny, one-hot-encoding for class]
                label:  [batch_size, maxnumobj, pos+ang] = [x, y, cos(th), sin(th)]
                padding_mask: [batch_size, maxnumobj], for Transformer (nn.TransformerEncoder) to ignore padded zeros (value=False for not masked)
                scenepaths: [batch_size,] <class 'numpy.ndarray'> of  <class 'numpy.str_'>
                fpoc:   [batch_size, maxnfpoc=51, pos], floor plan ordered corners; pointnet
                nfpc:   [batch_size], number of floor plan corners for each scene; pointnet
                fpmask: [batch_size, 256, 256, 3], floor plan binary mask; resnet
                fpbpn:  [batch_size, nfpbp=250, 4], floor plan boundary points and normals; pointnet_simple

            Clean input's labels are identical to the input. Noisy input's labels are ordered assignment to clean position for each 
            point (through earthmover distance assignment), using the original clean sample (noisy_orig_pos) as label, and angle labels
            are the corresponding angles from the assignment.
        
            Note that this implementation takes a non-hierarhical, non-scene-graph, free-for-all approach: emd assignment is done freely 
            among all objects of a class. 

            Also note that in generating the noisy positions, chairs are given greater noise while tables are given less. This is to
            teach the network to move certain types less.
        """
        clean_scenepaths, clean_nobj, clean_pos, clean_ang, clean_sha, clean_vol, clean_fpoc, clean_nfpc, clean_fpmask, clean_fpbpn = self._gen_3dfront_batch_preload(batch_size, data_partition=data_partition, use_floorplan=use_floorplan, random_idx=random_idx)
        if False and random_idx == [1748]: # for rebuttal: manually swapping tv_stand and multi_seat_sofa (teaser right, b9d17d23-66f0-445d-acb7-f11887cc4f7f_LivingDiningRoom-192367)
            clean_pos[0, 1, 0:2] = [-2.1/(self.room_size[0]/2), 1.87363994e+00/(self.room_size[0]/2)]
            clean_ang[0, 1, 0:2] = [9.13540407e-06, -1.00000000e+00]
            clean_pos[0, 5, 0:2] = [5.40149999e-01/(self.room_size[0]/2), 1.97992003e+00/(self.room_size[0]/2)]
            clean_ang[0, 5, 0:2] = [9.13540407e-06, 1.00000000e+00]
        if replica=="room_0": clean_scenepaths, clean_nobj, clean_pos, clean_ang, clean_sha, clean_vol, clean_fpoc, clean_nfpc, clean_fpmask, clean_fpbpn = gen_room0(batch_size)
        
        # input: pos, ang, siz, cla
        perturbed_pos, perturbed_ang = self.clever_add_noise( clean_pos, clean_ang, clean_sha, clean_nobj, clean_fpoc, clean_nfpc, clean_vol, noise_level_stddev, angle_noise_level_stddev,
                                                              weigh_by_class=weigh_by_class, within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=pen_siz_scale)

        input = np.concatenate([perturbed_pos, perturbed_ang, clean_sha], axis=2) # [batch_size, maxnumobj, dims]

        # label: pos, ang
        if is_classification: # for 3dfront model evaluation
            classification_token = np.concatenate([np.array([[[0,0, 1,0, 1,1]]]), np.zeros((1,1,self.cla_dim))], axis=2) # [1, 1, 2+2+2+19=25] -> [batch_size, 1, dims=25]
            input = np.concatenate([input, np.repeat(classification_token, batch_size, axis=0)], axis=1)  # [batch_size, maxnumobj+1, pos+ang+siz+cla=dims=25]
            # classification labels: [batch_size, 1]; clean is 1, noisy is 0 = probability(clean)
            labels = np.concatenate([(np.zeros((batch_size//2, 1))+1), np.zeros((batch_size//2, 1))], axis=0)
        else:
            # Calculate absolute labels for noisy
            if use_emd:
                labels = self.emd_by_class(np.copy(perturbed_pos), np.copy(clean_pos), np.copy(clean_ang), np.copy(clean_sha), np.copy(clean_nobj))
            else:
                labels = np.copy(np.concatenate([clean_pos, clean_ang], axis=2) ) # directly use original scene
            # If needed, overwrite it with relative
            if not abs_pos:
                labels[:,:,:self.pos_dim] = labels[:,:,:self.pos_dim] - perturbed_pos
            if not abs_ang: # relative: change noisy_labels from desired pos and ang to displacement
                ang_diff = np_angle_between(perturbed_ang, labels[:,:,self.pos_dim:self.pos_dim+self.ang_dim]) # [batch_size, nobj, 1].
                    # both input to np angle between are normalized, from noisy_ang(noisy_input) to noisy_labels(noisy_orig_ang), in [-pi, pi], 
                labels[:,:,self.pos_dim:self.pos_dim+1]              = np.cos(ang_diff) # [batch_size, nobj, 1]
                labels[:,:,self.pos_dim+1:self.pos_dim+self.ang_dim] = np.sin(ang_diff) # [batch_size, nobj, 1]


        # padding mask: for Transformer to ignore padded zeros
        padding_mask = np.repeat([np.arange(self.maxnobj)], batch_size, axis=0) #[batch_size,maxnobj]
        padding_mask = (padding_mask>=clean_nobj.reshape(-1,1)) # [batch_size,maxnobj] >= [batch_size, 1]: [batch_size,maxnobj] 
            # False=not masked (unchanged), True=masked, not attended to (isPadding)
        if is_classification: # add to padding for classifcation token at the end (not masked: false==0)
            padding_mask = np.concatenate([padding_mask, np.zeros((batch_size, 1))], axis=1) # [batch_size, maxnobj+1]


        # floor plan (3 representaitons): fpoc + nfpc, fpmask, fpbpn
        fpoc, nfpc, fpmask, fpbpn = None, None, None, None
        if use_floorplan: fpoc, nfpc, fpmask, fpbpn = clean_fpoc, clean_nfpc, clean_fpmask, clean_fpbpn

        return input, labels, padding_mask, clean_scenepaths, fpoc, nfpc, fpmask, fpbpn



    def gen_3dfront_halfsplit(self, half_batch_size, data_partition='trainval', use_emd=True, abs_pos=True, abs_ang=True, use_floorplan=True,
                              noise_level_stddev=0.1, angle_noise_level_stddev=np.pi/12,
                              weigh_by_class = False, within_floorplan = False, no_penetration = False, pen_siz_scale=0.92, is_classification=False):
        """  
            Half of the generated scene (clean) do not have any noise added, whereas the other half (noisy) does.
        """
        clean_scenepaths, clean_nobj, clean_pos, clean_ang, clean_sha, clean_vol, clean_fpoc, clean_nfpc, clean_fpmask, clean_fpbpn = self._gen_3dfront_batch_preload(half_batch_size,  data_partition=data_partition, use_floorplan=use_floorplan) # NOTE: no noise for clean
        noisy_orig_scenepaths, noisy_orig_nobj, noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_vol, noisy_orig_fpoc, noisy_orig_nfpc, noisy_orig_fpmask, noisy_orig_fpbpn = self._gen_3dfront_batch_preload(half_batch_size, data_partition=data_partition, use_floorplan=use_floorplan)
        noisy_pos, noisy_ang = self.clever_add_noise(noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_nobj, noisy_orig_fpoc, noisy_orig_nfpc, noisy_orig_vol, noise_level_stddev, angle_noise_level_stddev, 
                                                     weigh_by_class=weigh_by_class,within_floorplan=within_floorplan,no_penetration=no_penetration, pen_siz_scale=pen_siz_scale)

        # scenepath, nobj
        scenepaths = np.concatenate([clean_scenepaths, noisy_orig_scenepaths], axis=0)# 1d list (batch_size,) <class 'numpy.ndarray'> of  <class 'numpy.str_'>
        nobjs = np.concatenate([clean_nobj, noisy_orig_nobj], axis=0) # [half_batch_size + half_batch_size = batch_size]
        # pos, ang, siz, cla
        clean_input = np.concatenate([clean_pos, clean_ang, clean_sha], axis=2) # [half_batch_size, maxnumobj, dims]
        noisy_input = np.concatenate([noisy_pos, noisy_ang, noisy_orig_sha], axis=2) # [half_batch_size, maxnumobj, dims]
        input = np.concatenate([clean_input, noisy_input], axis=0)  # [batch_size, maxnumobj, dims]
        
        if is_classification:
            classification_token = np.concatenate([np.array([[[0,0, 1,0, 1,1]]]), np.zeros((1,1,self.cla_dim))], axis=2) # [1, 1, 2+2+2+19=25] -> [batch_size, 1, dims=25]
            input = np.concatenate([input, np.repeat(classification_token, half_batch_size*2, axis=0)], axis=1)  # [batch_size, maxnumobj+1, pos+ang+siz+cla=dims=25]
            # classification labels: [batch_size, 1]; clean is 1, noisy is 0 (probability to be clean)
            labels = np.concatenate([(np.zeros((half_batch_size, 1))+1), np.zeros((half_batch_size, 1))], axis=0)
        else:
            clean_labels_pos = np.copy(clean_pos) if abs_pos else np.zeros((half_batch_size, self.maxnobj, self.pos_dim))
            clean_labels_ang = np.copy(clean_ang) if abs_ang else np.zeros((half_batch_size, self.maxnobj, self.ang_dim))
            clean_labels = np.copy(np.concatenate([clean_labels_pos, clean_labels_ang], axis=2))
            
            # first calculate absolute labels for noisy
            if use_emd:
                noisy_labels = self.emd_by_class(np.copy(noisy_pos), np.copy(noisy_orig_pos), np.copy(noisy_orig_ang), np.copy(noisy_orig_sha), np.copy(noisy_orig_nobj))
            else:
                noisy_labels = np.copy(np.concatenate([noisy_orig_pos, noisy_orig_ang], axis=2) ) # directly use original scene
            # if needed, overwrite it with relative
            if not abs_pos:
                noisy_labels[:,:,:self.pos_dim] = noisy_labels[:,:,:self.pos_dim] - noisy_pos
            if not abs_ang: # relative: change noisy_labels from desired pos and ang to displacement
                ang_diff = np_angle_between(noisy_ang, noisy_labels[:,:,self.pos_dim:self.pos_dim+self.ang_dim]) # [half_batch_size, nobj, 1].
                    # both input to np angle between are normalized, from noisy_ang(noisy_input) to noisy_labels(noisy_orig_ang), in [-pi, pi], 
                noisy_labels[:,:,self.pos_dim:self.pos_dim+1]              = np.cos(ang_diff) # [half_batch_size, nobj, 1]
                noisy_labels[:,:,self.pos_dim+1:self.pos_dim+self.ang_dim] = np.sin(ang_diff) # [half_batch_size, nobj, 1]

            labels = np.concatenate([clean_labels, noisy_labels], axis=0)
            
        # padding mask
        padding_mask = np.repeat([np.arange(self.maxnobj)], half_batch_size*2, axis=0) #[batch_size,maxnobj]
        padding_mask = (padding_mask>=nobjs.reshape(-1,1)) # [batch_size,maxnobj] >= [batch_size, 1]: [batch_size,maxnobj] 
            # False: not masked (unchanged), True=not attended to (isPadding)
        if is_classification: # add to padding for classifcation token at the end (not masked: false==0)
            padding_mask = np.concatenate([padding_mask, np.zeros(( half_batch_size*2, 1))], axis=1) # [batch_size, maxnobj+1]

        # floor plan (3 representaitons): fpoc, nfpc + fpmask + fpbpn
        fpoc, nfpc, fpmask, fpbpn = None, None, None, None
        if use_floorplan: 
            fpoc = np.concatenate([clean_fpoc, noisy_orig_fpoc], axis=0) # [batch_size, maxnfpoc, pos_dim]
            nfpc = np.concatenate([clean_nfpc, noisy_orig_nfpc], axis=0) # [batch_size] 
            fpmask = np.concatenate([clean_fpmask, noisy_orig_fpmask], axis=0) # [batch_size, 256,256,3]
            fpbpn = np.concatenate([clean_fpbpn, noisy_orig_fpbpn], axis=0) # [batch_size, self.nfpbp=250, 4]

        return input, labels, padding_mask, scenepaths, fpoc, nfpc, fpmask, fpbpn





    ## VISUALIZATION
    def read_one_scene(self, scenepath=None, normalize=False):
        """ Reads data from boxes npz files

            scenepath: if not provided, randomly choose a scene. 
                       Not the full path, just the room scene id / directory name containing the boxes.npz file
            normalize: if true, normalize data to [-1,1] as in training data preparation. Default to false for visualization.

            Returns:
            input: [nobj, pos_d+ang_d+siz_d+cla_d]
        """
        scenepath = random.choice(self.scenes_test) if scenepath is None else os.path.join(self.scene_dir, scenepath)
        # print("\n", scenepath)
        scene_data = np.load(os.path.join(scenepath, "boxes.npz"), allow_pickle=True)

        nobj = scene_data['jids'].shape[0]
        
        scene_pos = np.zeros((nobj, self.pos_dim))
        scene_ang = np.zeros((nobj, self.ang_dim))
        scene_siz = np.zeros((nobj, self.siz_dim))
        scene_cla = np.zeros((nobj, self.cla_dim))

        scene_pos[:,0] = scene_data['translations'][:,0] # [-3, 3]
        scene_pos[:,1] = scene_data['translations'][:,2]
        if normalize: 
            scene_pos[:,0] /= (self.room_size[0]/2) # [-1,1]
            scene_pos[:,1] /= (self.room_size[2]/2) # [-1,1]
            
        scene_ang[:,0:1] = np.cos(scene_data['angles']*-1) # since pos z (out of screen) -> neg y (vertical flip)
        scene_ang[:,1:2] = np.sin(scene_data['angles']*-1)

        scene_siz[:,0] = scene_data['sizes'][:,0]*2 # [0,3]*2 original: half of bounding box length
        scene_siz[:,1] = scene_data['sizes'][:,2]*2
        if normalize: 
            scene_siz[:,0] = scene_siz[:,0]/(self.room_size[0]/2)-1 # [-1,1]
            scene_siz[:,1] = scene_siz[:,1]/(self.room_size[2]/2)-1

        scene_cla = scene_data['class_labels'][:,:self.cla_dim] # subsequent columns are 0

        input = np.concatenate([scene_pos, scene_ang, scene_siz, scene_cla], axis=1)

        return input, scenepath
    


    def visualize_tdf_2d_denoise(self, traj, args=None, vis_traj=True, scenepath=None, fp='vis_2d_pointcloud_eval.jpg', title='Final assignment', fpoc=None):
        """ Graph the given one scene/room on 2d xy plane.
            traj: [iter, numobj, pos_d+ang_d+siz_d+cla_d]
            args: needed only for 3d visualization; fields needed: {to_3dviz, room_type, to_keyshot}
            vis_traj: whether to show movement trajectory of objects through lines
            scenepath: example = <full_path>/fe174e21-5e11-4004-8347-f4e5e2e7b30c_LivingDiningRoom-20428 (directory with boxes.npz)
            fpoc: only used if scenepath is None, [nfpc, pos_dim=2]
        """
        P, A, S = self.pos_dim, self.ang_dim, self.siz_dim
        nobj, cla_idx = TDFDataset.parse_cla(traj[0,:,P+A+S:]) # uses initial cla for all snapshot in traj (generated input, never perturbed)
        
        if not vis_traj: traj = traj[-2:-1] # only visualize final

        # back to original scale in boxes.npz (absolute scale in 6x6m)
        traj[:,:nobj,0:1] *= (self.room_size[0]/2) # [-1,1] -> [-3,3] (okay to modify in place)
        traj[:,:nobj,1:2] *= (self.room_size[2]/2) # [-1,1] -> [-3,3] # use z as y
        traj[:, :nobj, P+A:P+A+1] = (traj[:, :nobj, P+A:P+A+1]+1)*(self.room_size[0]/2) #[-1,1] -> [0,2] -> [0,6] (full bbox len)
        traj[:, :nobj, P+A+1:P+A+2] = (traj[:, :nobj, P+A+1:P+A+2]+1)*(self.room_size[2]/2) #[-1,1] -> [0,2] -> [0,6] (full bbox len) # use z as y

        t=traj if vis_traj else None

        self.visualize_tdf_2d(traj[-1], f"{fp}", f"{title}", args=args, traj=t, nobj=nobj, cla_idx=cla_idx, scenepath=scenepath, fpoc=fpoc)


    def visualize_tdf_2d(self, scene, fp, title, args=None, traj=None, nobj=None, cla_idx=None, scenepath=None, show_corner=False, show_fpbpn=False, fpoc=None):
        """ Visualize one given scene.
            scene:     has shape [:, pos_d+ang_d+siz_d+cla_d]. Visualized on top of trajectory (if applicable).
            fp:        filepath
            title:     title of saved figure
            args:      needed only for 3d visualization; fields needed: {to_3dviz, room_type, to_keyshot}
            traj:      if given, show movement trajectory of objects through lines
            scenepath: for visualizing floor plan. If given, should be the full path to directory containing boxes npz.
            fpoc:      floor plan ordered corners, only used if scenepath is None, [nfpc, pos_dim=2]
        """
        if (nobj is None) or (cla_idx is None):  
            nobj, cla_idx = TDFDataset.parse_cla(scene[:,self.pos_dim+self.ang_dim+self.siz_dim:]) # generated input, never perturbed

        siz = scene[:nobj, self.pos_dim+self.ang_dim : self.pos_dim+self.ang_dim+self.siz_dim]  # bbox lengths
        arrows = np_rotate(scene[:nobj,self.pos_dim:self.pos_dim+self.ang_dim], np.zeros((nobj,1))+np.pi/2) # rot counterclockwise about origin as these are directional vec
            # NOTE: add 90 deg because of offset between 0 degree angle visualization (1,0) and 0 degree obj, which faces positive y/(0, 1)
                
        fig = plt.figure(dpi=300, figsize=(5,5))
        for o_i in range(nobj):
            c =self.cla_colors[cla_idx[o_i]].reshape(1,-1) # (4,) -> (1,4)
            if traj is not None: plt.plot(traj[:,o_i,0], traj[:,o_i,1], c=c, alpha=0.45) # bottom layer
            plt.text(scene[o_i,0]-0.1, scene[o_i,1]+0.05, self.object_types[cla_idx[o_i]], size=6) 
            plt.scatter(scene[o_i,0], scene[o_i,1], c=c) 
            plt.quiver(scene[o_i,0], scene[o_i,1], arrows[o_i,0], arrows[o_i,1], color=c)
            plt.gca().add_patch(plt.Rectangle(
                                    (scene[o_i,0]-siz[o_i,0]/2, scene[o_i,1]-siz[o_i,1]/2), 
                                    siz[o_i,0], siz[o_i,1], linewidth=1, edgecolor=c, facecolor='none',
                                    transform = mt.Affine2D().rotate_around(scene[o_i,0], scene[o_i,1], np.arctan2(scene[o_i,3], scene[o_i,2])) +plt.gca().transData 
                                ))
        
        R = (room_info["room_size"][self.room_type][0]/2)
        rang = [-R, R] #[-3,3] for bedroom

        # floor plan related
        if scenepath is not None:
            scene_data = np.load(os.path.join(scenepath, "boxes.npz"), allow_pickle=True)
            plt.imshow(np.squeeze(scene_data["remapped_room_layout"]), alpha=0.2,  extent=(rang[0],rang[1],rang[1],rang[0]), cmap='gray', vmin=0, vmax=255.) # flip vertically       
            if show_corner:
                corners = scene_data["floor_plan_ordered_corners"] # num_pt, 2
                plt.scatter(corners[:,0], corners[:,1]) 
            if show_fpbpn:
                fpbpn = scene_data["floor_plan_boundary_points_normals"] # num_pt, 4
                plt.plot(fpbpn[:,0], fpbpn[:,1], 'or', markersize=3, alpha=0.7)
                
                # choice = np.round(np.linspace(0, len(self.nfpbpn)-1, num=5)).astype(int)
                for i in range(0, self.nfpbpn, 5): # 250/5 = 50 points 
                    plt.quiver(fpbpn[i,0], fpbpn[i,1], fpbpn[i,2]+0.00001, fpbpn[i,3]+0.00001, width=0.005) # at the start pt
        else:
            if fpoc is not None:
                plt.plot(np.append(fpoc[:,0], [fpoc[0,0]]), np.append(fpoc[:,1], [fpoc[0,1]]), 'o-', c='k', markersize=3, alpha=1) # line connecting circle

        plt.gca().set(xlim=rang, ylim=rang)
        roomid="" if scenepath is None else os.path.split(scenepath)[1]
        plt.title(f"{title}\n{roomid}", fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        fig.tight_layout()
        plt.savefig(f"{fp}.jpg")
        plt.close(fig)
            
        # if args and args["to_3dviz"]:
        #     room_side = 6.2 if args["room_type"] == "livingroom" else 3.1 # bedroom
        #     render_scene(scene, scenepath, fp, room_side, topdownview=True, render_keyshot=args["to_keyshot"])
            

if __name__ == '__main__':
    tdf=TDFDataset("livingroom", use_augment=False)

    ## Example data generation
    input, labels, padding_mask, clean_scenepaths, fpoc, nfpc, fpmask, fpbpn  = tdf.gen_3dfront(1, data_partition='trainval', use_floorplan=True)
    print(f"input: {input}\nlabels: {labels}\n")
    print(f"input.shape={input.shape}, labels.shape={labels.shape}, padding_mask.shape={padding_mask.shape}, clean_scenepaths.shape={clean_scenepaths.shape}")
    print(f"fpoc.shape={fpoc.shape}, nfpc.shape={nfpc.shape}, fpmask.shape={fpmask.shape}, fpbpn.shape={fpbpn.shape}\n\n\n")

    ## Example scene visualization
    sceneid = "b9d17d23-66f0-445d-acb7-f11887cc4f7f_LivingDiningRoom-192367" #_0 for augmented（teaser right, random idx = 1748)
    input, scenepath = tdf.read_one_scene(scenepath=sceneid) # scenepath: full path
    tdf.visualize_tdf_2d(input, f"TDFront_{sceneid}", f"Original", traj=None, scenepath=scenepath, show_corner=False)

    input[1, 0:4] = [-2.1, 1.87363994e+00, 9.13540407e-06, -1.00000000e+00]  # tv_stand 
    input[5, 0:4] = [5.40149999e-01, 1.97992003e+00, 9.13540407e-06, 1.00000000e+00] # multi_seat_sofa
    tdf.visualize_tdf_2d(input, f"TDFront_{sceneid}_swapped", f"Swapped", traj=None, scenepath=scenepath, show_corner=False)
