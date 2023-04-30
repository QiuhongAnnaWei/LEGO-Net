
import os, random
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import copy
from scipy.interpolate import interp1d

sys.path.insert(1, os.getcwd())

from data.utils import *
from data.distance import *
from data.TDFront import TDFDataset, room_info


def process_floorplan_iterative_closest_point(tdf, scenepath=None, savedir=None, to_vis=True, to_save_new_contour=True):
        """ Returns mapped_corner: numpy array of shape [numpt, 2], scaled in [-3,3] (not normalized)
            tdf: an instance of TDFDataset
        """
        scenepath = random.choice(tdf.scenes_test) if scenepath is None else os.path.join(tdf.scene_dir, scenepath)
        scene_data = np.load(os.path.join(scenepath, "boxes.npz"))
        
        ## Source: contour points found on room_layout mask
        room_layout = np.squeeze(scene_data["room_layout"])
        all_contours, hierarchy = cv.findContours(room_layout, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # all_contours: tuple of contours in image. Each contour = numpy array of (x,y) coordinates of boundary points [numpt, 1, 2]
        contour = all_contours[0] if len(all_contours)==1 else max(all_contours, key = cv.contourArea) # a few edge cases have > 1 enclosed region
        contour = np.squeeze(contour) # (numcontourpt,1,2) -> (numcontourpt,2)
        R = (room_info["room_size"][tdf.room_type][0]/2)
        contour = contour / (room_layout.shape[0]) * (R*2) - R #[-3,3]
        original_contour = np.copy(contour)

        ## Target: ATISS's generated points, extracted from 3DFRONT mesh objects in json
        corners = np.unique(scene_data["floor_plan_vertices"],axis=0) # num_pt, 2 -> num_uniquept, 2
        corners += -scene_data["floor_plan_centroid"]  # +((-R)-floor_plan_centroid) centers it, +(R) changes from [-6,0] to [-3,3] # centroid=(3,) 
        corners = corners[:,[0,2]] 

        ## Iterative closest point
        max_iter = 100
        dist_to_discard = 0.25 if tdf.room_type=="bedroom" else 0.8
        for iter in range(max_iter):
            scale_sum, new_contour, mapped_corner = 0, [], []
            for conpt in contour:
                distance = np.array([euclidean_distance(conpt, c) for c in corners]) # 1d array
                min_index = np.argmin(distance)
                if distance[min_index] > dist_to_discard: continue # no matching mesh corner points, discard
                
                new_contour.append(conpt) # keep it in next iteration
                mapped_corner.append(corners[min_index]) # for if we break
                scale_sum += np.linalg.norm(corners[min_index]) / np.linalg.norm(conpt) # we take its average
            
            new_contour = np.array(new_contour)
            transform_scale = scale_sum/new_contour.shape[0]
            if abs(transform_scale-1) < 0.01: break
            contour = new_contour*transform_scale # transform

        ordered_unique_idx = sorted(np.unique(mapped_corner, axis=0, return_index=True)[1]) # mapped_corner retains order of contour
        mapped_corner = np.array([mapped_corner[i] for i in ordered_unique_idx])

        if to_vis:
            R = (room_info["room_size"][tdf.room_type][0]/2)
            rang = [-R, R] #[-3,3] for bedroom
            roomid="" if scenepath is None else os.path.split(scenepath)[1]
            fig = plt.figure(dpi=300, figsize=(5,5))
            plt.imshow(np.squeeze(scene_data["room_layout"]), alpha=0.2,  extent=(rang[0],rang[1],rang[1],rang[0]), cmap='gray', vmin=0, vmax=255.) 
            
            plt.scatter(corners[:,0], corners[:,1], s=11, c='b')
            plt.scatter(original_contour[:,0], original_contour[:,1], s=11, c='orange')
            plt.scatter(mapped_corner[:,0], mapped_corner[:,1], s=11 , c='red', alpha=0.5)
            for i in range(mapped_corner.shape[0]):
                plt.text(mapped_corner[i,0]-0.1, mapped_corner[i,1]+0.05, i, size=7) 
            
            plt.gca().set(xlim=rang, ylim=rang)
            plt.title(f"ICP({iter}): vertices(blue){corners.shape[0]} - contour(orange){original_contour.shape[0]} - mappedcontour(red){mapped_corner.shape[0]}\n{roomid}", fontsize=8)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            fig.tight_layout()
            plt.savefig(os.path.join(savedir, f"{scenepath.split('/')[-1]}_icpidx.jpg"))
            plt.close(fig)
        
        if to_save_new_contour:
            new_contour_mask = np.zeros((256,256,1))
            ctr = np.expand_dims( (mapped_corner+R) /(R*2) * 256, 1).astype(np.int32) # [numpt, 1, 2], .astype(numpy.int32)
            cv.drawContours(new_contour_mask, [ctr], -1 , (255,255,255), thickness=-1) # thickness < 0 : fill
            if to_vis:
                cv.imwrite(os.path.join(savedir, f"{scenepath.split('/')[-1]}_newcontour.jpg"), new_contour_mask)
                cv.imwrite(os.path.join(savedir, f"{scenepath.split('/')[-1]}_room_mask.jpg"), scene_data["room_layout"])

        return mapped_corner, new_contour_mask


def fp_line_normal(fpoc):
    """ fpoc: [numpt, 2] np array, in scale [-3, 3]/[-6, 6], scene_data's floor_plan_ordered_corners.
        Returns normalized floor plan line normals.
    """
    fp_line_n = np.zeros((fpoc.shape[0], 2)) # for each line, get its normal. fpbp_normal[0] for fpoc[0] to fpoc[1]
    for i in range(fpoc.shape[0]):
        line_vec = fpoc[(i+1)%fpoc.shape[0]] - fpoc[i] # clockwise starting from bottom left
        line_len = np.linalg.norm(line_vec)   # possible for a line to have almost 0 len
        if line_len == 0: 
            print("! fp_line_normal: line_len==0!")
            continue
        fp_line_n[i, 0] = line_vec[1]/line_len   # dy for x axis
        fp_line_n[i, 1] = -line_vec[0]/line_len  # -dx for y axis (points inwards towards room center)
    return fp_line_n # normalized


def scene_sample_fpbp(tdf, fpoc, scenedir="", to_vis=False):
    """ fpoc: =scene_data["floor_plan_ordered_corners"]=output of process_floorplan_iterative_closest_point()
              [numpt, 2] np array, scaled in [-3,3] for bedroom or [-6,6] for living room (not normalized)

        returns floor plan boundary pt + normal: [nfpbp, 2+2] = [x, y, nx, ny], x, y in scale [-3/6,3/6], (nx, ny) is normalized. Same numpt for all scenes.
    """
    nfpbp = tdf.nfpbpn # if nfpbp is None else nfpbp

    x = np.append(fpoc[:,0], [fpoc[0,0]]) # (nline+1,), in [-3,3]
    y = np.append(fpoc[:,1], [fpoc[0,1]]) # append one extra to close the loop, 
    fp_line_n = fp_line_normal(fpoc) # numpt, 2

    # sample nfpbp points randomly from the contour outline:
    # Linear length on the line
    dist_bins = np.cumsum( np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ) ) # (nfpoc+1,) cumulative line seg len from bottom left pt
    dist_bins = dist_bins/dist_bins[-1] # (nfpoc+1,), [0, ..., 1] (values normalized to 0 to 1)

    fx, fy = interp1d(dist_bins, x), interp1d(dist_bins, y) # [0, 1] -> [-3/6, 3/6]

    seg_len = float(1)/nfpbp # total perimeter normalized to 1 (distance above)
    seg_starts = np.linspace(0, 1, nfpbp+1)[:-1] # (nfpbp,), starting point of each segment # [0.   0.25 0.5  0.75 1.  ][:-1]
    per_seg_displacement = np.random.uniform(low=0.0, high=seg_len, size=(nfpbp)) # one for each line segment
    sampled_distance = seg_starts + per_seg_displacement # (nfpbp=250, 1)
    sampled_x, sampled_y = fx(sampled_distance), fy(sampled_distance) # (nfpbp=250,), in [-3,3] (convert from 1d sampling to xy coord)

    fpbp = np.concatenate([np.expand_dims(sampled_x, axis=1), np.expand_dims(sampled_y, axis=1)], axis=1) #(nfpbp, 1+1=2)

    bin_idx = np.digitize(sampled_distance, dist_bins) # bins[inds[n]-1] <= x[n] < bins[inds[n]]
    bin_idx -= 1 # (nfpbp=250,) in range [0, nline-1] # example: [ 0  7 10 12 14 17 18 21 22 24] 
    fpbp_normal = fp_line_n[bin_idx, :] # fp_line_n: [nline, 2] -> fpbp_normal: [nfpbp, 2] 

    if to_vis:
        fig = plt.figure(dpi=300, figsize=(5,5))
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        R = (room_info["room_size"][tdf.room_type][0]/2)
        rang = [-R, R] #[-3,3] for bedroom
        plt.gca().set(xlim=rang, ylim=rang)

        plt.plot(x, y, 'o-', markersize=3) # line connecting circle
        for i in range(x.shape[0]):  plt.text(x[i]-0.2, y[i]+0.1, i, size=6) 
        for i in range(fp_line_n.shape[0]):  plt.quiver(x[i], y[i], fp_line_n[i,0], fp_line_n[i,1], width=0.005) # at the start pt
        plt.title(f"fpoc (blue); normal of each line at its starting point (black)\n{scenedir}", fontsize=8)
        plt.savefig(f"/Users/annawei/Tech/IVL/canonical-arrangement/_data_related/3tdfront/TDF_testscene/sample_fpbp/{scenedir}.jpg")
        
        plt.plot(sampled_x, sampled_y, 'or', markersize=3, alpha=0.7) # red circles
        plt.title(f"sample_fpbp: nfpbp = {nfpbp} (red); fpoc (blue)\n{scenedir}", fontsize=8)
        plt.savefig(f"/Users/annawei/Tech/IVL/canonical-arrangement/_data_related/3tdfront/TDF_testscene/sample_fpbp/{scenedir}_{nfpbp}.jpg")
        plt.close()

    return np.concatenate([fpbp, fpbp_normal], axis=1) # (nfpbp, 2+2=4)


def preprocess_floor_plan(tdf):
    """ Generates all 3 representations of floor plans from data in boxes npz and write them to boxes npz.
    """
    print("preprocess_floor_plan: start")
    counter = 0
    for e in os.listdir(tdf.scene_dir): # train, val, and test
        if not os.path.isdir(os.path.join(tdf.scene_dir, e)): continue

        counter += 1
        if counter % 500==0 : print(" ", counter)
        
        scene_data = dict(np.load(os.path.join(tdf.scene_dir, e, "boxes.npz")))
        mapped_corner, new_contour_mask = process_floorplan_iterative_closest_point(
                    tdf, scenepath = os.path.join(tdf.scene_dir, e), savedir = None, to_vis=False, to_save_new_contour=True)
        scene_data["floor_plan_ordered_corners"] = mapped_corner # not normalized
        scene_data["remapped_room_layout"] = new_contour_mask # (256,256,1)

        scene_fpbpn = scene_sample_fpbp(tdf, scene_data["floor_plan_ordered_corners"], scenedir=e, to_vis=False)
        scene_data["floor_plan_boundary_points_normals"] = scene_fpbpn # [nfpbp=250, 4], in [-3,3] + [-1,1] (unit circle)
        
        np.savez_compressed( os.path.join(tdf.scene_dir, e, "boxes"),  **scene_data)
    
    print("preprocess_floor_plan: done\n")


def write_all_data_summary_npz(tdf, trainval=False):
    """ preprocess_floor_plan must be called beforehand, as this funciton reads from the npz files it writes.
        
        Saves normalized data (ready for training) to data_tv/test_ctr(_livingroomonly).npz. Normalization code 
        same as _gen_3dfront_batch_onthefly.
        
        ctr saved as a compact representation of fpmask as the original fpmask takes too much space and time in 
        saving, loading, and copying. ctr is converted back into fpmask in _gen_3dfront_batch_preload.
    """
    print(f"write_all_data_summary_npz: trainval={trainval}")

    scenes, filename = tdf.scenes_tv if trainval else tdf.scenes_test, "data_tv_ctr" if trainval else "data_test_ctr"
    if tdf.livingroom_only: filename = f"{filename}_livingroomonly"

    batch_size = len(scenes) # all scenes
    print(" total:", batch_size)

    batch_scenedirs = [] # list of strings
    
    batch_nbj = []

    batch_pos = np.zeros((batch_size, tdf.maxnobj, tdf.pos_dim))
    batch_ang = np.zeros((batch_size, tdf.maxnobj, tdf.ang_dim))
    batch_siz = np.zeros((batch_size, tdf.maxnobj, tdf.siz_dim)) # shape
    batch_cla = np.zeros((batch_size, tdf.maxnobj, tdf.cla_dim)) # shape

    batch_vol = np.zeros((batch_size, tdf.maxnobj, 1)) # shape

    batch_fpoc = np.zeros((batch_size, tdf.maxnfpoc, tdf.pos_dim)) # Pad with other values? # floor plan ordered corners (neighboring pt form a line)
    batch_nfpc = []

    # batch_fpmask = np.zeros((batch_size, 256, 256, 3))  # 1 3-channel mask per scene
    batch_ctr = np.zeros((batch_size, 51, 1, 2))

    batch_fpbpn = np.zeros((batch_size, tdf.nfpbpn, 4))

    for i in range(batch_size):
        if i%500 == 0: print(" ", i)
        scenepath = scenes[i]
        scene_data = np.load(os.path.join(scenepath, "boxes.npz"), allow_pickle=True)

        batch_scenedirs.append(os.path.split(scenepath)[1])  # just the directory name

        nobj = scene_data['jids'].shape[0]
        batch_nbj.append(nobj)

        # normalizes data
        batch_pos[i,:nobj,0] = scene_data['translations'][:,0]/(tdf.room_size[0]/2) #[-3, 3] -> [-1,1]
        batch_pos[i,:nobj,1] = scene_data['translations'][:,2]/(tdf.room_size[2]/2) #[-3, 3] -> [-1,1] # use z as y
        batch_ang[i,:nobj,0:1] = np.cos(scene_data['angles']*-1) # since pos z (out of screen) -> neg y (vertical flip)
        batch_ang[i,:nobj,1:2] = np.sin(scene_data['angles']*-1)
        batch_siz[i,:nobj,0] = scene_data['sizes'][:,0]*2 /(tdf.room_size[0]/2)-1  #[0,3]*2 (only half of box len) -> [0,6]/3 -> [0,2]-1 -> [-1,1]
        batch_siz[i,:nobj,1] = scene_data['sizes'][:,2]*2 /(tdf.room_size[2]/2)-1  #[0,3]*2 (only half of box len) -> [0,6]/3 -> [0,2]-1 -> [-1,1]# use z as y
        batch_cla[i,:nobj,:] = scene_data['class_labels'][:,:tdf.cla_dim] # subsequent columns are 0 (start, end tokens)
        
        batch_vol[i,:nobj,0] = scene_data['sizes'][:,0] * scene_data['sizes'][:,1] * scene_data['sizes'][:,2] * 8 # absolute scale volume

        corners = scene_data["floor_plan_ordered_corners"] # (numpt, 2), in [-3, 3]
        batch_fpoc[i, :corners.shape[0], 0] = corners[:,0] /(tdf.room_size[0]/2) # scale to [-1, 1]
        batch_fpoc[i, :corners.shape[0], 1] = corners[:,1] /(tdf.room_size[2]/2) # scale to [-1, 1]
        batch_nfpc.append(corners.shape[0])

        ## If to save fpmask:
        # fpmask = scene_data["remapped_room_layout"]  #(256,256,1), binary occupancy
        # xy = generate_pixel_centers(256,256) / 128 -1 # [0 (0.5), 256 (255.5)] -> [0,2] -> [-1,1] # in the same coord system as vertices 
        # batch_fpmask[i] = np.concatenate([fpmask, xy], axis=2) #(256,256,1+2=3)
        R = (room_info["room_size"][tdf.room_type][0]/2) # 3 for bedroom
        ctr = np.expand_dims( (scene_data["floor_plan_ordered_corners"]+R) /(R*2) * 256, 1).astype(np.int32) # [numpt,1,3], .astype(numpy.int32), in [0,256]
        batch_ctr[i, :ctr.shape[0], :, :] = ctr

        batch_fpbpn[i,:,0:2] = scene_data["floor_plan_boundary_points_normals"][:,0:2] / (tdf.room_size[0]/2) # [nfpbpn, 2], [-3,3] -> [1,1] 
        batch_fpbpn[i,:,2:4] = scene_data["floor_plan_boundary_points_normals"][:,2:4] # [nfpbpn, 2], normals (already noramlized)


    np.savez_compressed( 
        os.path.join(tdf.scene_dir, filename), 
        # bedroom: trainval=5668, test=224 | livingroom: trainval=2338, test=587

        scenedirs = np.array(batch_scenedirs), # (5668,) # Ex: 00ecd5d3-d369-459f-8300-38fc159823dc_SecondBedroom-6249_0

        nbj = np.array(batch_nbj), # (5668,)

        pos = batch_pos, # (5668, 12, 2) numpy array 
        ang = batch_ang, # (5668, 12, 2)
        siz = batch_siz, # (5668, 12, 2)
        cla = batch_cla, # (5668, 12, 19)

        vol = batch_vol, # (5668, 12, 1)
        
        ## floor plan representation 1: floor plan ordered corners
        fpoc = batch_fpoc, # (5668, 25, 2)
        nfpc = np.array(batch_nfpc), # (5668,)
        
        ## floor plan representation 2: contour corner points (rescaled floor_plan_ordered_corners) (used to form floor plan binary mask when loaded)
        # fpmask = batch_fpmask # (5668, 256, 256, 3)
        ctr = batch_ctr, # (5668, 51, 1, 2)
        
        ## floor plan representation 3: floor plan boundary points & their normals
        fpbpn = batch_fpbpn # (5668, 250, 4)
    )
    print(f"write_all_data_summary_npz: trainval={trainval} - {filename} done\n")

    
def augment(tdf):
    """ preprocess_floor_plan must be called beforehand, as this funciton reads from the npz files it writes.
        Creates a new processed_<roomtype>_augmented directory containing 
        - dataset_stats_all.txt (not changed as object_types are unaffected by augmentation).
        - <jid_0-3>/boxes.npz: 0 corresponds to 0 degree counterclockwise, 1 to 90, 2 to 180, 3 to 270.
        - room_mask jpgs
        
        Augmentation: rotate by 90 degrees, affects pos, ang, siz, room_layout, floor_plan_ordered_corners, remapped_room_layout.
        The last 2 are calculated from floor_plan_vertices and floor_plan_centroid in boxes.npz, which are not changed as they 
        are not directly used (other than in preprocessing).
    """
    print(f"augment: start")
    new_scene_dir = os.path.join (os.path.split(tdf.scene_dir)[0], f"processed_{tdf.room_type}_augmented")

    if not os.path.exists(new_scene_dir): os.makedirs(new_scene_dir)
    os.system(f"cp {os.path.join(tdf.scene_dir, 'dataset_stats_all.txt')} {os.path.join(new_scene_dir, 'dataset_stats_all.txt')}")  

    count = 0
    for d in list(os.listdir(tdf.scene_dir)):
        if not os.path.isdir(os.path.join(tdf.scene_dir, d)): continue
        count += 1
        if count % 500 == 0: print(" ", count)
        scene_data = dict(np.load(os.path.join(tdf.scene_dir, d, "boxes.npz"), allow_pickle=True))

        nobj = scene_data['jids'].shape[0]
        for aug_i in range(4):
            new_scene_data = copy.deepcopy(scene_data) # not normalized (kept consistent with ATISS boxes files)
            new_id_dir = f"{d}_{aug_i}"
            new_savedir = os.path.join(new_scene_dir, new_id_dir)
            if not os.path.exists(new_savedir): os.makedirs(new_savedir)
            # print(new_savedir)

            new_scene_pos = np_rotate(new_scene_data['translations'][:, [0,2]] , np.zeros((nobj,1))+np.pi/2*aug_i) # [nobj, 2=pos_dim] in [-3, 3], rotate by origin
            new_scene_data['translations'][:,0] = new_scene_pos[:,0]
            new_scene_data['translations'][:,2] = new_scene_pos[:,1] # y in translation is unchanged

            new_scene_data['angles'] = (new_scene_data['angles'] - (np.pi/2*aug_i)) % (2*np.pi)  # numpt, 1 (each a theta)
                # -(-x+90) = x-90  NOTE: no need to swap x and y/z for sizes/bbox length as angles are changed

            # Loaded files should be preprocessed, so contain the following fields prepared in preprocess_floor_plan()
            new_scene_data['floor_plan_ordered_corners'] = np_rotate(new_scene_data["floor_plan_ordered_corners"], 
                                                           np.zeros((new_scene_data["floor_plan_ordered_corners"].shape[0],1))+np.pi/2*aug_i) # [-3,3], rot by origin
            
            new_scene_data["room_layout"] = np.rot90(new_scene_data["room_layout"], 4-aug_i)
            new_scene_data["remapped_room_layout"] = np.rot90(new_scene_data["remapped_room_layout"], 4-aug_i)  # (256,256,1)
            # NOTE: aug_i=1, 90 deg -> 270=-90 deg because we do a vertical flipping when visualizing floor maps (-90*-1 = 90)
            # NOTE: corners are upright, room_layout is actually upside down flipped
            
            new_scene_data["floor_plan_boundary_points_normals"][:,0:2] = np_rotate(new_scene_data["floor_plan_boundary_points_normals"][:,0:2],
                                                                          np.zeros((new_scene_data["floor_plan_boundary_points_normals"].shape[0],1))+np.pi/2*aug_i) # [-3,3], rot by origin
            new_scene_data["floor_plan_boundary_points_normals"][:,2:4] = np_rotate(new_scene_data["floor_plan_boundary_points_normals"][:,2:4],
                                                                          np.zeros((new_scene_data["floor_plan_boundary_points_normals"].shape[0],1))+np.pi/2*aug_i) # [-1,1](unit circle), rot by origin

            cv.imwrite(os.path.join(new_savedir, "room_mask_orig.jpg"), scene_data["room_layout"]) # unrotated
            cv.imwrite(os.path.join(new_savedir, "room_mask.jpg"), new_scene_data["room_layout"])
            cv.imwrite(os.path.join(new_savedir, "room_mask_remapped.jpg"), new_scene_data["remapped_room_layout"])

            np.savez_compressed( os.path.join(new_savedir, "boxes"),  **new_scene_data)

            # CHECK
            if False:
                input, scenepath = tdf.read_one_scene(new_savedir) 
                tdf.visualize_tdf_2d(input, 
                    f"/Users/annawei/Tech/IVL/canonical-arrangement/_data_related/3tdfront/TDF_testscene/augment/{new_id_dir}.jpg",
                    f"Augmented", traj=None, scenepath=new_savedir, show_corner=False, show_fpbpn=True)

    print(f"augment: done\n")




if __name__ == '__main__':
    exit()

    ### PREPROCESS
    # tdf=TDFDataset("livingroom", use_augment=False, livingroom_only=False)
    
    ## 1. Write floor_plan_ordered_corners, remapped_room_layout (part of fpmask), floor_plan_boundary_points_normals to same scene boxes npz files
    # preprocess_floor_plan(tdf) 
    # print("------------------------")

    ## 2. Normalize data and create summary data_tv/test_ctr(_livingroomonly).npz, 1 for each call
    # write_all_data_summary_npz(tdf, trainval=True)
    # write_all_data_summary_npz(tdf, trainval=False)



    ### AUGMENT
    #  Creates a new processed_<roomtype>_augmented directory containing:

    ## 1. dataset_stats_all.txt, <jid_0-3>/boxes.npz, room_mask.png
    # tdf=TDFDataset("livingroom", use_augment=False, livingroom_only=False)
    # preprocess_floor_plan(tdf) 
    # print("------------------------")
    # augment(tdf) # assume livingroom directory already has floor plan processed
    # print("------------------------")

    ## 2. data_tv/test_ctr(_livingroomonly).npz
    # tdf=TDFDataset("livingroom", use_augment=True, livingroom_only=False)
    # write_all_data_summary_npz(tdf, trainval=True)
    # write_all_data_summary_npz(tdf, trainval=False)

