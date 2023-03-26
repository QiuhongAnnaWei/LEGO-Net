import os, sys
sys.path.insert(1, os.getcwd())

import math
import random
import datetime
import argparse

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from data.TDFront import TDFDataset
from filepath import eval_dir


from mpmath import *
mp.dps = 15; mp.pretty = True
    # >>> pslq([0.35, 0.24, -1.6], tol=0.001)
    # [8, -5, 1]
    # >>> pslq([0.35, 0.24], tol=0.001)
    # [-24, 35]
    # >>> pslq([-0.2, 0.2], tol=0.001)
    # [1, 1]
    # >>> pslq([-0.2, 0.2, 0.5], tol=0.001)
    # [1, 1, 0]


def log(line, filepath=None, toPrint=True):
    if filepath is None: filepath=args['logfile']
    if args['log']:
        with open(filepath, 'a+') as f:
            f.write(f"{line}\n")
    if toPrint:
        print(line)


def pslqres_isvalid(res, inputele, tocrosscheck=False, dimtocheck=0):
    """ tocrosscheck: needed because it may not be relevant depending on the context.
        Checks whether the pslq results counts as a valid integer relation, returns 0 or 1. """
    if (res is None) or (np.count_nonzero(res)!=len(inputele)): return 0 # must use all ele in set, otheriwse counted by other subsets
    
    if tocrosscheck:
        if (abs(np.dot(res, inputele[:, 1-dimtocheck])) > crosscheck_thres): return 0

    return 1


def pslq_core(ele_in_aset):
    """ ele_in_aset: [neleinaset, pos+ang]
        returns number of linear relations found, number of trials made (taking #dimtocheck into account)
    """
    if ele_in_aset.shape[0] < min_nobj or ele_in_aset.shape[0] > max_nobj: 
        return 0,0 # bounds are inclusive (equality included)
    if np.min(np.abs(ele_in_aset[:,0]))==0 and np.min(np.abs(ele_in_aset[:,1]))==0: 
        return 0,0 # PSLQ requires a vector of nonzero numbers
    
    n_relation_satisfied, trial_ct = 0, 0 # trial_ct will be > 0
    for dimtocheck in dimstocheck:
        if np.min(np.abs(ele_in_aset[:,dimtocheck]))==0: continue
        trial_ct += 1
        if robust == "none":
            res = pslq(ele_in_aset[:,dimtocheck].tolist(), maxcoeff=maxcoeff, tol=tol)  # pass in 1d arr of len=nsetele, res=coefficients ([8, -5, 1])
            n_relation_satisfied += pslqres_isvalid(res, ele_in_aset, tocrosscheck=crosscheckdims, dimtocheck=dimtocheck)
        elif robust=="coeffpslq": # new
            res = pslq(ele_in_aset[:,dimtocheck].tolist(), maxcoeff=maxcoeff, tol=tol)
            if pslqres_isvalid(res, ele_in_aset, tocrosscheck=crosscheckdims, dimtocheck=dimtocheck):
                coeff_res = pslq(res, maxcoeff=2, tol=0.0001)
                coeff_valid = pslqres_isvalid(coeff_res, res, tocrosscheck=False) # res is 1d array, no other dimension to cross check
                n_relation_satisfied+=coeff_valid
            # if coeff_valid==0:
            #     print(f"robust fail: primary coefficients = {res}")
            # else:
            #     print(f"robust success: primary coefficients = {res}")
        elif robust == "translation": # old
            robust_pass_ct = 0
            for _ in range(total_pass_ct):
                res = pslq((ele_in_aset[:,dimtocheck]+random.uniform(-1, 1)).tolist(), maxcoeff=maxcoeff, tol=tol) # random translation
                if pslqres_isvalid(res, ele_in_aset, tocrosscheck=crosscheckdims, dimtocheck=dimtocheck): robust_pass_ct+=1
            if robust_pass_ct>=thres_ct: n_relation_satisfied += 1 # need to pass all translation trials

    return n_relation_satisfied, trial_ct


def scene_linear_relations_sample(posang, nobj, n_sampledsubset=100, while_ct_break=500000):
    """ pos: [nobj, pos+ang]
        Returns number of linear relations found, number of trials made.

        Looks at n_sampledsubset number of subsets.
    """
    # nobj, cla_idx = TDFDataset.parse_cla(traj[0,:,P+A+S:]) # uses initial cla for all snapshot in traj (generated input, never perturbed)
    subsets = np.array(range(1, int(math.pow(2, int(nobj))) )) # for each scene, examine the power set of its objects

    n_relation_satisfied, trial_ct, subset_sample_ct, while_ct = 0, 0, 0, 0
    while subset_sample_ct < min(n_sampledsubset, len(subsets)): # adjusted for at the end
        while_ct+=1
        if while_ct>while_ct_break: 
            print("had to break because of while_ct")
            break

        aset = random.choice(subsets)

        ele_in_aset = np.array([posang[o_i] for o_i in range(nobj) if (aset & (1 << o_i)) > 0 ]) # [neleinset, pos+ang]
        n, ct = pslq_core(ele_in_aset)
        n_relation_satisfied += n
        trial_ct += ct
        if ct>0: subset_sample_ct += 1 # different from trial_ct in that one subset can only contribute at max 1

    return n_relation_satisfied, trial_ct


def scene_linear_relations_all(posang, nobj):
    """ pos: [nobj, pos+ang]
        Returns number of linear relations found, number of trials made.

        Looks at all the subsets in the scene. 
    """
    # nobj, cla_idx = TDFDataset.parse_cla(traj[0,:,P+A+S:]) # uses initial cla for all snapshot in traj (generated input, never perturbed)
    subsets = np.array(range(1, int(math.pow(2, int(nobj))) )) # for each scene, examine the power set of its objects
    
    n_relation_satisfied, trial_ct = 0, 0
    for aset in subsets:
        ele_in_aset = np.array([posang[o_i] for o_i in range(nobj) if (aset & (1 << o_i)) > 0 ]) # [neleinset, pos+ang]
        n, ct = pslq_core(ele_in_aset)
        n_relation_satisfied += n
        trial_ct += ct
        
    return n_relation_satisfied, trial_ct #< (len(subsets)*2)


def scene_linear_relations_neighbor(posang, nobj, nneighbor=6, groupsize=3):
    """ pos: [nobj, pos+ang]
        Returns number of linear relations found, number of trials made.
    
        For each object in a scene:
        1. Select top few closest objects.
        2. Among the top nneighbor objects, randomly select objects to form a object set of groupsize. When nneighbor == groupsize-1, 
           the object set formed consists of the cloest objects.
        3. Find integer relation and report the ratio of integer relations found.
    """
    if groupsize-1 > min(nneighbor, nobj-1):
        print(f"groupsize-1 ({groupsize-1}) must be <= min(nneighbor, nobj-1) (nneighbor={nneighbor}, nobj-1={nobj-1})")
        return 0, 0

    n_relation_satisfied, trial_ct = 0, 0
    for o_i in range(nobj):
        oi_dists = np.linalg.norm(posang[o_i,:2]-posang[:,:2], axis=1) # [n, 2]
        sorted_idx = np.argsort(oi_dists) # ex: array([3, 1, 2, 0]) for [40, 20, 30, 10]
        sorted_idx = np.delete(sorted_idx, np.argwhere(sorted_idx==o_i)) #[nobj-1,]
       
        if len(sorted_idx) > nneighbor: sorted_idx = sorted_idx[:nneighbor] # only consider the few closest 
        neighbor_i = np.random.choice(sorted_idx, size=groupsize-1, replace=False) # cannot repeat
        ele_in_aset = np.array([posang[i] for i in range(nobj) if (i == o_i or i in neighbor_i) ]) # [groupsize, pos+ang]

        n, ct = pslq_core(ele_in_aset)
        n_relation_satisfied += n
        trial_ct += ct
        
    return n_relation_satisfied, trial_ct




## ENTRY POINT API

def evaluate_gen_data(method="all", numscene=2000, n_sampledsubset=200, while_ct_break=500000, nneighbor=6, groupsize=3, to_vis=False):
    """ Evaluate linear relations of 3DFRONT secens.
        n_sampledsubset: if None, check on all subsets; otherwise, check on the given number of subsets (random order)
        5892 total bedroom
    """
    log("\n(evaluate_gen_data)")
    noise_level_maxes = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    angle_noise_level_maxes = [np.pi/90, np.pi/24, np.pi/12, np.pi/4, np.pi/3, np.pi/2, np.pi]

    all_scene_n_relations, all_trial_cts, all_satisfaction_ratios = [], [], [] # [n_nl, nscene]
    nl_mean_satisfaction_ratios = [] # [n_nl]

    random_idx = tdf.gen_random_selection(numscene, data_partition="all") # same set
    log(f"random_idx = {random_idx}")
    for noise_i in range(len(noise_level_maxes)):
        nl, anl = noise_level_maxes[noise_i], angle_noise_level_maxes[noise_i]
        scene_n_relations, trial_cts, satisfaction_ratios = [], [], [] #[nscene,] # for one noise level (trial ct might be constant since the scenes are fixed)
        input, labels, _, scenepaths, _, _, _, _ = tdf.gen_3dfront(numscene, random_idx=random_idx, data_partition='all', noise_level_stddev=nl, angle_noise_level_stddev=anl,
                                                                    weigh_by_class = False, within_floorplan = False, no_penetration = False)                                                         
        
        for scene_i in range(numscene): # [batch_size, maxnumobj, pos+ang+siz+cla]
            scene_nobj, cla_idx = TDFDataset.parse_cla(input[scene_i,:,tdf.pos_dim+tdf.ang_dim+tdf.siz_dim:]) # uses initial cla for all snapshot in traj (generated input, never perturbed)
            if method=="all": # n_sampledsubset is None:
                n_relation_satisfied, trial_ct = scene_linear_relations_all(input[scene_i, :scene_nobj, :tdf.pos_dim+tdf.ang_dim], scene_nobj)
            elif method=="sample":
                n_relation_satisfied, trial_ct = scene_linear_relations_sample(input[scene_i, :scene_nobj, :tdf.pos_dim+tdf.ang_dim], scene_nobj, n_sampledsubset=n_sampledsubset, while_ct_break=while_ct_break)
            elif method=="neighbor":
                n_relation_satisfied, trial_ct = scene_linear_relations_neighbor(input[scene_i, :scene_nobj, :tdf.pos_dim+tdf.ang_dim], scene_nobj, nneighbor=nneighbor, groupsize=groupsize)
            scene_n_relations.append(n_relation_satisfied)
            trial_cts.append(trial_ct)

            if trial_ct==0:
                print("evaluate_gen_data: trial_ct is 0")
            else:
                satisfaction_ratios.append(n_relation_satisfied/trial_ct)

            if to_vis:
                input, scenepath = tdf.read_one_scene(scenepath=os.path.split(scenepaths[scene_i])[1]) 
                id = scenepath.split('/')[-1]
                tdf.visualize_tdf_2d(input, os.path.join(args['logsavedir'], f"{id}.jpg"), f"", traj=None, scenepath=scenepath, show_corner=False)
            
        scene_n_relations, trial_cts, satisfaction_ratios = np.array(scene_n_relations), np.array(trial_cts), np.array(satisfaction_ratios)
        all_scene_n_relations.append(scene_n_relations)
        all_trial_cts.append(trial_cts)
        all_satisfaction_ratios.append(satisfaction_ratios)
        nl_mean_satisfaction_ratios.append(np.mean(satisfaction_ratios)) # noise level

        log(f"nl={nl}, anl={round(anl/np.pi*180, 3)}: mean satisfaction ratio={round(np.mean(satisfaction_ratios),5)} (average scene: {np.mean(scene_n_relations)} relations satisfied out of {np.mean(trial_cts)} trials")
    
    
    all_scene_n_relations, all_trial_cts, all_satisfaction_ratios = np.array(all_scene_n_relations), np.array(all_trial_cts), np.array(all_satisfaction_ratios) # [n_nl, nscene]
    nl_mean_satisfaction_ratios = np.array(nl_mean_satisfaction_ratios) # [nl,]
    fname = "tdfgen"

    with open(os.path.join(args['logsavedir'], f"{fname}-{method}-scene_n_relations.txt"), 'w') as f: f.write(str(all_scene_n_relations))
    with open(os.path.join(args['logsavedir'], f"{fname}-{method}-trial_cts.txt"), 'w') as f: f.write(str(all_trial_cts))
    with open(os.path.join(args['logsavedir'], f"{fname}-{method}-satisfaction_ratios.txt"), 'w') as f: f.write(str(all_satisfaction_ratios))
    with open(os.path.join(args['logsavedir'], f"{fname}-{method}-nl_mean_satisfaction_ratios.txt"), 'w') as f: f.write(str(nl_mean_satisfaction_ratios))


    fig = plt.figure(dpi=300, figsize=(7,5))
    plt.plot(noise_level_maxes, np.mean(all_satisfaction_ratios, axis=1))  # [n_nl,]  (one average per noise level across all the scenes)
    # plt.gca().set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2])
    plt.title(f"Mean Satiscation Ratio of Linear Relations vs. Noise Level Standard Deviation ({args['room_type']}: {numscene} scenes)", fontsize=7)
    plt.ylabel("Mean Per-Scene Percentage of Linear Relations Satisfied", fontsize=7)
    plt.xlabel("Standard Deviation of Noise Level Distribution", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(os.path.join(args['logsavedir'], f"{fname}_satisfactionratios_2_noiselevel.jpg"))
    plt.close(fig)

    fig = plt.figure(dpi=300, figsize=(7,5))
    plt.plot(noise_level_maxes, np.mean(all_trial_cts, axis=1), label="Mean number of trials (dimensions included)")
    plt.plot(noise_level_maxes, np.mean(all_scene_n_relations, axis=1), label="Mean number of relations satisfied")  
    plt.legend(fontsize=7)
    plt.title(f"Mean Number of Linear Relations Staisfied and Tried vs. Noise Level Standard Deviation ({args['room_type']}: {numscene} scenes)", fontsize=7)
    plt.ylabel("Mean Per-Scene Number of Linear Relations", fontsize=7)
    plt.xlabel("Standard Deviation of Noise Level Distribution", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(os.path.join(args['logsavedir'], f"{fname}_nrelations_2_noiselevel.jpg"))
    plt.close(fig)
    # colors = list(plt.cm.rainbow(np.linspace(0, 1, len(noise_level_maxes))))
    # for i in range(all_scene_n_relations.shape[0]):
    #     plt.scatter(np.zeros(all_scene_n_relations[i].shape[0])+noise_level_maxes[i], all_scene_n_relations[i], color = [colors[i]], 
    #                 s=10, label=f"Noise: {noise_level_maxes[i]}, {round(angle_noise_level_maxes[noise_i]/np.pi*180, 1)}")



def evaluate_denoise_results(fp, denoise_method="grad_noise", method="all", n_sampledsubset=200, while_ct_break=500000, nneighbor=6, groupsize=3):
    """ Evaluate final denoised scenes stored in npz files saved from denoise_meta() in train.py. 
        denoise method: one of {direct_map_once, direct_map, grad_nonoise, grad_noise, ATISS_label}
    """
    P, A, S, C = tdf.pos_dim, tdf.ang_dim, tdf.siz_dim, tdf.cla_dim
    data = np.load(fp, allow_pickle=True)

    scene_n_relations, trial_cts, satisfaction_ratios = [], [], [] #[nscene,] # for one noise level (trial ct might be constant since the scenes are fixed)
    log(f"\n(evaluate_denoise_results)\n{denoise_method}: {data[f'{denoise_method}_trajs'].shape[0]} scenes in total")
    for scene_i in range(data[f"{denoise_method}_trajs"].shape[0]): # [batch_size, maxnumobj, pos+ang+siz+cla], numscene=data["scenepaths"].shape
        if scene_i %100 == 0: print("scene ", scene_i)
        final = data[f"{denoise_method}_trajs"][scene_i][-1] # (maxnobj, pos+ang+siz+cla) <- from (numscene=500, 2/variable niter, 12, 25) (roughly)
        if denoise_method=="ATISS_label": final = data[f"{denoise_method}_trajs"][scene_i] # ATISS: 500 scenes, only final scene saved (numscene=500, 12, 25) (roughly)
        
        scene_nobj, cla_idx = TDFDataset.parse_cla(final[:,P+A+S:]) 
        if method=="all": # n_sampledsubset is None:
            n_relation_satisfied, trial_ct = scene_linear_relations_all(final[:scene_nobj, :P+A], scene_nobj)
        elif method=="sample":
            n_relation_satisfied, trial_ct = scene_linear_relations_sample(final[:scene_nobj, :P+A], scene_nobj, n_sampledsubset=n_sampledsubset, while_ct_break=while_ct_break)
        elif method=="neighbor":
            n_relation_satisfied, trial_ct = scene_linear_relations_neighbor(final[:scene_nobj, :P+A], scene_nobj, nneighbor=nneighbor, groupsize=groupsize)
        scene_n_relations.append(n_relation_satisfied)
        trial_cts.append(trial_ct)

        if trial_ct==0:
            print("evaluate_denoise_results: trial_ct is 0")
        else:
            satisfaction_ratios.append(n_relation_satisfied/trial_ct)
        
    scene_n_relations, trial_cts, satisfaction_ratios = np.array(scene_n_relations), np.array(trial_cts), np.array(satisfaction_ratios)
    log(f"fp={fp}:\n\tmean satisfaction ratio={round(np.mean(satisfaction_ratios),7)} (average scene: {np.mean(scene_n_relations)} relations satisfied out of {np.mean(trial_cts)} trials")

    fname = os.path.split(os.path.split(fp)[0])[-1] # example: bedroom_pointnet_simple
    with open(os.path.join(args['logsavedir'], f"{fname}-{denoise_method}-{method}-scene_n_relations.txt"), 'w') as f: f.write(str(scene_n_relations))
    with open(os.path.join(args['logsavedir'], f"{fname}-{denoise_method}-{method}-trial_cts.txt"), 'w') as f: f.write(str(trial_cts))
    with open(os.path.join(args['logsavedir'], f"{fname}-{denoise_method}-{method}-satisfaction_ratios.txt"), 'w') as f: f.write(str(satisfaction_ratios))





def parse_parameters():
    if method=="neighbor" and (groupsize<min_nobj or groupsize>max_nobj):
        print(f"ERROR: method is neighbor, but min_nobj ({min_nobj}) != max_nobj ({max_nobj})")
        exit()


    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]
    args['logsavedir'] = os.path.join(eval_dir, "integer_relations", 
                                      f"{timestamp}_{args['room_type']}_maxcoeff{maxcoeff}_tol{tol}_obj{min_nobj}-{max_nobj}_{method}nneigh{nneighbor}_rob{robust}{thres_ct}{total_pass_ct}")
    args['logfile'] = os.path.join(args['logsavedir'], "log.txt")
    if not os.path.exists(args['logsavedir']): os.makedirs( args['logsavedir'])


    pprint(args)
    pprint(args, stream=open(args['logfile'], 'w'))

    log(f"")
    log(f"maxcoeff={maxcoeff}, tol={tol}, min_nobj={min_nobj}, max_nobj={max_nobj}")
    log(f"dimstocheck={dimstocheck}, crosscheckdims={crosscheckdims}, crosscheck_thres={crosscheck_thres}")
    log(f"numscene={numscene}")
    log(f"robust={robust}: thres_ct={thres_ct}, total_pass_ct={total_pass_ct}")
    log(f"method={method}: n_sampledsubset={n_sampledsubset}, while_ct_break={while_ct_break} | nneighbor={nneighbor}, groupsize={groupsize}")




maxcoeff, tol, min_nobj, max_nobj = 5, 0.01, 3, 3  # max coeff is non-inclusive, nobj are inclusive (used only in core)
dimstocheck, crosscheckdims, crosscheck_thres = [0,1], False, 0.001  # used directly in isvalid and core
numscene = 500

robust = "translation" # none, coeffpslq, translation (preferred)
thres_ct, total_pass_ct = 10, 10 # translation

method="neighbor" # all, sample, neighbor (preferred) (more description in specific functions)
n_sampledsubset, while_ct_break = 100, 10000 # sample
nneighbor, groupsize = 4, 3 # neighbor: nneighbor must be >= groupsize-1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--log", type = int, default=1, help="If 1, write to log file in addition to printing to stdout.")
    
    parser.add_argument("--res_filepath", type = str, help="filepath of npz file saved from denoise_meta() in train.py.")
    # ['random_idx', 'scenepaths', 'noise_level', 'direct_map_once_trajs', 'direct_map_once_perobj_distmoveds', 'direct_map_trajs', 'direct_map_perobj_distmoveds',
    #  'grad_nonoise_trajs', 'grad_nonoise_perobj_distmoveds', 'grad_noise_trajs', 'grad_noise_perobj_distmoveds']
    parser.add_argument("--room_type", type = str, default="livingroom", choices = ["bedroom", "livingroom"], help="3D-FRONT specific." )
    parser.add_argument("--denoise_method", type = str, default="grad_noise", choices = ["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"],
                        help="which inference method to evaluate.")
    args = vars(parser.parse_args()) #'dict' 


    parse_parameters()


    tdf = TDFDataset(args['room_type'], use_augment=False)
    evaluate_gen_data(method=method, numscene=numscene, n_sampledsubset=n_sampledsubset, while_ct_break=while_ct_break, nneighbor=nneighbor, groupsize=groupsize, to_vis=False)
    evaluate_denoise_results(args['res_filepath'], denoise_method=args['denoise_method'],method=method, n_sampledsubset=n_sampledsubset, while_ct_break=while_ct_break, nneighbor=nneighbor, groupsize=groupsize)

    log("\n### DONE ###")
    