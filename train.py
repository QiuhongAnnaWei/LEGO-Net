import os
import datetime
import argparse
import math
import glob

from pprint import pprint

import torch 
import matplotlib.pyplot as plt
import numpy as np

from model.models import PointNetSeg, PointNetPlusPlus, PointNetPlusPlus_dense, PointNetPlusPlus_dense_attention, PointTransformer, TransformerWrapper

from data.utils import *
from data.tablechair_horizontal import gen_data_tablechair_horizontal_bimodal
from data.tablechair_circle import gen_data_tablechair_circle_bimodal, ntable, nchair_min, nchair_max, maxnumobj, gen_no_table
from data.tablechair_shape import gen_data_tablechair_shape_bimodal
from data.TDFront import TDFDataset

from eval.denoise_res_eval import dist_2_gt


# denoising params
denoise_method_info = {
    # direct_map, add_noise
    "direct_map_once": [True, None],
    "direct_map": [True, None],
    "grad_nonoise": [False, False],
    "grad_noise": [False, True]
}
denoise_methods = list(denoise_method_info.keys())

# see more in adjust_parameters():
## loss parameter
L1_coeff = None
## denoise parameter
max_iter, step_size0, step_decay, noise_scale0, noise_decay, noise_drop_freq, pos_disp_break, ang_disp_pi_break, conse_break_meets_max = None, None, None, None, None, None, None, None, None
## data generation parameter (for when intersection is not allowed)
pen_siz_scale=0.92 # how much intersection is allowed in the input to denoise



def log(line, filepath=None, toPrint=True):
    if filepath is None: filepath=args['logfile']
    if args['log']:
        with open(filepath, 'a+') as f:
            f.write(f"{line}\n")
    if toPrint:
        print(line)

def load_checkpoint(model, model_fp):
    # Referencing global variables defined in main
    global optimizer
    global start_epoch

    checkpoint = torch.load(model_fp, map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    model = model.to(args['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    log("\n=> loaded checkpoint '{}' (epoch {})" .format(model_fp, checkpoint['epoch']))
    return model


def loss_fn(inp,tar):
    """ feat1, feat2: [batch_size, 6 or 14, 2] """
    # if args['predict_absolute_pos']: 
    MSE_loss = torch.nn.MSELoss()
    L1_loss  = torch.nn.L1Loss()

    if args['loss_func']=="MSE": 
        return MSE_loss(inp, tar)
    elif args['loss_func'] == 'MSE+L1':
        return MSE_loss(inp, tar) + L1_coeff * L1_loss(inp, tar)



def train(epoch, half_batch_numscene, args, data_type="3dfront", device="cpu"):
    """Variables defined in main"""
    log(f"\n\n------------BEGINNING TRAINING [Time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}] ------------")
    
    train_losses, val_losses, val_losses_epoch, min_val_loss = [], [], [], math.inf
    for e in range(start_epoch, start_epoch+epoch): # epoch = batch
        model.train()

        padding_mask, fpoc, nfpc, fpmask, fpbpn = None, None, None, None, None # variable-length data, 3dfront specific
        if data_type=="tablechair_horizontal":
            input, labels = gen_data_tablechair_horizontal_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'], 
                                                                    abs_pos=args["predict_absolute_pos"], abs_ang=args['predict_absolute_ang']) # deafult 0.25 and np.pi/4  # [half_batch_nscene, 14, 8]
        elif data_type=="tablechair_circle":
            input, labels = gen_data_tablechair_circle_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev']) # [half_batch_nscene, 14, 8]
        elif data_type =="tablechair_shape":
            input, labels = gen_data_tablechair_shape_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev']) # deafult 0.25 and np.pi/4
        elif data_type=="3dfront":
            input, labels, padding_mask, _, fpoc, nfpc, fpmask, fpbpn = tdf.gen_3dfront(half_batch_numscene*2, data_partition='trainval', use_emd=args['use_emd'], 
                                                                                        abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], use_floorplan=args["use_floorplan"],
                                                                                        noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'], pen_siz_scale=pen_siz_scale,
                                                                                        weigh_by_class = args['train_weigh_by_class'], within_floorplan = args['train_within_floorplan'], no_penetration = args['train_no_penetration'])
            padding_mask = torch.tensor(padding_mask).to(device) # boolean 
            if args['use_floorplan']:
                if args['floorplan_encoder_type'] == "pointnet":  fpoc, nfpc, fpmask, fpbpn = torch.tensor(fpoc).float().to(device), torch.tensor(nfpc).int().to(device), None, None
                elif args['floorplan_encoder_type'] == "resnet":  fpoc, nfpc, fpmask, fpbpn = None, None, torch.tensor(fpmask).float().to(device), None
                elif args['floorplan_encoder_type'] == "pointnet_simple": fpoc, nfpc, fpmask, fpbpn = None, None, None, torch.tensor(fpbpn).float().to(device)

        input, labels = torch.tensor(input).float().to(device), torch.tensor(labels).float().to(device)

        if args['model_type'] =="Transformer":
            pred = model(input, padding_mask, device, fpoc=fpoc, nfpc=nfpc, fpmask=fpmask, fpbpn=fpbpn)  # padding_mask: [batch_size, seq_len=maxnumobj]
        else:
            pred = model(input) # B, N, dim
        loss = loss_fn(pred, labels)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation + save best-so-far checkpoint
        if e%250 == 0 or e==start_epoch+epoch-1:
            with torch.no_grad():
                model.eval()
                val_padding_mask, val_fpoc, val_nfpc, val_fpmask, val_fpbpn = None, None, None, None, None # variable-length data, 3dfront specific
                if data_type=="tablechair_horizontal":
                    val_input, val_labels = gen_data_tablechair_horizontal_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'],
                                                                                   abs_pos=args["predict_absolute_pos"], abs_ang=args['predict_absolute_ang']) # deafult 0.25 and np.pi/4  # [half_batch_nscene, 14, 8]) 
                elif data_type=="tablechair_circle":
                    val_input, val_labels = gen_data_tablechair_circle_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev']) # deafult 0.25 and np.pi/4              
                elif data_type =="tablechair_shape":
                    val_input, val_labels = gen_data_tablechair_shape_bimodal(half_batch_numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev']) # deafult 0.25 and np.pi/4
                elif data_type=="3dfront":
                    val_input, val_labels, val_padding_mask, _, val_fpoc, val_nfpc, val_fpmask, val_fpbpn = tdf.gen_3dfront(half_batch_numscene*2, data_partition='test', use_emd=args['use_emd'], 
                                                                                                                            abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], use_floorplan=args["use_floorplan"],
                                                                                                                            noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'],  pen_siz_scale=pen_siz_scale,
                                                                                                                            weigh_by_class = args['train_weigh_by_class'], within_floorplan = args['train_within_floorplan'], no_penetration = args['train_no_penetration'])
                    val_padding_mask = torch.tensor(val_padding_mask).to(device) # boolean
                    if args["use_floorplan"]: 
                        if args['floorplan_encoder_type'] == "pointnet":  val_fpoc, val_nfpc, val_fpmask, val_fpbpn = torch.tensor(val_fpoc).float().to(device), torch.tensor(val_nfpc).int().to(device), None, None
                        elif args['floorplan_encoder_type'] == "resnet":  val_fpoc, val_nfpc, val_fpmask, val_fpbpn = None, None, torch.tensor(val_fpmask).float().to(device), None
                        elif args['floorplan_encoder_type'] == "pointnet_simple": val_fpoc, val_nfpc, val_fpmask, val_fpbpn = None, None, None, torch.tensor(val_fpbpn).float().to(device)

                val_input, val_labels = torch.tensor(val_input).float().to(device), torch.tensor(val_labels).float().to(device)
                
                if args['model_type'] == "Transformer":
                    val_pred = model(val_input, val_padding_mask, device, fpoc=val_fpoc, nfpc=val_nfpc, fpmask=val_fpmask, fpbpn=val_fpbpn)  # val_padding_mask: [batch_size, seq_len=maxnumobj]
                else:
                    val_pred = model(val_input) # B, N, dim
                val_loss = loss_fn(val_pred, val_labels)
                val_losses.append(val_loss.item())
                val_losses_epoch.append(e)
                log(f"Epoch {e}: train loss = {loss.item()}, val loss = {val_loss.item()}")

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    for old_best_checkpoint in glob.glob(os.path.join(args["logsavedir"], "best*")):  os.remove(old_best_checkpoint) 
                    state = { 'epoch': e + 1, 
                              'model_state_dict': model.state_dict(), 
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': loss }
                    minvalloss_model_fp = os.path.join(args["logsavedir"], (f"best_{e}iter_valloss" + "{0:.6f}.pt".format(val_loss.item())) )
                    torch.save(state, minvalloss_model_fp)
                    log(f"          Current best: saved to {minvalloss_model_fp}")
            
        # plot loss
        if (e>0 and e%1000 == 0) or e==start_epoch+epoch-1:
            fig = plt.figure(figsize=(10, 6))
            train, = plt.plot(list(range(len(train_losses))), train_losses, label = "Training Loss")
            val, = plt.plot( val_losses_epoch, val_losses, label = "Validation Loss")
            plt.figlegend(handles=[train, val], loc='upper right')
            plt.gca().set(xlabel='Training Epoch', ylabel=f'Loss', title=f"Training and Validation Loss vs. Training Epoch:\n{args['logsavedir']}")
            if e>1000: plt.gca().set(ylim=[0,0.1])
            plt.savefig(os.path.join(args["logsavedir"], f'TrainValLoss_Epoch{e}.jpg'), dpi=300)
            plt.close(fig)
    
        # save checkpoint
        if  (e>0 and e%5000 == 0) or e==start_epoch+epoch-1:
            state = { 'epoch': e + 1, 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss }
            torch.save(state, os.path.join(args["logsavedir"], f"{e}iter.pt"))
    
    log(f"------------FINISHED WITH TRAINING [Time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}]------------\n")



def denoise_1scene(i, args, data_type, method, savedir, fpprefix, title, device, to_vis, scenepath=None, padding_mask=None, fpoc=None, nfpc=None, fpmask=None, fpbpn=None, steer_away=False):
    """Deals with actual optimization process and visualization of results.
       Optimize from generated noisy scene to clean position based on network prediction through one of the denoise_methods. In scale [-1,1].
       i: [1, numobj, pos+ang+siz+cla], input scene
       Methods:
        a. direct_map: directly map to predicted position, use [network prediction] as next iteration's input position
        b. direct_map_once: do the above for one iteration
        c. grad_nonoise/noise: slowly move in direction of prediction, potentially add noise between each step
            c.1. nonoise: use [network input + (displacement * step size) ] as next iter's input
            c.2. noise:   use [network input + (displacement * step size)  + noise]  as next iter's input
       Criteria for ending: either reached max_iter or displacement size small enough """
    direct_map, add_noise = denoise_method_info[method]

    traj = [] # (iter, 6, 2)
    local_max_iter = 1 if method == "direct_map_once" else max_iter # global
    conse_break_meets = 0 # counts the consecutive number of times it has met the break condition
    for iter_i in range(local_max_iter):  
        step_size =  step_size0 / (1 + step_decay*iter_i) 
        noise_scale = noise_scale0 * noise_decay**(iter_i // noise_drop_freq) # standard deviation

        traj.append(torch.squeeze(i.detach().cpu()).tolist()) # add (6,2) or (14,2)

        if args['model_type'] == "Transformer":
            p = model(i, padding_mask, device, fpoc=fpoc, nfpc=nfpc, fpmask=fpmask, fpbpn=fpbpn)  # padding_mask: [batch_size=1, seq_len=maxnumobj]
        else:
            p = model(i) # (1,6,2) or (1,14,2) or (1, 14, 4)

        # normalize at inference time at every iteration
        if ang_d>0: p[:,:,pos_d:pos_d+ang_d] = torch_normalize(p[:,:,pos_d:pos_d+ang_d]) # i is already normalized

        pos_disp = p[:,:,0:pos_d]-i[:,:,0:pos_d] if args['predict_absolute_pos'] else p[:,:,0:pos_d]
        pos_disp_size = np.linalg.norm(pos_disp.detach().cpu().numpy())

        ang_disp_pi, ang_disp_pi_size = None, 1
        if ang_d>0: 
            ang_disp_pi = torch_angle_between(i[:,:,pos_d:pos_d+ang_d], p[:,:,pos_d:pos_d+ang_d]) if args['predict_absolute_ang'] else torch_angle(p[:,:,pos_d:pos_d+ang_d])
                # ang in [-pi, pi] from input to pred; [batch_size, numobj, 1]
            ang_disp_pi_size = torch.mean(torch.abs(ang_disp_pi)).item()
        
        if ((pos_disp_size < pos_disp_break) and (ang_disp_pi_size < ang_disp_pi_break)): 
            conse_break_meets += 1
            if conse_break_meets >= conse_break_meets_max: break  # break on the max-th time
        else:
            conse_break_meets = 0

        # apply the change/prepare input for next iteration (type=i[:,:,pos_d+ang_d:] unchanged)
        if direct_map: step_size=1 
        if steer_away: pos_disp = get_obstacle_avoiding_displacement_bbox(i, pos_disp, step_size, pos_d, ang_d)
        i[:,:,0:pos_d] += pos_disp*step_size 

        if ang_d>0: i[:,:,pos_d:pos_d+ang_d] = torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], ang_disp_pi*step_size) # length preserved (stay normalized)
        
        if add_noise==True: 
            # NOTE: zero-mean gaussian distributions for noise (std-dev designated by noise scale)
            i[:,:,0:pos_d]+=torch.tensor(np.random.normal(size=(i[:,:,0:pos_d]).shape, loc=0.0, scale=noise_scale)).to(device)
            if ang_d>0: 
                ang_noise_scale = noise_scale/noise_scale0 * ((np.pi/4)/10)  # noise_scale0=0.01 correspond to (np.pi/4)/10
                rads = torch.tensor(np.random.normal(size=((i[:,:,pos_d:pos_d+ang_d]).shape[0], (i[:,:,pos_d:pos_d+ang_d]).shape[1], 1), loc=0.0, scale=ang_noise_scale)).to(args['device'])
                i[:,:,pos_d:pos_d+ang_d]= torch_rotate_wrapper(i[:,:,pos_d:pos_d+ang_d], rads).to(args['device'])


    traj.append(torch.squeeze(i.detach().cpu()).tolist()) # add (6/14,2/4/6)
    traj = np.array(traj, dtype=float) # [iter, nobj, pos+ang+sha]
    traj_to_return = np.copy (traj) # traj modified/denormalized in visualize function
    break_idx = iter_i
    perobj_distmoved = np.mean( np.linalg.norm(traj[-1,:,:pos_d]-traj[0, :, :pos_d], axis=1) ) # [nobj, pos_d->None] -> scalar: on avg, how much each obj moved

    if to_vis:
        final_angle = traj[-1,:,pos_d:pos_d+ang_d] if ang_d>0 else None
        if data_type=="3dfront":
            if args["replica"]!="room_0": # normal
                scenepath, fpoc = scenepath, None
                tdf.visualize_tdf_2d_denoise(traj, args=args, scenepath=scenepath, fp=os.path.join(savedir, f"{fpprefix}"), title=f"{title}: {break_idx} iters", fpoc=fpoc)
            else: # replica living room: no boxes.npz to read from for floor plan -> use fpoc instead
                scenepath, fpoc = None, fpoc[0, :nfpc[0], :]*6 #  [nfpc, pos_dim]
                tdf.visualize_tdf_2d_denoise(np.copy(traj), args=args, scenepath=scenepath, fp=os.path.join(savedir, f"{fpprefix}"), title=f"{title}: {break_idx} iters", fpoc=fpoc, vis_traj=True)
                tdf.visualize_tdf_2d_denoise(traj, args=args, scenepath=scenepath, fp=os.path.join(savedir, f"{fpprefix}_notraj"), title=f"{title}: {break_idx} iters", fpoc=fpoc, vis_traj=False)
            
        elif data_type in ["tablechair_horizontal", "tablechair_circle", "tablechair_shape"]:
            visualize_tablechair_denoise(traj, data_type, n_table=n_table, numobj=n_obj, fp=os.path.join(savedir, f"{fpprefix}.jpg"), title=f"{title}: {break_idx} iters")
            # visualize_tablechair_3d(traj[-1,:,0:pos_d], final_angle=final_angle, initial_sha=traj[0,:,pos_d+ang_d:], numobj=n_obj, obj_fp=os.path.join(savedir, f"{fpprefix}-final.obj"))
        else:
            visualize_chair_denoise(traj, data_type, fp=os.path.join(savedir, f"{fpprefix}.jpg"), title=f"{title}: {break_idx} iters")
            visualize_2d_pointcloud_3d(traj[-1,:,0:pos_d], data_type,  os.path.join(savedir, f"{fpprefix}-final.obj"), final_angle=final_angle)


    return traj_to_return, break_idx, pos_disp_size, ang_disp_pi_size, perobj_distmoved


def denoise_batch(de_input, args, savedir="", fpprefix="", numscene=10, startidx=0, method="direct_map", data_type="3dfront", device="cpu",
                  to_vis=True, scenepaths=None, padding_mask=None, fpoc=None, nfpc=None, fpmask=None, fpbpn=None, steer_away=False):
    """ Wrapper around denoise_1scene, dealing with stats and data logging.
        scenepaths, fpoc, nfpc, fpmask, fpbpn: for 3dfront 
    """
    log(f"\n## fpprefix={fpprefix}, method={method}, numscene={numscene}, savedir={savedir}")
    
    trajs, perobj_distmoveds = [], []
    for scene_idx in range(numscene):
        sp=scenepaths[scene_idx] if scenepaths is not None else None
        pm=padding_mask[scene_idx:scene_idx+1] if padding_mask is not None else None
        oc=fpoc[scene_idx:scene_idx+1] if fpoc is not None else None
        nc=nfpc[scene_idx:scene_idx+1] if nfpc is not None else None
        m=fpmask[scene_idx:scene_idx+1] if fpmask is not None else None
        bpn=fpbpn[scene_idx:scene_idx+1] if fpbpn is not None else None

        traj, break_idx, pos_disp_size, ang_disp_pi_size, perobj_distmoved = denoise_1scene(de_input[scene_idx:scene_idx+1], args,data_type, method,
                                                                            savedir, f"{startidx+scene_idx}_{fpprefix}-{method}", f"{startidx+scene_idx} {fpprefix} ({method})", 
                                                                            device, to_vis, scenepath=sp, padding_mask=pm, fpoc=oc, nfpc=nc, fpmask=m, fpbpn=bpn, steer_away=steer_away)
        trajs.append(traj) # append [niter, nobj, pos+ang+siz+cla] # traj[0] is initial, traj[-1] is final
        perobj_distmoveds.append(perobj_distmoved) # append a scalar -> [nscene,]

        scenename = sp.split("/")[-1] if sp else ""
        log(f"scene {scene_idx}: break_idx={break_idx} | pos_disp_size={round(pos_disp_size, 5)} | ang_disp_pi_size={round(ang_disp_pi_size, 5)} | perobj_distmoved ={round(perobj_distmoved, 5)} | {scenename}")

    return np.array(trajs), np.array(perobj_distmoveds) # [nscene, 2, nobj, pos+ang+siz+cla], # [nscene,]


def denoise_meta(args, models, model_names, numscene=10, startidx=0, use_methods=[True, True, True, True], data_type="3dfront", device="cpu", 
                 to_vis=True, noise_levels=[0.25], angle_noise_levels=[np.pi/4], use_sameset = True, save_results=False):
    """ Denoise for each noise level, each model, each denoise method.

        models: list of model to run the denoising process with.
        model_names: to be used in methods and file prefixes
        numscene: number of scene for each noise level
        use_methods: whether to use each of the globally defined denoise_methods ("direct_map_once", "direct_map", "grad_nonoise", "grad_noise")
        use_sameset: use same set of scenes for all noise levels, all models, all denosing methods
        save_results: write denoising data to npz files, including scenepaths, initial scene, trajectories, statistics, etc
    """
    log(f"\n### DENOISE")

    random_idx = None
    data_partition = "test"
    if data_type=="3dfront" and args["replica"]!="room_0" and use_sameset:
        random_idx = tdf.gen_random_selection(numscene, data_partition=data_partition) # gen_stratified_selection
        # random_idx=[1748] # b9d17d23-66f0-445d-acb7-f11887cc4f7f LivingDiningRoom-192367_0 (teaser)
        log(f"random_idx={random_idx}")
        with open(os.path.join(args["logsavedir"], f"random_idx.npy"), 'wb') as f: np.save(f, random_idx)

    all_emd2gts, all_perobj_distmoveds = [], [] # [n_nl, n_model*n_method=4]
    for noise_i in range(len(noise_levels)):
        noise_emd2gts, noise_perobj_distmoveds = [], []
        nl, anl = noise_levels[noise_i], angle_noise_levels[noise_i] # from global variables
        log(f"\n\n------ Denoise: noise level={nl}, angle noise level={round(anl/np.pi*180)} ")

        savedir = os.path.join(args["logsavedir"], f"denoise_data{round(nl, 5)}")
        if args['random_mass_denoising']:  savedir = os.path.join(args["logsavedir"], current_dir, f"de_pos{round(noise_levels[noise_i], 3)}_angle{round(angle_noise_levels[noise_i]/np.pi*180, 3)}")
        if not os.path.exists(savedir): os.makedirs(savedir)

        de_padding_mask, de_scenepaths, de_fpoc, de_nfpc, de_fpmask, de_fpbpn = None, None, None, None, None, None # variable-length data, 3dfront specific
        if data_type=="tablechair_horizontal":
            de_input, de_label = gen_data_tablechair_horizontal_bimodal(numscene, noise_level_stddev=args['train_pos_noise_level_stddev'], angle_noise_level_stddev=args['train_ang_noise_level_stddev'],
                                                                        abs_pos=args["predict_absolute_pos"], abs_ang=args['predict_absolute_ang']) # deafult 0.25 and np.pi/4  # [half_batch_nscene, 14, 8]) 
            de_input, de_label = de_input[numscene:], de_label[numscene:] # Use noisy ones for denoising. NOTE: abs/rel only impacts labels
        elif data_type=="tablechair_circle":
            de_input, de_label = gen_data_tablechair_circle_bimodal(numscene, noise_level_stddev=nl, angle_noise_level_stddev=anl)
            de_input, de_label = de_input[numscene:], de_label[numscene:]
        elif data_type =="tablechair_shape":
            de_input, de_label = gen_data_tablechair_shape_bimodal(numscene, noise_level_stddev=nl, angle_noise_level_stddev=anl)
            de_input, de_label = de_input[numscene:], de_label[numscene:]
        if data_type=="3dfront":
            de_input, de_label, de_padding_mask, de_scenepaths, de_fpoc, de_nfpc, de_fpmask, de_fpbpn = tdf.gen_3dfront(numscene, random_idx=random_idx, data_partition=data_partition, use_emd=args['use_emd'], # random_idx: same set of scenes everytime
                                                                                                                        abs_pos=args["predict_absolute_pos"], abs_ang=args["predict_absolute_ang"], use_floorplan=args["use_floorplan"],
                                                                                                                        noise_level_stddev=nl, angle_noise_level_stddev=anl,
                                                                                                                        weigh_by_class = args['denoise_weigh_by_class'], within_floorplan = args['denoise_within_floorplan'], no_penetration = args['denoise_no_penetration'], 
                                                                                                                        pen_siz_scale=pen_siz_scale, replica=args["replica"])

            de_padding_mask = torch.tensor(de_padding_mask).to(device) # boolean
            if args["use_floorplan"]: # models below may have any variation of encoder regardless of floorplan_encoder_type in args
                de_fpoc, de_nfpc, de_fpmask, de_fpbpn = torch.tensor(de_fpoc).float().to(device), torch.tensor(de_nfpc).int().to(device), torch.tensor(de_fpmask).float().to(device), torch.tensor(de_fpbpn).float().to(device)
        de_input, de_label = torch.tensor(de_input).float().to(device), torch.tensor(de_label).float().to(device) # [numscene*2,nobj,d] - first half is clean
        
        # Visualizing and saving scene's initial state
        vis_input = de_input.detach().clone().cpu()
        initials, groundtruths = [], []
        for scene_idx in range(numscene):
            if data_type=="3dfront":
                scene = vis_input[scene_idx:scene_idx+1].detach().numpy() # [1, numobj, pos_d+ang_d+siz+cla] 
                if args["replica"]!="room_0": # normal
                    scenepath, fpoc = de_scenepaths[scene_idx], None
                else: # replica living room: no boxes.npz to read from for floor plan -> use fpoc instead
                    scenepath, fpoc = None, de_fpoc[scene_idx, :de_nfpc[scene_idx], :]*6 #  [nfpc, pos_dim]
                
                tdf.visualize_tdf_2d_denoise(np.copy(scene), args=args, scenepath=scenepath, fpoc=fpoc, # note the copy here!
                                             fp=os.path.join(savedir, f"{startidx+scene_idx}_initial"), title=f"{startidx+scene_idx}: initial")
                initials.append(scene[0])# [nscene, numobj, pos+ang+siz+cla]
                
                if args["replica"]!="room_0":  # normal
                    read_original, _ = tdf.read_one_scene(os.path.split(de_scenepaths[scene_idx])[1], normalize=False) # [numobj, pos_d+ang_d+siz+cla] 
                    tdf.visualize_tdf_2d(read_original, os.path.join(savedir, f"{startidx+scene_idx}_groundtruth"), f"{startidx+scene_idx}: ground truth",
                                        args=args, traj=None, nobj=None, cla_idx=None, scenepath=de_scenepaths[scene_idx])
                    
                    read_original_normalized, _ = tdf.read_one_scene(os.path.split(de_scenepaths[scene_idx])[1], normalize=True) # [numobj, pos_d+ang_d+siz+cla] 
                    groundtruths.append(read_original_normalized)# [nscene, numobj, pos+ang+siz+cla]
                 
            else:
                initials.append(vis_input[scene_idx].detach().numpy())# [nscene, numobj, pos+ang+siz+cla]

        if save_results:
            saveres_fp = os.path.join(args['logsavedir'], f"pos{round(noise_levels[noise_i], 4)}_ang{round(angle_noise_levels[noise_i]/np.pi*180, 2)}_initial_groundtruth")
            if args['random_mass_denoising']: saveres_fp = os.path.join(args["logsavedir"], current_dir, f"pos{round(noise_levels[noise_i], 4)}_ang{round(angle_noise_levels[noise_i]/np.pi*180, 2)}_initial_groundtruth")
            np.savez_compressed( saveres_fp,  
                random_idx = random_idx,
                scenepaths = np.array(de_scenepaths),
                initial = np.array(initials), # in [-1,1]
                groundtruth = np.array(groundtruths) # read one scene has normalize default to false, in [-1,1]
            )
            log(f"\nDenoising results saved to {saveres_fp}\n")

        # denoising
        for model_i in range(len(models)):
            global model
            model, mn = models[model_i], model_names[model_i] # gloabl variable used in denoise_1scene
            
            model.eval()
            with torch.no_grad():
                log(f"\n")
                model_results = {}
                for method_i in range(len(denoise_methods)): # 4 [direct_map_once, direct_map, grad_nonoise, grad_noise]
                    if not use_methods[method_i]: continue

                    p_m        = None if de_padding_mask is None else de_padding_mask.detach().clone() 
                    fpoc       = None if de_fpoc is None else de_fpoc.detach().clone()
                    nfpc       = None if de_nfpc is None else de_nfpc.detach().clone()
                    fpmask     = None if de_fpmask is None else de_fpmask.detach().clone()
                    fpbpn      = None if de_fpbpn is None else de_fpbpn.detach().clone()
                    trajs, perobj_distmoveds = denoise_batch(de_input.detach().clone(), args, savedir, mn, numscene, startidx=startidx, method=denoise_methods[method_i], data_type=data_type,
                                                 device=device, to_vis=to_vis, scenepaths=de_scenepaths, padding_mask=p_m,  
                                                 fpoc=fpoc, nfpc=nfpc, fpmask=fpmask, fpbpn=fpbpn, steer_away = args['denoise_steer_away'])
                    if data_type=="3dfront" and args["replica"]!="room_0":
                        emd2gt = dist_2_gt(trajs, de_scenepaths, tdf, use_emd=True) # trajs: normalized 
                        noise_emd2gts.append(emd2gt)
                        log(f"{mn}: emd to ground truth = {emd2gt}")
                    
                    model_results[f"{denoise_methods[method_i]}_trajs"] = trajs # [nscene, niter or 2, nobj, pos+ang+siz+cla], in [-1,1] (copied before visualization function)
                    model_results[f"{denoise_methods[method_i]}_perobj_distmoveds"] = perobj_distmoveds # [nscene,]
                    noise_perobj_distmoveds.append(np.mean(perobj_distmoveds))
                    log(f"{mn}: mean per-scene perobj_distmoved = {np.mean(perobj_distmoveds)}")

            if save_results:
                saveres_fp = os.path.join(args['logsavedir'], f"pos{round(noise_levels[noise_i], 4)}_ang{round(angle_noise_levels[noise_i]/np.pi*180, 2)}_{mn}")
                if args['random_mass_denoising']: saveres_fp = os.path.join(args["logsavedir"], current_dir, f"pos{round(noise_levels[noise_i], 4)}_ang{round(angle_noise_levels[noise_i]/np.pi*180, 2)}_{mn}")
                np.savez_compressed( saveres_fp,  
                    random_idx = random_idx,
                    scenepaths = np.array(de_scenepaths),
                    noise_level = np.array([noise_levels[noise_i], angle_noise_levels[noise_i]]),
                    **model_results
                )
                log(f"\nDenoising results saved to {saveres_fp}\n")
            
        all_emd2gts.append(noise_emd2gts)
        all_perobj_distmoveds.append(noise_perobj_distmoveds)
        startidx += numscene

    log(f"\nall_emd2gts={all_emd2gts}") # to print with the commas
    log(f"all_perobj_distmoveds={all_perobj_distmoveds}") # to print with the commas
    log(f"noise_levels={noise_levels}\nangle_noise_levels={angle_noise_levels}")
    log(f"denoise_methods={denoise_methods}\nuse_methods={use_methods}")


def adjust_parameters():
    args['learning_rate'] = 1e-4 * args['train_batch_size']/128
    
    log(f"\nNumber of GPUs (torch.cuda.device_count()) = {gpucount}, torch.cuda.is_available()={torch.cuda.is_available()}\n") 
    if gpucount >= 1: 
        args['train_batch_size'] *= gpucount # break down first dim (batch_size)
        args['learning_rate'] *= gpucount

    pprint(args)
    pprint(args, stream=open(args['logfile'], 'w'))
    log("\n")
    
    if args['loss_func'] == "MSE+L1": 
        global L1_coeff
        # default:
        L1_coeff = 0.07 # sqrt(stabalized MSE=0.005) for 3dfront ->  ~ 0.005 + sqrt(0.005)^2 (equal weights)

        if args['data_type'] == "tablechair_horizontal":
            L1_coeff = 0.3  # sqrt(stabalized MSE=0.09) -> ~0.09 + sqrt(0.09)^2 (equal weights) # l1 loss around 0.23 
        elif args['data_type'] == "tablechair_circle":
            L1_coeff = 0.28 # circle: stablizied MSE around 0.078 -> sqrt(0.08)~0.28
        elif args['data_type'] =="tablechair_shape":
            L1_coeff = 0.09 # # sqrt(stabalized MSE=0.009) -> sqrt(0.009)~0.09

        log(f"loss parameters:\n  L1_coeff={L1_coeff}")

    if not args['train']:
        global max_iter, step_size0, step_decay, noise_scale0, noise_decay, noise_drop_freq, pos_disp_break, ang_disp_pi_break, conse_break_meets_max
            # step_size =  step_size0 / (1 + step_decay*iter_i) 
            # noise_scale = noise_scale0 * noise_decay**(iter_i // noise_drop_freq)
        # default
        pos_disp_break, ang_disp_pi_break, conse_break_meets_max = 0.01, 0.005, 3 # 0.3 degree for angle, break on the 3rd time
        max_iter = 1500
        step_size0, step_decay = 0.1, 0.005 # roughly step_size0 / step_decay*iteration: 0.067 at 100 iter
        noise_scale0, noise_decay, noise_drop_freq = 0.01, 0.9, 10  # multiply by noise_decay=0.9 every noise_drop_freq=2 iterations: 0.005 at iter 50   
        
        if args['data_type']=="3dfront":
            if args['room_type'] == "bedroom":
                step_size0 = 0.08
                noise_scale0 = 0.008
                noise_drop_freq = 8
        elif args['data_type'] in ["tablechair_horizontal", "tablechair_circle", "tablechair_shape"] :
            step_size0 = 0.12 # 0.05
            noise_drop_freq = 2 #4
            conse_break_meets_max = 1

        log(f"\ndenoising parameters:")
        log(f"  max_iter={max_iter}, step_size0={step_size0}, step_decay={step_decay}; noise_scale0={noise_scale0}, noise_decay={noise_decay}, noise_drop_freq={noise_drop_freq}")
        log(f"  pos_disp_break={pos_disp_break}, ang_disp_pi_break={ang_disp_pi_break}, conse_break_meets_max={conse_break_meets_max}; pen_siz_scale={pen_siz_scale}")
   


def logsavedir_from_args(args):
    """  Construct name of directory (logsavedir) to save results to. """
    dateprefix ="log"
    if args['train']:
        if args['predict_absolute_pos']==1 and args['predict_absolute_ang']==1: predtype='abs'
        elif args['predict_absolute_pos']==0 and args['predict_absolute_ang']==0: predtype='rel'
        elif args['predict_absolute_pos']==0 and args['predict_absolute_ang']==1: predtype='relposabsang'
        elif args['predict_absolute_pos']==1 and args['predict_absolute_ang']==0: predtype='absposrelang'
        branch = '1branch' if args['use_two_branch']==0 else '2branch'

        if args['data_type']!="3dfront":
            return os.path.join(f"{dateprefix}_{args['data_type']}", f"{args['timestamp']}") # _{args['loss_func']}_{predtype}
        if args['data_type']=="3dfront":
            cons = f"_{'T' if args['train_weigh_by_class'] else 'F'}{'T' if args['train_within_floorplan'] else 'F'}{'T' if args['train_no_penetration'] else 'F'}_"
            return os.path.join(f"{dateprefix}_{args['room_type']}", f"{args['timestamp']}_{cons}") # _livonly{args['livingroom_only']}_aug{args['use_augment']}_{args['loss_func']}_{args['floorplan_encoder_type']}
    
    elif args['train']==0 and args['compareeval']==0:
        numiterk = int(int(os.path.split(args['model_path'])[-1][:-7]) / 1000)
        if args['data_type']!="3dfront":
            return os.path.join(os.path.split(args['model_path'])[0], f"E_{args['timestamp']}_{numiterk}k") 
        else:
            cons = f"{'T' if args['denoise_weigh_by_class'] else 'F'}{'T' if args['denoise_within_floorplan'] else 'F'}{'T' if args['denoise_no_penetration'] else 'F'}"
            return os.path.join(os.path.split(args['model_path'])[0], f"E_{args['timestamp']}_{cons}_{numiterk}k")  # _livonly{args['livingroom_only']}_aug{args['use_augment']}

    elif args['train']==0 and args['compareeval']==1:
        return os.path.join(f"CE_{dateprefix}_{args['room_type']}", f"{args['timestamp']}")
    


def initialize_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--log", type = int, default=1, help="If 1, write to log file in addition to printing to stdout.")

    parser.add_argument("--train", type = int, default=1, help="Binary flag for train versus inference.")
    parser.add_argument("--model_type", type = str, default="Transformer", # only matters for train
                        choices = ["PointNetSeg", "PointNetPlusPlus_dense", "PointNetPlusPlus_dense_attention",  "PointTransformer", "Transformer"] )
    parser.add_argument("--use_two_branch", type = int, default=0, help="Transformer-specific; if 0, uses 1 branch for position and angle; else use 2 separate branches.")
    parser.add_argument("--predict_absolute_pos", type = int, default=1, help="Absolute position prediction (versus relative)")
    parser.add_argument("--predict_absolute_ang", type = int, default=1, help="Absolute angle/orientation prediction (versus relative)")
   
    ## for train==1:
    parser.add_argument("--resume_train", type = int, default=0, help="Training specific. If 1, resume training from model_path passed in.")
    parser.add_argument("--train_epoch", type = int, default=500000, help="Training specific. Number of epoches to train for.")
    parser.add_argument("--train_batch_size", type = int, default=128, help="Training specific. Each item in batch is a scene.") 
    parser.add_argument("--loss_func", type = str, default="MSE+L1", choices = ["MSE", "MSE+L1"]) 
    parser.add_argument("--train_pos_noise_level_stddev", type = float, default=0.1,  # 0.25 for table chair experiments
                        help="Training specific. Standard deviation of the Gaussian distribution from which the training data position noise level (stddev of actual noise distribution) is drawn")
    parser.add_argument("--train_ang_noise_level_stddev", type = float, default=np.pi/12, # np.pi/4 for table chair experiments
                        help="Training specific. Standard deviation of the Gaussian distribution from which the training data angle noise level (stddev of actual noise distribution) is drawn")
    ### for 3dfront
    parser.add_argument("--train_weigh_by_class", type = int, default=0, help="Training specific. 3D-FRONT specific. If 1, in training data generation, objects with higher volume will move less.") 
    parser.add_argument("--train_within_floorplan", type = int, default=0, help="Training specific. 3D-FRONT specific. If 1, in training data generation, objects must be within boundary of floor plan.") 
    parser.add_argument("--train_no_penetration", type = int, default=0, help="Training specific. 3D-FRONT specific. If 1, in training data generation, objects cannot intersect one another (or only minimally).") 

    
    ## for (train==1 && resume_train==1) or (train==0):
    parser.add_argument("--model_path", type = str, help="If train==1 && resume_train==1, serves as the model path from which to resume training; otherwise, serves as the model to use for inference.")


    ## for train==0
    parser.add_argument("--compareeval", type = int, default=0, help="Inference specific. For comparing multiple models.")

    parser.add_argument("--normal_denoisng", type = int, default=1, help="Inference specific. If 1, will do general purpose inference.")
    parser.add_argument("--stratefied_mass_denoising", type = int, default=0, help="Inference specific. If 1, will do inference with stratefied noise levels, no visualization, same set for all noise level. \
                                                                                    For use cases such as FID/KID/analysis.")
    parser.add_argument("--random_mass_denoising", type = int, default=0, help="Inference specific. If 1, will do inference for a wide range of noise levels and scenes. For use cases such as examining qualitative performance.")
    ### for 3dfront
    parser.add_argument("--denoise_weigh_by_class", type = int, default=0, help="Inference specific. 3D-FRONT specific. If 1, in inference data generation, objects with higher volume will move less.") 
    parser.add_argument("--denoise_within_floorplan", type = int, default=1, help="Inference specific. 3D-FRONT specific. If 1, in inference data generation, objects must be within boundary of floor plan.") 
    parser.add_argument("--denoise_no_penetration", type = int, default=1, help="Inference specific. 3D-FRONT specific. If 1, in inference data generation, objects cannot intersect one another (or only minimally).") 
    parser.add_argument("--denoise_steer_away", type = int, default=0, help="Inference specific. 3D-FRONT specific. If 1, during inference, attempts to steer objects away from one another to avoid penetration.") 



    ## data:
    parser.add_argument("--data_type", type = str, default="3dfront",
                        choices=["tablechair_horizontal", "tablechair_circle", "tablechair_shape", "3dfront"],
                        help = '''
                                -tablechair_horizontal: 2 tables with 6 chairs each in rectangular formation (fixed relative distance).
                                -tablechair_circle: 2 tables with 2-6 chairs each in circular formation.
                                -tablechair_shape: 1 table with 6 chairs of 2 types, one type on each side.
                                -3dfront: 3D-FRONT dataset (professionally designed indoor scenes); currently support bedroom and livingroom.
                               ''')
    parser.add_argument("--replica", type = str, default="", choices = ["", "room_0”"], help="if room_0, use room 0 (living room) from replica dataset as clean scene.")  
    ### for 3dfront
    parser.add_argument("--room_type", type = str, default="livingroom", choices = ["bedroom", "livingroom"], help="3D-FRONT specific." ) # "diningroom", "library
    parser.add_argument("--livingroom_only",  type = int, default=0, help="3D-FRONT specific. livingroom specific. If 1, discard livingdiningroom.") 
    parser.add_argument("--use_emd", type = int, default=1, help="3D-FRONT specific. Earthmover's distance.") # whether to use earthmover distance assignment or original scene for noisy label
    parser.add_argument("--use_floorplan", type = int, default=1, help="3D-FRONT specific. Takes in floor plan of scene. Exact format determined by floorplan_encoder_type.")
    parser.add_argument("--use_augment",  type = int, default=1, help="3D-FRONT specific. Augmentation involved creating 4 rotated versions of each scene.") # for room_type==livingroom (if true, discard livingdiningroom )
    parser.add_argument("--floorplan_encoder_type", type = str, default="pointnet_simple", choices = ["pointnet", "resnet", "pointnet_simple"],
                        help='''3D-FRONT specific.
                                -pointnet: uses fpoc (floor plan ordered corners; [batch_size, maxnfpoc=51, pos_dim]), nfpc (number of floor plan corners; [batch_size])
                                -resnet: uses fpmask (floor plan binary mask; [batch_size, 256, 256, 3])
                                -pointnet_simple (preferred): fpbpn (floor plan boundary points and normals; [batch_size, tdf.nfpbp=250, 4])
                                ''')
    parser.add_argument("--to_3dviz", type = int, default=0, help="3D-FRONT specifc. If 1, will do 3D visualization with 3dviz")



    args = parser.parse_args() #  'argparse.Namespace'
    return vars(args) #'dict'



if __name__ == "__main__":
    args = initialize_parser() # global variable
    args["timestamp"] = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")[:-3] # trim 3/6 digits of ms, used to be str(int(time.time()))
    args["logsavedir"] = logsavedir_from_args(args)
    args["logfile"] = os.path.join(args['logsavedir'], "log.txt")
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args['logsavedir']): os.makedirs(args['logsavedir'])
    gpucount = torch.cuda.device_count()
    
    adjust_parameters()

    print_str, maxnfpoc, nfpbpn  = "", None, None # to be overriden
    if args['data_type']=="tablechair_horizontal":
        n_obj, n_table, pos_d, ang_d, siz_d, cla_d, invsha_d = 14, 2, 2, 2, 2, 2, 0
    elif args['data_type']=="circle":
        n_obj, pos_d, ang_d,  siz_d, cla_d, invsha_d = 6, 2, 2, 0, 0, 0
    elif args['data_type']=="tablechair_circle":
        n_obj, n_table, pos_d, ang_d, siz_d, cla_d, invsha_d = None, 2, 2, 2, 2, 2, 0 # NOTE: None used in visualize_tablechair_denoise and visualize_tablechair_3d to call find_numobj
        print_str = f"gen_no_table={gen_no_table}, ntable={ntable}, nchair_min={nchair_min}, nchair_max={nchair_max}, maxnumobj={maxnumobj}\n\n"
    elif args['data_type'] =="tablechair_shape":
        n_obj, n_table, pos_d, ang_d, siz_d, cla_d, invsha_d = 7, 1, 2, 2, 2, 2, 128
    elif args['data_type']=="3dfront":
        tdf = TDFDataset(args['room_type'], use_augment=args['use_augment'], livingroom_only=args['livingroom_only'])
        n_obj, pos_d, ang_d, siz_d, cla_d, invsha_d, maxnfpoc, nfpbpn = tdf.maxnobj, tdf.pos_dim, tdf.ang_dim, tdf.siz_dim, tdf.cla_dim, 0, tdf.maxnfpoc, tdf.nfpbpn # shape = size+class
        print_str = f"room_type={args['room_type']}, no lamp"
    else:  # rect, rect_trans
        n_obj, pos_d, ang_d,  siz_d, cla_d, invsha_d = 6, 2, 0, 0, 0, 0
    log(f"args['data_type']={args['data_type']}: n_obj={n_obj}, pos_d={pos_d}, ang_d={ang_d}, siz_d={siz_d}, cla_d={cla_d}, invsha_d={invsha_d}\n{print_str}\n")

    input_d = pos_d + ang_d + siz_d + cla_d + invsha_d
    out_d = pos_d + ang_d
    sha_code = True if input_d > pos_d else False
    subtract_f = True

    transformer_config = None
    if args['model_type'] == "PointNetSeg":
        model = PointNetSeg(input_dim=input_d, out_dim=out_d, num_obj=n_obj)
    elif args['model_type'] == "PointNetPlusPlus_dense":
        model = PointNetPlusPlus_dense(input_dim=input_d, out_dim=out_d, shape_code=sha_code, shape_dim=input_d-pos_d, subtract_feats=subtract_f)
    elif args['model_type'] == "PointNetPlusPlus_dense_attention":
        model = PointNetPlusPlus_dense_attention(input_dim=input_d, out_dim=out_d, shape_code=sha_code, shape_dim=input_d-pos_d, subtract_feats=subtract_f)
    elif args['model_type'] == "PointTransformer":
        model = PointTransformer(input_dim=input_d, out_dim=out_d, device=args['device'])
    elif args['model_type'] == "Transformer":
        transformer_use_floorplan = (args['use_floorplan'] and args['data_type']=="3dfront")
        transformer_config = {"pos_dim": pos_d, "ang_dim": ang_d, "siz_dim": siz_d, "cla_dim": cla_d, 
                              "maxnfpoc": maxnfpoc, "nfpbpn": nfpbpn,
                              "invsha_d": invsha_d, "use_invariant_shape": (invsha_d>0),
                              "ang_initial_d": 128, "siz_initial_unit": None, "cla_initial_unit": [128, 128],
                               "invsha_initial_unit": [128, 128], "all_initial_unit": [512, 512], "final_lin_unit": [256, out_d], 
                               "use_two_branch": args['use_two_branch'], "pe_numfreq": 32, "pe_end": 128, 
                               "use_floorplan": transformer_use_floorplan, "floorplan_encoder_type": args['floorplan_encoder_type']}
        log(f"args['model_type']={args['model_type']}: {transformer_config}\n")
        model = TransformerWrapper(**transformer_config)
        
    if gpucount > 1: model = torch.nn.DataParallel(model)

    model = model.to(args['device'])
    log(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    
    if args['train']:
        start_epoch = 0
        if args['resume_train']: model=load_checkpoint(model, args['model_path']) # updates model, optimizer, start_epoch
        
        train(args['train_epoch'], int(args['train_batch_size']/2), args = args, data_type=args['data_type'], device=args['device'])

    else:
        # loading models
        if args['compareeval']:
            modelconfigs = {  
                "modelpaths": ["log1105_livingroom/1105_122803_409_FFF_livonly0_aug1_MSE+L1_pointnet_simple/50000iter.pt", 
                                "log1105_livingroom/1105_122803_409_FFF_livonly0_aug1_MSE+L1_pointnet_simple/50000iter.pt"],
                "model_names": ["liv0.1-with_steer", "liv0.1-without_steer" ]
            } # only labels change for abs vs rel -> no impact
            log(f"\nmodelconfigs: {modelconfigs}")

            models = []
            for model_i in range(len(modelconfigs['modelpaths'])): 
                for key in modelconfigs: 
                    if key in transformer_config: transformer_config[key] = modelconfigs[key][model_i]
                local_model = TransformerWrapper(**transformer_config)
                models.append(load_checkpoint(local_model, modelconfigs['modelpaths'][model_i])) # local_model with state dict loaded
            model_names =modelconfigs["model_names"]

        else:
            model=load_checkpoint(model, args['model_path'])  # updates model, optimizer, start_epoch
            models = [model]
            model_epoch = args['train_epoch'] if args['train'] else args['model_path'].split("/")[-1][:-7]
            model_names = [f"train{model_epoch}"]

        
        # inference/denoising
        ## 1. Normal denoising
        if args['normal_denoisng']:
            noise_levels = [0.1, 0.3, 0.5]  # [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7] 
            angle_noise_levels = [np.pi/12, np.pi/4, np.pi/2]  # [np.pi/20, np.pi/12, np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]  # 9, 15, 22.5, 30, 45, 60, 90, 180 
            denoise_meta(args, models, model_names, numscene=2, startidx=0, use_methods=[True, True, True, True], data_type=args['data_type'], device=args['device'],
                         to_vis=True, noise_levels=noise_levels, angle_noise_levels=angle_noise_levels, use_sameset=False, save_results=True)

        ## 2. Mass inference/denoising
        if args['stratefied_mass_denoising']:
            noise_levels = [0.25]
            angle_noise_levels = [np.pi/4] # [np.pi/24, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]  # 15, 30, 45, 60, 90, 180 
            denoise_meta(args, models, model_names, numscene=500, startidx=0, use_methods=[True, True, True, True], data_type=args['data_type'], device=args['device'],
                         to_vis=False, noise_levels=noise_levels, angle_noise_levels=angle_noise_levels, use_sameset=True, save_results=True)

        ## 3. Mass inference/denoising
        if args['random_mass_denoising']:
            numscene, startidx, n_nl, nset = 1, 0, 10, 2 # structure of set for ease of navigation
            for i in range(nset):
                current_dir = f"set{i}"
                noise_levels = sorted(np.random.uniform(low=0.05, high=0.7, size=[n_nl])) 
                angle_noise_levels = sorted(np.random.uniform(low=np.pi/24, high=np.pi, size=[n_nl]))
                log(f"Denoising: noise_levels={noise_levels}, angle_noise_levels={angle_noise_levels}")
                denoise_meta(args, models, model_names, numscene=numscene, startidx=startidx, use_methods=[False, False, True, True], data_type=args['data_type'], device=args['device'],
                             to_vis=True, noise_levels=noise_levels, angle_noise_levels=angle_noise_levels, use_sameset=False, save_results=True)
                startidx += numscene*n_nl


    log(f"\nTime: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}")
    log(f"### Done (results saved to {args['logfile']})\n")