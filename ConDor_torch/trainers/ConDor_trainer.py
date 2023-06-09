import torch
import os
from ConDor_torch.models import *
from ConDor_torch.utils import *
from ConDor_torch.utils.train_utils import random_rotate, mean_center, perform_rotation, orthonormalize_basis
import ConDor_torch.datasets
from ConDor_torch.datasets import dictionary_from_3D_FRONT
from ConDor_torch.datasets import dataset_from_3D_FRONT as dataset_utils

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import hydra
from ConDor_torch.utils.pointcloud_utils import kdtree_indexing, save_pointcloud, convert_yzx_to_xyz_basis
from scipy.spatial.transform import Rotation
from pytorch3d.loss import chamfer_distance
import tqdm

class ConDor_trainer(pl.LightningModule):
    '''
    Segmentation trainer to mimic NeSF
    '''

    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()
        self.hparam_config = configs
        self.ConDor = getattr(eval(self.hparam_config.model["file"]), 
                                  self.hparam_config.model["type"])(**self.hparam_config.model["args"])
        
        self.hparam_config.dataset.root = os.path.join(hydra.utils.get_original_cwd(),self.hparam_config.dataset.root)
        self.loss_weights = self.hparam_config.loss


    
    def train_dataloader(self):
        
        train_data_set = getattr(getattr(datasets, self.hparam_config.dataset.file), self.hparam_config.dataset.type)(data_path = self.hparam_config.dataset.root, files_list = self.hparam_config.dataset.train_files,**self.hparam_config.dataset.args)
        train_dataloader = DataLoader(train_data_set, **self.hparam_config.dataset.loader.args, shuffle = True)

        return train_dataloader

    def val_dataloader(self):

        val_data_set = getattr(getattr(datasets, self.hparam_config.dataset.file), self.hparam_config.dataset.type)(data_path = self.hparam_config.dataset.root, files_list = self.hparam_config.dataset.val_files,**self.hparam_config.dataset.args)
        val_dataloader = DataLoader(val_data_set, **self.hparam_config.dataset.loader.args, shuffle = False)
        return val_dataloader


    def forward_pass(self, batch, batch_idx, return_outputs=False):
        
        x = batch["pc"].clone()
        x = mean_center(x)
        if self.hparam_config.feature.rotation.use:
            if return_outputs:
                x, rotation = random_rotate(x, return_rotation=True)
            else:
                x = random_rotate(x)
                
        if (return_outputs) and (not self.hparam_config.feature.rotation.use):
            rotation = torch.eye(3).unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1)
            

        # KDT indexing required
        x = kdtree_indexing(x)
        out_dict = self.ConDor(x)
        
        # Computing losses
        loss_dictionary = self.compute_loss(batch, x, out_dict)

        if return_outputs:
            out_dict["x_input"] = x
            out_dict["random_rotation"] = rotation
            return loss_dictionary, out_dict
        else:
            return loss_dictionary

    def compute_loss(self, batch, x, outputs):
        """
        Computing losses for 
        """

        loss_dictionary = {}
        loss = 0.0
        
        inv, basis = outputs["points_inv"], outputs["E"]
        basis = torch.stack(basis, dim = 1)

        orth_basis = orthonormalize_basis(basis)
        input_pcd_pred = torch.einsum("bvij, bpj->bvpi", orth_basis, inv)
        input_pcd_pred = torch.stack([input_pcd_pred[..., 2], input_pcd_pred[..., 0], input_pcd_pred[..., 1]], dim = -1)

        error_full = torch.mean(torch.mean(torch.sqrt(torch.square(x.unsqueeze(1) - input_pcd_pred) + 1e-8), dim = -1), dim = -1)
        values, indices = torch.topk(-error_full, k = 1)
        orth_basis_frames = orth_basis
        # print(orth_basis.shape)
        # print(indices.shape)
        orth_basis = orth_basis[torch.arange(indices.shape[0]), indices[:, 0]]
        # print(orth_basis.shape)

        y_p = torch.einsum("bij, bpj->bpi", orth_basis, inv)
        y_p = torch.stack([y_p[..., 2], y_p[..., 0], y_p[..., 1]], dim = -1)
        outputs["y_p"] = y_p
        basis_instance_to_canonical = torch.pinverse(convert_yzx_to_xyz_basis(orth_basis))
        canonical_x = torch.einsum('bij,bkj->bki', basis_instance_to_canonical, x)
        outputs["x_canonical"] = canonical_x
        outputs["basis_instance_to_canonical"] = basis_instance_to_canonical

        # Losses
        separation_loss_basis = -torch.mean(torch.abs(basis[:, None] - basis[:, :, None]))
        l2_loss = torch.mean(torch.sqrt(torch.square(x - y_p) + 1e-8))
        chamfer_loss = chamfer_distance(x, y_p)[0]
        orth_loss = torch.mean(torch.abs(basis - orth_basis_frames.detach()))



        if self.loss_weights.l2_loss > 0.0:
            loss += self.loss_weights.l2_loss * l2_loss
        
        if self.loss_weights.chamfer_loss > 0.0:
            loss += self.loss_weights.chamfer_loss * chamfer_loss
        
        if self.loss_weights.orth_loss > 0.0:
            loss += self.loss_weights.orth_loss * orth_loss
        
        if self.loss_weights.separation_loss_basis > 0.0:
            loss += self.loss_weights.separation_loss_basis * separation_loss_basis
        
        loss_dictionary["l2_loss"] = l2_loss
        loss_dictionary["chamfer_loss"] = chamfer_loss  
        loss_dictionary["orth_loss"] = orth_loss  
        loss_dictionary["separation_loss_basis"] = separation_loss_basis  

        loss_dictionary["loss"] = loss
        
        return loss_dictionary

    def training_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary)

        return loss_dictionary["loss"]

    def validation_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary, val = True)

        return loss_dictionary["loss"]


    def configure_optimizers(self):

        optimizer1 = getattr(torch.optim, self.hparam_config.optimizer.type)(list(self.ConDor.parameters()), **self.hparam_config.optimizer.args)
        scheduler1 = getattr(torch.optim.lr_scheduler, self.hparam_config.scheduler.type)(optimizer1, **self.hparam_config.scheduler.args)

        return [optimizer1], [scheduler1]



    def log_loss_dict(self, loss_dictionary, val = False):

        for key in loss_dictionary:
            if val:
                self.log("val_" + key, loss_dictionary[key], **self.hparam_config.logging.args)
            else:
                self.log(key, loss_dictionary[key], **self.hparam_config.logging.args)


    def test_step(self, x):
        '''
        Input:
            x - B, N, 3
        Output:
            output_dictionary - dictionary with all outputs and inputs
        '''
        output_dictionary = self.forward_pass(x)

        return output_dictionary


    def save_outputs(self, save_file_name, out_dict):
        """
        Save outputs to file
        """
        save_keys = ["x_input", "points_inv", "y_p", "x_canonical"]
        for key in out_dict:
            #print(key)
            if key in save_keys:
                pcd_name = save_file_name + "_" + key + ".ply"
                save_pointcloud(out_dict[key], pcd_name)

    

        
        
    

    def run_test(self, dataset_num = 1, save_directory = "./pointclouds", max_iters = None, skip = 20):
        '''
        Run test pass on the dataset
        '''

        self.hparam_config.dataset.loader.args.batch_size = 1
        loader = self.val_dataloader()

        if max_iters is not None:
            max_iters = min(max_iters, len(loader))
        else:
            max_iters = len(loader)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(loader)):
                if i % skip == 0:
                    #print(i)
                    batch["pc"] = batch["pc"].cuda()
                    x = batch   
                    out, pcd_output = self.forward_pass(x, 0, return_outputs = True)
                    
                    save_file_name = os.path.join(save_directory, "") + "%d" % i
                    self.save_outputs(save_file_name, pcd_output)

    def update_dictionary(self, output_dict, out_dict):
    
        save_keys = ["x_input", "x_canonical", "basis_instance_to_canonical", "points_code", "random_rotation"]
        
        
        # for key in out_dict:
        for key in save_keys:
            if output_dict.get(key) is None:
                output_dict[key] = []
            output_dict[key].append(out_dict[key].detach().squeeze(0).cpu().numpy())
    
    
    def run_test_on_dataset(self, h5_files_path, output_h5_file = "./ConDor_outputs/test.h5", apply_rotation = True, skip = 1, num_points = 1024):
        '''
        Run test pass on the dataset
        '''
        
        self.hparam_config.feature.rotation.use = apply_rotation
        
        if apply_rotation:
            print("[INFO]: Applying random rotations while testing")
        else:
            print("[INFO]: Not applying random rotations while testing")
            
        if type(h5_files_path) is str:
            h5_files_path = [h5_files_path]
        
        
        output_dictionary = {}
        output_h5_file = os.path.join(hydra.utils.get_original_cwd(), output_h5_file)
        for h5_idx in range(len(h5_files_path)):
            h5_file = os.path.join(hydra.utils.get_original_cwd(), h5_files_path[h5_idx])
            data_dictionary = dictionary_from_3D_FRONT.h5_to_dictionary(h5_file)
            
            if not os.path.exists(os.path.dirname(output_h5_file)):
                os.makedirs(os.path.dirname(output_h5_file))

            with torch.no_grad():
                for i, key in enumerate(tqdm.tqdm(data_dictionary)):
                    
                    if i % skip != 0:
                        continue
                    
                    x_np = data_dictionary[key]["data"]
                    if num_points < x_np.shape[0]:
                        idx = np.random.permutation(np.arange(x_np.shape[0]))[:num_points]
                        x_np = x_np[idx, ...]
                     
                    x = {}   
                    x["pc"] = torch.from_numpy(x_np).unsqueeze(0).to(torch.float32).cuda() # B, N, 3
                    out, pcd_output = self.forward_pass(x, 0, return_outputs = True)
                    
                    self.update_dictionary(output_dictionary, pcd_output)
                    if output_dictionary.get("id") is None:
                        output_dictionary["id"] = []
                    output_dictionary["id"].append(key)
                    # output_dictionary[key] = self.get_update_dictionary(pcd_output)
        
        for key in output_dictionary:
            if key != "id":
                output_dictionary[key] = np.array(output_dictionary[key])
        
        dataset_utils.write_dict_to_h5(output_dictionary, output_h5_file)
        output_dictionary_loaded = dictionary_from_3D_FRONT.h5_to_dictionary(output_h5_file)
        for key in output_dictionary_loaded:
            print(output_dictionary_loaded[key].keys(), key)
            break