""" For general data processing. """

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import torch
import h5py


## GENERAL
def generate_indices_array(m,n):
    r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out # out[2,1] = array([2, 1])

def generate_pixel_centers(m,n):
    r0 = np.arange(m) + 0.5 # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n) + 0.5
    out = np.empty((m,n,2))
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out # out[2,1] = array([2, 1])

def normalize(x, minimum, maximum):
    """ Normalize x from [minimum, maximum] to range [-1, 1].
        x has shape (pt, d). minimum/maximum have shape (d,)"""
    X = np.clip(x.astype(np.float32), minimum, maximum)
    X = ((X - minimum) / (maximum - minimum))
    X = 2 * X - 1
    return X

def denormalize(x, minimum, maximum):
    """ Denormalize x from [-1, 1] to [minimum, maximum] 
        x has shape (pt, d). minimum/maximum have shape (d,)"""
    X = (x + 1) / 2
    X = X * (maximum - minimum) + minimum
    return X



# GENERAL: data related
def generate_6_points(scale=0.25):
    """ Used as ground truth. Duplicated from data_rect.py to circumvent circular logic in importing.
        Default scale=0.25 gives x in range [-0.5, 0.5] and y in range (-0.25, -.25)"""
    x = [[-2., -1.],[0.,-1.],[2.,-1.],
         [-2., 1.],[0.,1.],[2.,1.]]
    x = np.array(x)
    return scale*x

def angles_to_circle_scene(angles, center, radius):
    """ angles: numpy array of shape (numpt,1), in radians
        center: (2,), center coordinate of circular formation
        radius: scalar, radius of circular formation

        Returns: pos and ang both have shape (numpt,2)
    """
    cos = np.cos(angles) # (numpt,1)
    sin = np.sin(angles) # (numpt,1)
    pos = np.concatenate([center[0]+radius*cos, center[1]+radius*sin], axis=1) # (numpt, 1+1)
    ang = np.concatenate([-cos, -sin], axis=1) # (numpt, 1+1)
    return np.around(np.array(pos), decimals=3), np.around(np.array(ang), decimals=3)


def apply_trans(input_pre, labels_pre, x_max=0.5, y_max=0.5):
    """Generate x and y translation for each scene, and apply it to both the input and label of that scene."""
    numscene, numobj = input_pre.shape[0], input_pre.shape[1]

    trans_x = np.random.uniform(0, x_max, size=(numscene,1))
    trans_y = np.random.uniform(0, y_max, size=(numscene,1))
    trans = np.concatenate([trans_x, trans_y], axis=1) * np.random.choice([-1, 1], size=(numscene, 2)) #(numscene, 2): trans per scene
    
    trans = np.expand_dims(trans, axis=1) # (numscene, 1, 2)
    trans = np.repeat(trans, repeats=numobj, axis=1) # (numscene, numobj, 2)

    input_post = input_pre + trans
    labels_post = labels_pre + trans
    return input_post, labels_post


def apply_rot(input, labels, ang_min=-np.pi/2, ang_max=np.pi/2):
    """ input, labels: [batch_size=numscene, numobj, >=2] where [:,:,0:2] is pos and [:,:,2:4] (if exists)
                       is [cos(th), sin(th)]. [:,:,4:] unchanged

        Generate scene-level rotation (with respect to origin) for each scene and apply it to both the input 
        and label of that scene (both to position and angles).
          
    """
    numscene, numobj = input.shape[0], input.shape[1]

    rot_ang = np.random.uniform(ang_min, ang_max, size=(numscene,1)) 
    rot_ang = np.expand_dims(rot_ang, axis=1) # [numscene, 1, 1]
    rot_ang = np.repeat(rot_ang, repeats=numobj, axis=1) # [numscene, numobj, 1]

    # Rotate position (with respect to origin and positive x-axis)
    input[:,:,0:2]  = np_rotate_wrapper(input[:,:,0:2], rot_ang) # [batch_size, numobj, 2]
    labels[:,:,0:2] = np_rotate_wrapper(labels[:,:,0:2], rot_ang) # [batch_size, numobj, 2]

    if input.shape[2]>2: # Rotate angle vector: cos[th], sin[th]
        input[:,:,2:4]  = np_rotate_wrapper(input[:,:,2:4], rot_ang) # [batch_size, numobj, 2]
        labels[:,:,2:4] = np_rotate_wrapper(labels[:,:,2:4], rot_ang) # [batch_size, numobj, 2]

    return input, labels




## INTERSECTION DETECTION
# Reference: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def do_segment(p, q, r):
    """ Given three collinear points p, q, r, checks if q lies on line segment pr """
    if ((q[0] <= max(p[0],r[0])) and (q[0] >= min(p[0],r[0])) and (q[1] <= max(p[1],r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def orientation(p, q, r):
    """ Find the orientation of an ordered triplet (p,q,r)
        0 : Collinear points; 1 : Clockwise points;  2 : Counterclockwise
    """
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(r[1] - q[1]) * (q[0] - p[0]))
    if (val > 0): # slope of the line qp < slope of the line rq
        return 1 # clockwise
    elif (val < 0): # slope of the line qp > slope of the line rq
        return 2 # counterclockwise
    else:
        return 0 # colinear

def do_intersect(p1,q1, p2,q2):
    """returns true if the line segment 'p1q1' and 'p2q2' intersect.
       p1, q1, p2, q2: 1d iterables
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)): return True

    # Special Cases
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and do_segment(p1, p2, q1)): return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and do_segment(p1, q2, q1)): return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and do_segment(p2, p1, q2)): return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and do_segment(p2, q1, q2)): return True

    # If none of the cases
    return False



def get_obstacle_avoiding_displacement(input_mat, pos_disp, step_size, pos_d=2, ang_d = 2):

    new_pose = input_mat[:, :, :pos_d] + pos_disp * step_size # 1, N, 2

    
    bounding_box_length = input_mat[:, :, pos_d+ang_d: pos_d+ang_d+2]
    l, h = bounding_box_length[:, :, 0], bounding_box_length[:, :, 1]

    r = (torch.square(l) + torch.square(h) + 1e-8).type_as(l) / 2# 1, N

    denom_length = torch.square(r[:, :, None] + r[:, None]) # 1, N, N
    error = new_pose[:, :, None] - new_pose[:, None] # 1, N, N, 2
    collision_error = (torch.sum(torch.square(error), -1) + 1e-6) / (denom_length + 1e-8) # 1, N, N
    collision_error = torch.exp(-collision_error) # 1, N, N
    grad = collision_error.unsqueeze(-1) * -2 * error / (denom_length.unsqueeze(-1) + 1e-8) # 1, N, N, 2
    weigh_factor = torch.ones((grad.shape[0], grad.shape[1], grad.shape[2])).type_as(grad) # B, N, N
    I = torch.eye(grad.shape[1]).unsqueeze(0).repeat(grad.shape[0], 1, 1).type_as(grad) # B, N, N
    # print(I.shape, weigh_factor.shape)
    weigh_factor = (weigh_factor - I).unsqueeze(-1)
    # print(weigh_factor.shape, grad.shape)
    grad = torch.sum(weigh_factor * grad,  axis = -2) # 1, N, 2
    # print(torch.max(grad))
    # print(torch.max(r))
    # grad = 0.002 * torch.sqrt(r + 1e-8).unsqueeze(-1) * (grad / torch.norm(grad, p = 2, dim = -1, keepdim= True))
    # print(torch.max(grad), torch.min(grad))
    # norm_grad = torch.norm(grad, p = 2, dim = -1, keepdim= True)
    # grad =  5e-4 * torch.sqrt(norm_grad + 1e-6) * (grad + 1e-6) / (norm_grad + 1e-6)
    grad =  5e-4 * grad #torch.sqrt(norm_grad + 1e-6) * (grad + 1e-6) / (norm_grad + 1e-6)
    
    # loss = torch.sum(collision_error, axis=-1) # 1, N

    # min_l2_distance, min_indices = torch.min(l2_distances, -1) # 1, N
    return pos_disp - grad


def get_obstacle_avoiding_displacement_bbox(input_mat, pos_disp, step_size, pos_d=2, ang_d = 2):

    new_pose = input_mat[:, :, :pos_d] + pos_disp * step_size # 1, N, 2
    
    bounding_box_length = input_mat[:, :, pos_d+ang_d: pos_d+ang_d+2]
    l, h = bounding_box_length[:, :, 0], bounding_box_length[:, :, 1]
    c, s = input_mat[:, :, pos_d:pos_d + 1], input_mat[:, :, pos_d+1:pos_d + 2]# 1, N, 2
    
    inv_rot_mat = torch.stack([torch.concat([c, s], axis = -1), torch.concat([-s, c], axis = -1)], -2) # 1, N, 2, 2

    b_pts_t_r = torch.stack([l/2, h/2], -1) # B, N, 2
    b_pts_t_l = torch.stack([-l/2, h/2], -1)
    b_pts_b_l = torch.stack([-l/2, -h/2], -1)
    b_pts_b_r = torch.stack([l/2, -h/2], -1)

    b_pts = torch.stack([b_pts_t_r, b_pts_b_r, b_pts_b_l, b_pts_t_l], -2) # B, N, 4, 2

    # inv rotate boundary pts
    # print(b_pts.shape, inv_rot_mat.shape)
    b_pts = torch.einsum("bnij, bnpi->bnpj", inv_rot_mat, b_pts) # B, N, 4, 2
    b_pts = b_pts + new_pose.unsqueeze(-2)

    grad_new = torch.zeros_like(pos_disp)

    for obj_idx in range(input_mat.shape[1]):
        
        obj_pos = new_pose[:, obj_idx] # B, 2
        l_obj, h_obj = l[:, obj_idx].unsqueeze(-1), h[:, obj_idx].unsqueeze(-1) # B, 1

        new_pose_rel_obj = new_pose - obj_pos.unsqueeze(1) # B, N, 2

        bool_boundary_tensor = torch.ones((b_pts.shape[0], b_pts.shape[1], b_pts.shape[2])) == 1.0
        bool_boundary_tensor[:, obj_idx] = False
        bool_boundary_tensor = bool_boundary_tensor.reshape(b_pts.shape[0], -1)
        b_pts_obj = b_pts.reshape(b_pts.shape[0], -1, 2)
        new_pose_b_pts = b_pts_obj - obj_pos.unsqueeze(1)

        new_pose_rel_obj_ur = torch.concat([new_pose_rel_obj, new_pose_b_pts], axis = 1) # B, N + 4N, 2,  relative position of other objs to selected obj (without rotation change)
        new_pose_rel_obj = torch.einsum("bij, bnj->bni", inv_rot_mat[:, obj_idx], new_pose_rel_obj_ur) # B, N, 2
        
        collision_obj = torch.logical_and(torch.logical_and(new_pose_rel_obj[:, :, 0] <= l_obj/2, new_pose_rel_obj[:, :, 0] >= -l_obj/2), torch.logical_and(new_pose_rel_obj[:, :, 1] <= h_obj/2, new_pose_rel_obj[:, :, 1] >= -h_obj/2)) # B, N

        # print(collision_obj.shape, l.shape, bool_boundary_tensor.shape)
        # Setting current object collision to False
        collision_obj[:, obj_idx] = False
        collision_obj[:, l.shape[1]:] = bool_boundary_tensor

        error = new_pose_rel_obj_ur
        collision_error = torch.sum(torch.square(error), -1)  # B, N
        grad_obj = torch.mean(collision_obj.unsqueeze(-1).type_as(error) * torch.exp(-collision_error).unsqueeze(-1) * 2 * error, 1)

        grad_new[:, obj_idx] = grad_obj
        # boundary_obj_pts = obj[:, 0] + l

    grad =  1 * grad_new #torch.sqrt(norm_grad + 1e-6) * (grad + 1e-6) / (norm_grad + 1e-6) 
    # 0.05
    
    # loss = torch.sum(collision_error, axis=-1) # 1, N
    # min_l2_distance, min_indices = torch.min(l2_distances, -1) # 1, N
    
    return pos_disp - grad




## NOISE: positions
def np_add_gaussian_noise_scale(x, sigma):
    g = np.random.normal(size=x.shape, loc=0.0, scale=sigma)
    return x + g

def np_add_gaussian_noise_noiselevel(x,  noise_level_min=0.1, noise_level_max=0.5):
    """ x: [nobj, d], represents one scene. All elements within this array share one noise level"""
    noise_level = np.random.uniform(low=noise_level_min, high=noise_level_max) 
    g = np.random.normal(size=x.shape, loc=0.0, scale=noise_level)
    return x + g

def np_add_uniform_gaussian_noise(x, noise_level_min=0.0, noise_level_max=0.5):
    """x: [B, nobj, d] """
    noise = np.zeros((x.shape))
    for scene_idx in range(x.shape[0]): # each scene has its own noise level
        noise_level = np.random.uniform(low=noise_level_min, high=noise_level_max)
        noise[scene_idx] = np.random.normal(size=x.shape[1:], loc=0.0, scale=noise_level)
    return x + noise


def np_add_gaussian_gaussian_noise(x, noise_level_stddev=0.1): 
    """ x: [B, nobj, d] 
        noise_level_stdev:
            0.1:  68% < 1 stddev=0.1 (higher probability closer to 0), 95% < 0.2, 99.7% < 0.3
            0.15: 50% has noise_level < 0.1 (higher probability closer to 0); 68% <0.15; 82% < 0.2, 95% < 0.3
    """
    noise = np.zeros((x.shape))
    scene_noise_levels = abs(np.random.normal(size=x.shape[0], loc=0.0, scale=noise_level_stddev))
    for scene_idx in range(x.shape[0]):
        noise[scene_idx] = np.random.normal(size=x.shape[1:], loc=0.0, scale=scene_noise_levels[scene_idx])
    return x + noise



## NOISE: angles

def np_add_gaussian_gaussian_angle_noise(x, noise_level_stddev=np.pi/12):
    """x: [batch_size, maxnumobj, ang_dim=2=[cos(th), sin(th)] ]"""
    noise_rads = np.zeros((x.shape[0], x.shape[1], 1)) # [B, maxnumobj, 1]
    scene_noise_levels = abs(np.random.normal(size=x.shape[0], loc=0.0, scale=noise_level_stddev))
    for scene_idx in range(x.shape[0]):
        noise_rads[scene_idx] = np.random.normal(size=(x.shape[1], 1), loc=0.0, scale=scene_noise_levels[scene_idx])
    return np_rotate_wrapper(x, noise_rads)


def np_add_uniform_gaussian_angle_noise(x, noise_level_min=0.0, noise_level_max=np.pi/2):
    """x: [batch_size, maxnumobj, ang_dim=2=[cos(th), sin(th)] ]"""
    noise_rads = np.zeros((x.shape[0], x.shape[1], 1)) # [B, maxnumobj, 1]
    for scene_idx in range(x.shape[0]):
        noise_level = np.random.uniform(low=noise_level_min, high=noise_level_max)
        noise_rads[scene_idx] = np.random.normal(size=(x.shape[1], 1), loc=0.0, scale=noise_level)
    return np_rotate_wrapper(x, noise_rads)


def np_add_angle_noise(v, low=-np.pi/6, high=np.pi/6):
    """ v: has shape [batch_size, numobj, 2], magnitude preserved, each object rotated indepndently.
        Returns rot_a, which has shape [batch_size, numobj, 2]
        NOTE: deprecated as it is doing uniform not Gaussian noise.
    """
    return np_rotate_wrapper(v, np.random.uniform(low, high, size=(v.shape[0], v.shape[1], 1)))


def trig2ang(cossin):
    """ Convert given unit vector [cos, sin] to angle (in radians) with respect to (1, 0).
        cossin: [num_angle, 2]
        Return: [num_angle, 1] angles in radians [-pi, pi]"""
    # arccos = np.arccos(cossin[:,0]) #(num_angle,) # [-1, 1] -> [0,pi] (rad)
    # arccos = ((cossin[:,1]<0)*-2 + 1) * arccos % (2*np.pi)# if sin < 0, rad*(-1) # [0, 2pi]
    # return np.expand_dims(arccos, axis=1) 
    return np.arctan2(cossin[:,1:2], cossin[:,0:1])  # [num_angle, 1]


def np_rotate_center(vs, rads, centers):
    """
    Rotate  counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
        vs     : [numpt, 2], length perserved.
        rads   : [numpt, 1], in radians, counterclockwise
        centers: [2,], center of rotation
        
        Rotates each vs[i] by rads[i] counterclockwise with respect to origins[i]
        Returns rotated vs, which has shape [numpt, 2]
    """
    rot_vs = np.zeros(vs.shape)
    diff = vs-centers #[numpt, 2]
    rot_vs[:,0:1] = centers[0] + (np.cos(rads)*diff[:,0:1] - np.sin(rads)*diff[:,1:2])
    rot_vs[:,1:2] = centers[1] + (np.sin(rads)*diff[:,0:1] + np.cos(rads)*diff[:,1:2])
    return rot_vs

def np_rotate(vs, rads):
    """ vs  : [numpt, 2], length perserved.
        rads: [numpt, 1], in radians, counterclockwise
        
        Rotates each vs[i] by rads[i] counterclockwise with respect to the positive x axis about the origin.
        Returns rotated vs, which has shape [numpt, 2]
    """
    rot_vs = np.zeros(vs.shape)
    rot_vs[:,0:1] = np.cos(rads)*vs[:,0:1] - np.sin(rads)*vs[:,1:2]
    rot_vs[:,1:2] = np.sin(rads)*vs[:,0:1] + np.cos(rads)*vs[:,1:2]
    return rot_vs

def np_rotate_wrapper(vs, rads):
    """ vs  : [batch_size, numobj, 2]
        rads: [batch_size, numobj, 1], positive is counterclockwise
        Returns rotated vs, which has shape [batch_size, numobj, 2]
    """
    numscene, numobj = vs.shape[0], vs.shape[1]
    rot_vs = np_rotate(vs.reshape(-1, vs.shape[2]),rads.reshape(-1, rads.shape[2]))
    return rot_vs.reshape(numscene, numobj, -1)

def torch_rotate(vs, rads):
    """ vs  : [numpt, 2], length perserved.
        rads: [numpt, 1], in radians, counterclockwise
        Rotates each vs[i] by rads[i] with respect to the positive x axis about the origin.
        Returns rotated vs, which has shape [numpt, 2]
    """
    rot_vs = torch.zeros(vs.shape)
    rot_vs[:,0:1] = torch.cos(rads)*vs[:,0:1]-torch.sin(rads)*vs[:,1:2]
    rot_vs[:,1:2] = torch.sin(rads)*vs[:,0:1]+torch.cos(rads)*vs[:,1:2]
    return rot_vs

def torch_rotate_wrapper(vs, rads):
    """ vs  : [batch_size, numobj, 2]
        rads: [batch_size, numobj, 1], positive is counterclockwise
        Returns rotated vs, which has shape [batch_size, numobj, 2]
    """
    numscene, numobj = vs.size(dim=0), vs.size(dim=1)
    rot_vs = torch_rotate(torch.flatten(vs, start_dim=0, end_dim=1), torch.flatten(rads, start_dim=0, end_dim=1))
    return rot_vs.reshape(numscene, numobj, -1)




def np_normalize(v):
    """ Returns a copy of v normalized along the last dimension, same shape as input """
    norm = np.expand_dims(np.linalg.norm(v, axis=-1), axis=-1)
    return v / norm

def np_single_angle_between(v1, v2):
    """ Returns the angle in radians between 1d vectors v1 and v2, in [0, pi] """
    v1_u = np_normalize(v1)
    v2_u = np_normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def np_angle_between(v1, v2):
    """ v1 and v2: [batch_size, numobj, 2], last dim = (cos(th), sin(th))
        Returns angles in [-pi, pi] from v1 to v2 (order matters), of shape [batch_size, numobj, 1].
    """
     # NOTE: only use case in TDFront.py (already normalized)
    cross_prod = v1[:,:,0]*v2[:,:,1] - v1[:,:,1]*v2[:,:,0] # sin; [batch_size, numobj]
    inner_prod = (v1 * v2).sum(axis=2) # cos; [batch_size, numobj]
    ang_rad_pi = np.arctan2(cross_prod, inner_prod)  # [-pi, pi]; [batch_size, numobj]
    return np.expand_dims(ang_rad_pi, axis=2) # [batch_size, numobj, 1]



def torch_normalize(v):
    """ Returns a copy of v normalized along the last dimension, same shape as input."""
    norm = torch.unsqueeze(torch.linalg.norm(v, axis=-1), -1) # (:,:,...,1) (same # of dim) as v
    return v/norm

def torch_angle_between(v1, v2):
    """ v1 and v2: [batch_size, numobj, 2], last dim = (cos(th), sin(th))
        Returns angles in [-pi, pi] from v1 to v2 (order matters), of shape [batch_size, numobj, 1].
    """
    # NOTE: only use case in train_pointnet.py (already normalized)
    # v1 = torch_normalize(v1)
    # v2 = torch_normalize(v2)
    cross_prod = v1[:,:,0]*v2[:,:,1] - v1[:,:,1]*v2[:,:,0] # sin; [batch_size, numobj]
    inner_prod = (v1 * v2).sum(dim=2) # cos; [batch_size, numobj]
    ang_rad_pi = torch.atan2(cross_prod, inner_prod)  # [-pi, pi]; [batch_size, numobj]
    return torch.unsqueeze(ang_rad_pi, 2) # [batch_size, numobj, 1]

def torch_angle(v):
    """ v1: [batch_size, numobj, 2], along the last dim = (cos(th), sin(th))
        Returns angles in [-pi, pi] from positive x-axis to v, of shape [batch_size, numobj, 1].
    """
    # NOTE: only use case in train_pointnet.py (already normalized)
    # v = torch_normalize(v) 
    ang_rad_pi = torch.atan2(v[:,:,1], v[:,:,0]) # [-pi, pi]; [batch_size, numobj]
    return torch.unsqueeze(ang_rad_pi, 2) # [batch_size, numobj, 1]





## From ConDor
def h5_to_dictionary(h5_file_path):
    f = h5py.File(h5_file_path, "r")
    assert "id" in f.keys(), "No ID keyword found in h5 file"
    output_dictionary = {}
    model_ids = f["id"].asstr()
    for i in range(len(model_ids)):
        mesh_id = model_ids[i]
        mesh_dictionary = {}
        for key in f.keys():
            if key != "id":
                mesh_dictionary[key] = f[key][i]
        output_dictionary[mesh_id] = mesh_dictionary
        
    return output_dictionary
    



## VISUALIZATION

# simple 3D visualization set up
import trimesh
import trimesh.transformations as t

# Models from 3D FUTRUE
chair_jid = "75fd3650-4824-4ab2-81f8-1cc9423c2b56"
chair_bbox = [0.5561800003051758, 0.7970709991455078, 0.6548519897460937] # y up convention, original size from json
#  {"jid": "75fd3650-4824-4ab2-81f8-1cc9423c2b56", "uid": "159106489565995680/model", "aid": [], "category": "Lounge Chair / Book-chair / Computer Chair", "bbox": [0.5561800003051758, 0.7970709991455078, 0.6548519897460937], "valid": true}
rectable_jid = "fc93abbf-984c-432a-abb7-7f1ce951955f"
rectable_bbox = [2.5, 0.7522930145263672, 1.1]
# {"jid": "fc93abbf-984c-432a-abb7-7f1ce951955f", "uid": "159102138805663680/model", "aid": [], "category": "Dining Table", "bbox": [2.5, 0.7522930145263672, 1.1], "valid": true},
roundtable_jid = "5b3eb52a-e3ed-4f13-bcb7-2564bf39d34b"
roundtable_bbox =  [1.2, 0.7388890075683594, 1.2]
# {"jid": "5b3eb52a-e3ed-4f13-bcb7-2564bf39d34b", "uid": "159371160936073312/model", "aid": [], "category": "Dining Table", "bbox": [1.2, 0.7388890075683594, 1.2], "valid": true}
chair2_jid = "0d831b81-3ec1-4f8c-bb27-e9ebd83ec3b4"
chair2_bbox =  [0.4598830032348633, 0.7535749816894531, 0.5532799911499023]
# {"jid": "0d831b81-3ec1-4f8c-bb27-e9ebd83ec3b4", "uid": "159400933126177504/model", "aid": [], "category": "Dining Chair", "bbox": [0.4598830032348633, 0.7535749816894531, 0.5532799911499023], "valid": true}



## VISUALIZATION: prior experiments - 1 type of objects (chair)
def scatter(idx, vals, target):
    """target[idx] += vals, but allowing for repeats in idx"""
    np.add.at(target, idx.ravel(), vals.ravel())

def visualize_2d_pointcloud_res(x, res, fp='visualize_2d_pointcloud_text.jpg'):
    """width and height: res*2. Top left is (0,0) 
       x: has shape (6,2) """
    img = np.zeros((4*res*res))
    idx = np.rint(res*x) + res #(6,2)
    idx = idx[..., 0] + 2*res*idx[..., 1] # flattened array index
    idx = np.ndarray.astype(idx, int)  # (6,)
    scatter(idx, np.ones((idx.shape[0],)), img)
    img = np.reshape(img, (2*res, 2*res))
    plt.imsave(fp, img,  dpi=300)


def visualize_heat_map(x, model, res, batch_size, j):
    """Example: x = generate_6_points_batch(batch_size=batch_size)"""
    heat_map = np.zeros((4*res*res))
    t = 0
    r = np.arange(batch_size)
    for i in range((4*res*res) // batch_size):
        print(f"\n\n{i}")
        idx = r + t
        t0 = np.mod(r + t, 2*res)
        t1 = (r + t - t0) // (2*res)
        x_ = np.stack([t0, t1], axis=-1)
        x_ = (np.ndarray.astype(x_, np.float) - res)/res
        x[:, j, :] = x_
        y = model(x)
        y = y[:, 0]
        y = np.array(y)
        scatter(idx, y, heat_map)
        t += batch_size
    img = np.reshape(heat_map, (2 * res, 2 * res))
    plt.imsave(os.path.join(f'visualize_heat_map.jpg'), img, dpi=300)

def visualize_2d_pointcloud(xy, fp='visualize_2d_pointcloud_eval.jpg', title='Assignment'):
    """ Graph a set of points.
        xy: numpy array of shape (numpt,2), positions of points to visualize """
    fig = plt.figure(dpi=300, figsize=(5,5))
    plt.scatter(xy[:,0], xy[:,1]) # red
    for i in range(xy.shape[0]):
        plt.text(xy[i,0]-0.2, xy[i,1]+0.1, "[{:0.3f}, {:0.3f}]".format(xy[i,0], xy[i,1]), size='small')
    plt.gca().set(title=title, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    plt.savefig(fp)
    plt.close(fig)


def visualize_2d_pointcloud_eval(xy_1, xy_2, data_type, label_1="input", label_2="prediction", 
        fp='visualize_2d_pointcloud_eval.jpg', title='Assignment'):
    """ Graph 2 sets of corresponding points, with arrows from xy_1 to xy_2.
        xy_1: has shape (numpt,>=2), ex: input position to network
        xy_2: has shape (numpt,>=2), ex: prediction outputted by network  """
    diff = xy_2-xy_1

    fig = plt.figure(dpi=300, figsize=(5,5))
    if data_type=="rect":
        gt = generate_6_points() # ground truth: (6,2)
        plt.scatter(gt[:,0], gt[:,1], c="#2ca02c", label="ground truth", alpha=0.5) # green
    plt.scatter(xy_1[:,0], xy_1[:,1], c="#1f77b4", label=label_1) # blue
    plt.scatter(xy_2[:,0], xy_2[:,1], c="#d62728", label=label_2) # red

    for pt_i in range(xy_1.shape[0]):
        plt.arrow(xy_1[pt_i, 0], xy_1[pt_i, 1], diff[pt_i, 0], diff[pt_i, 1], 
            length_includes_head=True, head_length=0.02, head_width=0.05, fc='k', ec='k')
    
    plt.legend()
    plt.gca().set(title=title, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    plt.savefig(fp)
    plt.close(fig)




def visualize_chair_denoise(traj, data_type, numobj=None, fp='visualize_chair_denoise.jpg', title='Denoise'):
    """ Graph trajectory of each point.
        traj: numpy array with shape (iter,numpt,2), containing trajectory of objects.

        Called in train.py as default case.
    """

    fig = plt.figure(dpi=300, figsize=(5,5))
    handles = []
    numobj = traj.shape[1]
    if data_type=="tablechair_circle": numobj=find_numobj(traj[0,:,4:6])

    # initial & final/ground truth
    initial = plt.scatter(x=traj[0,:numobj,0], y=traj[0,:numobj,1], c="#1f77b4", label="Initial") # blue
    handles.append(initial)
    if data_type=="rect":
        gt = generate_6_points() # ground truth: (6,2)
        gt = plt.scatter(gt[:,0], gt[:,1], c="#2ca02c", label="Ground Truth", alpha=0.5) # green
        handles.append(gt)
    else:
        final = plt.scatter(x=traj[-1,:numobj,0], y=traj[-1,:numobj,1], c="#d62728", label="Final") # red
        handles.append(final)
    # trajectory
    for obj_i in range(numobj):
        h, = plt.plot(traj[:,obj_i,0], traj[:,obj_i,1], label=f"Obj {obj_i}", c="#000000")
    # orientation
    if traj.shape[2] >2:
        plt.quiver(traj[0,:,0], traj[0,:,1], traj[0,:,2], traj[0,:,3], color="#1f77b4", label="initial")
        plt.quiver(traj[-1,:,0], traj[-1,:,1], traj[-1,:,2], traj[-1,:,3], color="#d62728", label="final")
    # circles
    if data_type=="circle":
        center = tuple(np.mean(traj[-1],axis=0)) # based on final
        plt.scatter(x=center[0], y=center[1], c="#000000", label="Center")
        plt.gca().add_patch(plt.Circle(center, radius=0.5, fill = False, alpha=0.3))
        for obj_i in range(numobj):
            plt.text(traj[0,obj_i,0]-0.02, traj[0,obj_i,1]-0.07, f"{obj_i}", size='small')
            plt.text(traj[-1,obj_i,0]-0.02, traj[-1,obj_i,1]-0.07, f"{obj_i}", size='small')

    plt.legend(handles=handles)
    plt.gca().set(title=title, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    plt.savefig(fp)
    plt.close(fig)


def visualize_2d_pointcloud_3d(final_pos, data_type,  obj_fp, final_angle=None):
    """ final_pos: numpy array with shape (numpt,2), containing trajectory of objects.
        Called in train.py as default case.
    """
    final_pos = final_pos * 10 # Correspondance roughly 0.1 in code -> 1 here
    scene = trimesh.Scene()
    meshes = []
    for i in range(final_pos.shape[0]):
        mesh = trimesh.Trimesh.copy(c_mesh_model)
        mesh.apply_transform(trimesh.transformations.translation_matrix([final_pos[i,0], final_pos[i,1], 0]))
        if final_angle is not None:
            final_angle = np_normalize(final_angle)
            ang = trig2ang(final_angle[i:i+1,:])
            mesh.apply_transform(t.rotation_matrix(ang, [0, 0, 1], [final_pos[i,0], final_pos[i,1], 0]))
        meshes.append(mesh)
        # print("\n", mesh.center_mass)
    scene.add_geometry(meshes)

    # with open(f"{int(datetime.now().timestamp())}_pos.png", 'wb') as f:
    #     f.write(scene.save_image(resolution=[1920, 1080]))
    trimesh.exchange.export.export_scene(scene, obj_fp, file_type='obj', include_texture=True)
    # scene.show()



## VISULIZATION: 2 types of objects (table, chair)
def find_numobj(sha):
    """sha: shape [14,2].
       Returns the number of objects in the scene that corresponds to ang
    """
    for i in range(sha.shape[0]):
        if sha[i,0]==0 and sha[i,1]==0:
            return i
    return sha.shape[0]

def visualize_tablechair_simple(posang, numobj=14, fp='visualize_tablechair_simple.jpg', title='visualize_tablechair_simple'):
    """ posang has shape (1,14,>=2). Visualize 14 points with their angles (if provided). """

    if numobj is None: numobj = find_numobj(posang[0,:,4:6])
    
    fig = plt.figure(dpi=300, figsize=(5,5))
    table_initial = plt.scatter(x=posang[0,:2,0], y=posang[0,:2,1], c="#2ca02c", label="table") # green
    chair_initial = plt.scatter(x=posang[0,2:numobj,0], y=posang[0,2:numobj,1], c="#1f77b4", label=f"chair") # blue

    if posang.shape[2]>2:
        plt.quiver(posang[0,:2,0], posang[0,:2,1], posang[0,:2,2], posang[0,:2,3], color="#2ca02c", label="table")
        plt.quiver(posang[0,2:numobj,0], posang[0,2:numobj,1], posang[0,2:numobj,2], posang[0,2:numobj,3], color="#1f77b4", label="chair")

    plt.legend(handles=[table_initial, chair_initial])
    plt.gca().set(title=title, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    plt.savefig(fp)
    plt.close(fig)


def visualize_tablechair_eval(xy_1, xy_2, numobj=14, data_type="", label_1="input", label_2="prediction", 
        fp='visualize_tablechair_eval.jpg', title='Assignment'):
    """ Graph 2 sets of tables and chairs, with arrows from xy_1 to xy_2.
        xy_1: has shape (numpt,6), ex: input position to network
        xy_2: has shape (numpt,6), ex: prediction outputted by network  
    """
    if numobj is None: numobj = find_numobj(xy_1[:,4:6]) # generated input

    fig = plt.figure(dpi=300, figsize=(5,5))

    plt.scatter(xy_1[0:2,0], xy_1[0:2,1], c="#2ca02c", label=f"table {label_1}") # green
    plt.scatter(xy_2[0:2,0], xy_2[0:2,1], c="#8934eb", label=f"table {label_2}") # purple
    if data_type=="tablechair_circle":
        plt.gca().add_patch(plt.Circle((xy_2[0,0], xy_2[0,1]), radius=0.25, fill = False, alpha=0.3))
        plt.gca().add_patch(plt.Circle((xy_2[1,0], xy_2[1,1]), radius=0.25, fill = False, alpha=0.3))

    plt.scatter(xy_1[2:numobj,0], xy_1[2:numobj,1], c="#1f77b4", label=f"chair {label_1}") # blue
    plt.scatter(xy_2[2:numobj,0], xy_2[2:numobj,1], c="#d62728", label=f"chair {label_2}") # red
    
    diff = xy_2[:numobj,:2]-xy_1[:numobj,:2]
    for pt_i in range(numobj):
        plt.arrow(xy_1[pt_i, 0], xy_1[pt_i, 1], diff[pt_i, 0], diff[pt_i, 1], 
            length_includes_head=True, head_length=0.02, head_width=0.05, fc='k', ec='k')
    
    plt.legend()
    plt.gca().set(title=title, xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])
    plt.savefig(fp)
    plt.close(fig)



def visualize_tablechair_denoise(traj, data_type, n_table=2, numobj=14, fp='visualize_tablechair_denoise.jpg', title='Denoise'):
    """ Graph trajectory of tables and chairs. For "tablechair_horizontal", "tablechair_circle", "tablechair_shape"
        traj: numpy array with shape (iter, numobj, pos+ang+sha), containing trajectory of objects. 
        
        Called in train.py for 3 tablechair data types.
    """
    if numobj is None: numobj=find_numobj(traj[0,:,6:8]) # initial=generated input ("tablechair_circle")

    fig = plt.figure(dpi=300, figsize=(5,5))
    # initial & final
    table_initial = plt.scatter(x=traj[0,:n_table,0], y=traj[0,:n_table,1],  c="#8934eb", label="table initial") # purple
    table_final = plt.scatter(x=traj[-1,:n_table,0], y=traj[-1,:n_table,1], c="#2ca02c", label=f"table final") # green
    if data_type in ["tablechair_horizontal", "tablechair_circle"]:
        chair_initial = plt.scatter(x=traj[0,n_table:numobj,0], y=traj[0,n_table:numobj,1], c="#d62728", label=f"chair initial") # red
        chair_final =plt.scatter(x=traj[-1,n_table:numobj,0], y=traj[-1,n_table:numobj,1], c="#1f77b4", label=f"chair final") # blue
        handles = [table_initial, table_final, chair_initial, chair_final]
    if data_type=="tablechair_shape":
        chair1_initial = plt.scatter(x=traj[0,n_table:4,0], y=traj[0,n_table:4,1], c="#d62728", label=f"chair 1 initial") # red
        chair1_final =plt.scatter(x=traj[-1,n_table:4,0], y=traj[-1,n_table:4,1], c="#1f77b4", label=f"chair 1 final") # blue
        chair2_initial = plt.scatter(x=traj[0,4:numobj,0], y=traj[0,4:numobj, 1], c="#f45431", label=f"chair 2 initial") # orange
        chair2_final =plt.scatter(x=traj[-1,4:numobj,0], y=traj[-1,4:numobj,1], c="#1fafb4", label=f"chair 2 final") # cyan
        handles = [table_initial, table_final, chair1_initial, chair1_final, chair2_initial, chair2_final]
    
    # trajectory
    for o_i in range(numobj):
        c="#2ca02c" if o_i < n_table else "#1f77b4" # initial colors
        if data_type=="tablechair_shape" and o_i >=4: c = "#1fafb4" 
        h, = plt.plot(traj[:,o_i,0], traj[:,o_i,1], label=f"Obj {o_i}", c=c, alpha=0.4)
    
    # orientation
    if traj.shape[2]>2:
        arrows_initial = np_rotate(traj[0,:,2:4], np.zeros((traj.shape[1],1))+np.pi/2)
        arrow_final = np_rotate(traj[-1,:,2:4], np.zeros((traj.shape[1],1))+np.pi/2) 
        plt.quiver(traj[0,:n_table,0], traj[0,:n_table,1], arrows_initial[:n_table,0], arrows_initial[:n_table,1], color="#8934eb", label="table initial")
        plt.quiver(traj[-1,:n_table,0], traj[-1,:n_table,1], arrow_final[:n_table,0], arrow_final[:n_table,1], color="#2ca02c", label="table final")
        
        if data_type in ["tablechair_horizontal", "tablechair_circle"]:
            plt.quiver(traj[0,n_table:numobj,0], traj[0,n_table:numobj,1], arrows_initial[n_table:numobj,0], arrows_initial[n_table:numobj,1], color="#d62728", label="chair initial")
            plt.quiver(traj[-1,n_table:numobj,0], traj[-1,n_table:numobj,1], arrow_final[n_table:numobj,0], arrow_final[n_table:numobj,1], color="#1f77b4", label="chair final")
        if data_type=="tablechair_shape":
            plt.quiver(traj[0,n_table:4,0], traj[0,n_table:4,1], arrows_initial[n_table:4,0], arrows_initial[n_table:4,1], color="#d62728", label="chair 1 initial")
            plt.quiver(traj[-1,n_table:4,0], traj[-1,n_table:4,1], arrow_final[n_table:4,0], arrow_final[n_table:4,1], color="#1f77b4", label="chair 1 final")
            plt.quiver(traj[0,4:numobj,0], traj[0,4:numobj,1], arrows_initial[4:numobj,0], arrows_initial[4:numobj,1], color="#f45431", label="chair 2 initial")
            plt.quiver(traj[-1,4:numobj,0], traj[-1,4:numobj,1], arrow_final[4:numobj,0], arrow_final[4:numobj,1], color="#1fafb4", label="chair 2 final")

    # rectangles and circles
    scene = traj[-1]
    siz = scene[:,4:6] 
    for o_i in range(traj.shape[1]):
        c="#2ca02c" if o_i < n_table else "#1f77b4"
        if data_type=="tablechair_shape" and o_i >=4: c = "#1fafb4"

        if data_type=="tablechair_circle" and o_i < n_table: 
            plt.gca().add_patch(plt.Circle((scene[o_i,0], scene[o_i,1]), radius=scene[o_i,4], fill = False, edgecolor=c))
        else: # recttables and chairs
            plt.gca().add_patch(plt.Rectangle(
                            (scene[o_i,0]-siz[o_i,0]/2, scene[o_i,1]-siz[o_i,1]/2), 
                            siz[o_i,0], siz[o_i,1], linewidth=1, edgecolor=c, facecolor='none',
                            transform = mt.Affine2D().rotate_around(scene[o_i,0], scene[o_i,1], np.arctan2(scene[o_i,3], scene[o_i,2])) +plt.gca().transData 
                        ))

    plt.legend(handles=handles, fontsize=7)
    plt.gca().set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2])
    plt.title(f"{title}", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    plt.savefig(fp)
    plt.close(fig)


def visualize_tablechair_3d(final_pos, final_angle=None, initial_sha=None, numobj=14, obj_fp="visualize_tablechair_3d.jpg"):
    """ Old/simple way of 3d visualization.
        final_pos, final_angle, initial_sha: [14, 2], assume first 2 rows are tables, and the 2:numobj are chair positions.
    """
    if numobj is None: numobj=find_numobj(initial_sha) # using initial (generated input)
    if final_angle is not None: final_angle = np_normalize(final_angle)
    
    final_pos = final_pos * 10 # Correspondance roughly 0.1 in code -> 1 here
    scene = trimesh.Scene()
    meshes = []
    for i in range(numobj):
        mesh = trimesh.Trimesh.copy(t_mesh_model) if i < 2 else trimesh.Trimesh.copy(c_mesh_model)
        # TODO: get round table
        mesh.apply_transform(trimesh.transformations.translation_matrix([final_pos[i,0], final_pos[i,1], 0]))
        if final_angle is not None:
            ang = trig2ang(final_angle[i:i+1,:])
            mesh.apply_transform(t.rotation_matrix(ang, [0, 0, 1], [final_pos[i,0], final_pos[i,1], 0]))
        else:
            if final_pos[i, 1] < np.mean(final_pos, axis=0)[1]: # below the avg y coord
                mesh.apply_transform(t.rotation_matrix(math.pi, [0, 0, 1], [final_pos[i,0], final_pos[i,1], 0]))
        meshes.append(mesh)
        # print("\n", mesh.center_mass)
    scene.add_geometry(meshes)

    # with open(f"{int(datetime.now().timestamp())}_pos.png", 'wb') as f:
    #     f.write(scene.save_image(resolution=[1920, 1080]))

    trimesh.exchange.export.export_scene(scene, obj_fp, file_type='obj', include_texture=True)

    # scene.show()
