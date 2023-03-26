import os, sys
sys.path.insert(1, os.getcwd())

import torch

from model.layers import Attention_block, Embedding, PointNetSetAbstraction, MLP_stacked
# from model.point_transformer import PointTransformerSeg, PointTransformerBlock
from model.transformer import TransformerWrapper


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.fc_1 = torch.nn.Linear(49,128)
        # self.bn1 = torch.nn.BatchNorm2d(100) #numscenex100x1x1 -> numscenex1x1
        self.fc_2 = torch.nn.Linear(128,128)
        self.fc_3 = torch.nn.Linear(128,128)
        self.fc_4 = torch.nn.Linear(128,1)
        self.sigmoid = torch.nn.Sigmoid() # using [0,1]
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)   # numscene,numobj=7,7 -> numscene, 49
        x = self.fc_1(x)
        x = self.activation(x)
        # x = torch.unsqueeze(x, -1)
        # x = torch.unsqueeze(x, -1) # [200, 100, 1, 1]
        # x = self.bn1(x)
        # x = torch.squeeze(x) # [200, 100]
        x = self.fc_2(x)
        x = self.activation(x)
        x = self.fc_3(x)
        x = self.activation(x)
        x = self.fc_4(x)
        x = self.sigmoid(x)
        return x


class PointNetClass(torch.nn.Module):
    """Pointnet classification network"""
    def __init__(self, input_dim=7, out_dim=1):
        super(PointNetClass, self).__init__()
        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.units = [64,64,64,128,1024,512,256,64]
        self.maxpool= torch.nn.MaxPool1d(7)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid() # using [0,1]
        self.linear_layers = []
        self.bn_layers = []
        for i in range(len(self.units)):
            self.bn_layers.append(torch.nn.BatchNorm1d(self.units[i]))
            if i == 0:
                self.linear_layers.append(torch.nn.Linear(input_dim, self.units[i]))
            else:
                self.linear_layers.append(torch.nn.Linear(self.units[i-1], self.units[i]))
        self.linear_layers.append(torch.nn.Linear(self.units[-1], out_dim))
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)
    
    def forward(self, x):
        """Input x is normalized and of dimension [numscene,numobj=7,featperobj=7=pose_d+shape_d]"""
        
        for i in range(len(self.linear_layers)):

            if i == 5: # x = numscene, numobj, 1024
                x = x.permute((0,2,1))
                x = self.maxpool(x)
                x = x.reshape(x.size(0),-1) # numscene, 1024

            if i < 5:
                x = self.linear_layers[i](x) # numscene,numobj=7,featperobj=7 -> numscene, numobj, 64
                x = x.permute((0,2,1)) # numscene, 64 (number of features), numobj (sequence length)
                x = self.bn_layers[i](x)
                x = x.permute((0,2,1)) # numscene, numobj, 64
                x = self.activation(x)
            elif i == len(self.linear_layers)-1: 
                x = self.dropout(x)
                x = self.linear_layers[i](x)
            else:
                x = self.linear_layers[i](x)
                x = self.bn_layers[i](x)
                x = self.activation(x)
            
        x = self.sigmoid(x)
        
        return x


class PointNetSeg(torch.nn.Module):
    """Pointnet segmentation network"""
    def __init__(self, input_dim=2, out_dim=2, num_obj=6, point_and_gloabl_feat=True):
        super(PointNetSeg, self).__init__()
        self.point_and_gloabl_feat = point_and_gloabl_feat

        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.units = [64,64,64,128,1024,512,256,64] # last=128 in paper
        self.maxpool= torch.nn.MaxPool1d(num_obj)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid() # using [0,1]
        self.linear_layers = []
        self.bn_layers = []
        for i in range(len(self.units)):
            self.bn_layers.append(torch.nn.BatchNorm1d(self.units[i]))
            if i == 0:
                self.linear_layers.append(torch.nn.Linear(input_dim, self.units[i]))
            elif i == 5:
                seg_in = self.units[4]+self.units[1] if self.point_and_gloabl_feat else self.units[4]
                self.linear_layers.append(torch.nn.Linear(seg_in, self.units[i]))
            else:
                self.linear_layers.append(torch.nn.Linear(self.units[i-1], self.units[i]))
        self.linear_layers.append(torch.nn.Linear(self.units[-1], out_dim))
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)
    
    def forward(self, x):
        """Input x is normalized and of dimension [numscene,numobj=7,featperobj=7=pose_d+shape_d]"""
        pointfeat = None
        for i in range(len(self.linear_layers)):
            if i == 5: # x = numscene, numobj, 1024
                numscene, numobj, feat_dim = x.size(dim=0), x.size(dim=1), x.size(dim=2)
                x = x.permute((0,2,1))
                x = self.maxpool(x) # numscene, 1024, numobj->1
                x = x.reshape(numscene, feat_dim) # numscene, 1024
                x = x.reshape(numscene, 1, feat_dim).repeat(1, numobj, 1) # numscene, 1->numobj, 1024
                if self.point_and_gloabl_feat: # global-feature + point-feature
                    x =  torch.cat([x, pointfeat], 2) #numscene, numobj, 1024+64=1088
            
            if i == len(self.linear_layers)-1: 
                x = self.dropout(x)
                x = self.linear_layers[i](x) # numscene, numobj, out_dim=m
            else:
                x = self.linear_layers[i](x) # numscene,numobj=7,featperobj=7 -> numscene, numobj, 64
                x = x.permute((0,2,1)) # numscene, 64 (number of features), numobj (sequence length)
                x = self.bn_layers[i](x)
                x = x.permute((0,2,1)) # numscene, numobj, 64
                x = self.activation(x)
                if i==1: # after second layer 
                    pointfeat = x # numscene, numobj, 64

        return x


class PointNetClass_transform(torch.nn.Module):
    def __init__(self):
        super(PointNetClass_transform, self).__init__()
        self.input_transform = transform(7)
        self.feature_transform = transform(64)
        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.units = [64,64,64,128,1024, 512, 256, 64]
        self.maxpool= torch.nn.MaxPool1d(7)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid() # using [0,1]
        self.linear_layers = []
        self.bn_layers = []
        for i in range(len(self.units)):
            self.bn_layers.append(torch.nn.BatchNorm1d(self.units[i]))
            if i == 0:
                self.linear_layers.append(torch.nn.Linear(7, self.units[i]))
            else:
                self.linear_layers.append(torch.nn.Linear(self.units[i-1], self.units[i]))
        self.linear_layers.append(torch.nn.Linear(self.units[-1], 1))
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)
    
    def forward(self, x):
        """Input x is normalized and of dimension [numscene,numobj=7,featperobj=7=pose_d+shape_d]"""
        # x = x [:,1:,:2]
        input_transform = self.input_transform(x)
        x = torch.matmul(x, input_transform)
        for i in range(len(self.linear_layers)):
            if i == 2:
                feature_transform = self.feature_transform(x)
                x = torch.matmul(x, feature_transform)
            if i == 5:
                x = x.permute((0,2,1))
                x = self.maxpool(x)
                x = x.reshape(x.size(0),-1)
            if i < 5:
                x = self.linear_layers[i](x) # numscene,numobj=7,featperobj=7 -> numscene, numobj, 64
                x = x.permute((0,2,1)) # numscene, 64, numobj
                x = self.bn_layers[i](x) # numscene, 64 (number of features), numobj (sequence length)
                x = x.permute((0,2,1)) # numscene, numobj, 64
                x = self.activation(x)
            elif i == len(self.linear_layers)-1: 
                x = self.dropout(x)
                x = self.linear_layers[i](x)
            # elif i == len(self.linear_layers)-2: 
            #     x = self.dropout(x)
            #     x = self.linear_layers[i](x)
            #     x = self.bn_layers[i](x)
            #     x = self.activation(x)
            else:
                x = self.linear_layers[i](x)
                x = self.bn_layers[i](x)
                x = self.activation(x)
            
        x = self.sigmoid(x)
        
        return x

class transform(torch.nn.Module):
    def __init__(self,k):
        super(transform, self).__init__()
        self.k = k
        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.units = [64,128,1024, 512, 256]
        self.maxpool= torch.nn.MaxPool1d(6)
        self.linear_layers = []
        self.bn_layers = []
        for i in range(len(self.units)):
            self.bn_layers.append(torch.nn.BatchNorm1d(self.units[i]))
            if i == 0:
                self.linear_layers.append(torch.nn.Linear(k, self.units[i]))
            else:
                self.linear_layers.append(torch.nn.Linear(self.units[i-1], self.units[i]))
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.bn_layers = torch.nn.ModuleList(self.bn_layers)
    def forward(self, x):
        """Input x is normalized and of dimension [numscene,numobj=7,featperobj=7=pose_d+shape_d]"""
        for i in range(len(self.linear_layers)):
            if i == 3:
                x = x.permute((0,2,1))
                x = self.maxpool(x)
                x = x.reshape(x.size(0),-1)
            if i < 3:
                x = self.linear_layers[i](x) # numscene,numobj=7,featperobj=7 -> numscene, numobj, 64
                x = x.permute((0,2,1)) # numscene, 64, numobj
                x = self.bn_layers[i](x) # numscene, 64 (number of features), numobj (sequence length)
                x = x.permute((0,2,1)) # numscene, numobj, 64
                x = self.activation(x)
            else:
                x = self.linear_layers[i](x)
                x = self.bn_layers[i](x)
                x = self.activation(x)
            
        weights = torch.zeros(self.units[-1],self.k*self.k, requires_grad=True)
        bias = torch.eye(self.k, requires_grad=True).flatten()
        
        x = torch.matmul(x, weights)
        x = torch.add(x, bias)
        x = x.reshape(x.size(0), self.k, self.k)
        return x




class Discriminator_attention_positional_position(torch.nn.Module):
    """positional_position: xy coords alone, no angle"""
    def __init__(self, input_dim=2, out_dim=2):
        super(Discriminator_attention_positional_position, self).__init__()
        self.bn = torch.nn.BatchNorm1d(60)
        self.activation = torch.nn.ReLU()
        self.fc_1_1 = torch.nn.Linear(42, 60)
        self.fc_2 = Attention_block(60)
        self.fc_3 = torch.nn.Linear(60*2, out_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()  # using [0,1]
        self.positional_embedding = Embedding(in_channels = input_dim, N_freqs = 10, )

    def forward(self, x_pose):
        # x_pose - B, N, 2
        x = self.positional_embedding(x_pose)
        # x_pose - B, N, 42 = 2 * 10 * 2 + 2
        x = self.fc_1_1(x)
        # print(x.shape)
        x = self.activation(x)
        x = x.permute(0, -1, 1)
        x = self.bn(x)
        x = x.permute(0, -1, 1)
        # x = self.dropout(x)
        x = self.fc_2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x_global = torch.max(x, dim=1).values
        x = torch.cat([x, x_global.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x = self.fc_3(x)
        # x = self.sigmoid(x)
        # print(x.shape)

        return x


class PointNetPlusPlus_attention(torch.nn.Module):
    def __init__(self, input_dim=2, out_dim=2):
        super(PointNetPlusPlus_attention, self).__init__()
        self.bn = torch.nn.BatchNorm1d(60)
        self.bn_2 = torch.nn.BatchNorm1d(60)
        self.activation = torch.nn.ReLU()
        self.fc_1_1 = torch.nn.Linear(42, 60)
        # self.fc_2 = Attention_block(60)
        self.fc_3 = PointNetSetAbstraction(radius = 10000.0, nsample = 4, in_channel = 60 * 2 + input_dim, mlp =[30, 60], group_all = False)
        self.fc_4 = torch.nn.Linear(60, out_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()  # using [0,1]
        self.positional_embedding = Embedding(
            in_channels=input_dim, N_freqs=10)

    def forward(self, x_pose):
        # x_pose - B, N, 2
        x = self.positional_embedding(x_pose)
        # x_pose - B, N, 42 = 2 * 10 * 2 + 2
        x = self.fc_1_1(x)
        # print(x.shape)
        x = self.activation(x)
        x = x.permute(0, -1, 1)
        x = self.bn(x)
        x = x.permute(0, -1, 1)
        # x = self.dropout(x)
        # x = self.fc_2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x_global = torch.max(x, dim=1).values
        x = torch.cat([x, x_global.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x = x.permute(0, -1, 1)
        x_pose_in = x_pose.permute(0, -1, 1)
        x = self.fc_3(x_pose_in, x)
        x = self.bn_2(x)
        x = x.permute(0, -1, 1)
        x = self.activation(x)
        x = self.fc_4(x)
        # x = self.sigmoid(x)
        # print(x.shape)

        return x
    

class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, input_dim=2, out_dim=2, shape_code = False, shape_dim = 2):
        super(PointNetPlusPlus, self).__init__()
        self.bn = torch.nn.BatchNorm1d(60)
        self.bn_2 = torch.nn.BatchNorm1d(60)
        self.activation = torch.nn.ReLU()
        self.shape_code = shape_code
        if shape_code:
            self.shape_dim = shape_dim
            self.fc_1_1 = torch.nn.Linear(42 + shape_dim, 60)
        else:
            self.fc_1_1 = torch.nn.Linear(42, 60)
        # self.fc_2 = Attention_block(60)
        self.fc_3 = PointNetSetAbstraction(radius = 1.0, nsample = 4, in_channel = 60 * 2 + input_dim, mlp =[30, 60], group_all = False, subtract_feats = False)
        self.fc_4 = torch.nn.Linear(60, out_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()  # using [0,1]
        self.positional_embedding = Embedding(
            in_channels=input_dim, N_freqs=10)

    def forward(self, x):
        
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        if self.shape_code:
            x_shape = x[..., -self.shape_dim:]
        
        x_pose = x[..., :2]
        
        # x_pose - B, N, 2
        x = self.positional_embedding(x_pose)
        # x_pose - B, N, 42 = 2 * 10 * 2 + 2
        if self.shape_code:
            # print(x.shape)
            x = torch.cat([x, x_shape], axis = -1)
            # print(x.shape)
            # x_s = self.fc_1_2(x_shape)
        x = self.fc_1_1(x)
        # x = torch.cat([x, x_s], axis = -1)
        # print(x.shape)
        x = self.activation(x)
        x = x.permute(0, -1, 1)
        x = self.bn(x)
        x = x.permute(0, -1, 1)
        # x = self.dropout(x)
        # x = self.fc_2(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        x_global = torch.max(x, dim=1).values
        x = torch.cat([x, x_global.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x = x.permute(0, -1, 1)
        x_pose_in = x_pose.permute(0, -1, 1)
        x = self.fc_3(x_pose_in, x)
        x = self.bn_2(x)
        x = x.permute(0, -1, 1)
        x = self.activation(x)
        x = self.fc_4(x)
        # x = self.sigmoid(x)
        # print(x.shape)

        return x

class PointNetPlusPlus_dense(torch.nn.Module):
    def __init__(self, input_dim=2, out_dim=2, shape_code = False, shape_dim = 2, subtract_feats = False):
        super(PointNetPlusPlus_dense, self).__init__()
        self.activation = torch.nn.ReLU()
        self.shape_code = shape_code
        self.mlp_units = [20, 40, 80, 60]
        if shape_code:
            self.shape_dim = shape_dim
            self.fc_1_1 = MLP_stacked(42 + shape_dim, self.mlp_units, normalize = True, activation = self.activation)
            # self.fc_1_1 = torch.nn.Linear(42 + shape_dim, 60)
        else:
            self.fc_1_1 = MLP_stacked(42, self.mlp_units, normalize = True, activation = self.activation)

        self.fc_3 = PointNetSetAbstraction(radius = 1.0, nsample = 4, in_channel = self.mlp_units[-1] * 2 + input_dim, mlp =[30, 60], group_all = False, subtract_feats = subtract_feats)
        self.fc_4 = MLP_stacked(60, self.mlp_units, normalize = True, activation = self.activation)
        self.fc_5 = torch.nn.Linear(self.mlp_units[-1], out_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()  # using [0,1]
        self.positional_embedding = Embedding(
            in_channels=input_dim, N_freqs=10)

    def forward(self, x):
        
        # print(x.device)
        # x = x.permute(0, 2, 1)
        if self.shape_code:
            x_shape = x[..., -self.shape_dim:]
        
        x_pose = x[..., :2]
        
        # x_pose - B, N, 2
        x = self.positional_embedding(x_pose)
        # x_pose - B, N, 42 = 2 * 10 * 2 + 2
        if self.shape_code:
            # print(x.shape)
            x = torch.cat([x, x_shape], axis = -1)
        # print(x.shape, "network dense")
        x = self.fc_1_1(x)
        x_global = torch.max(x, dim=1).values
        x = torch.cat([x, x_global.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x = x.permute(0, -1, 1)
        x_pose_in = x_pose.permute(0, -1, 1)
        x = self.fc_3(x_pose_in, x) # x=points=features
        # print(x.shape, "pointnet++")
        x = x.permute(0, -1, 1)
        x = self.fc_4(x)
        x = self.fc_5(x)

        return x


class PointNetPlusPlus_dense_attention(torch.nn.Module):
    def __init__(self, input_dim=2, out_dim=2, shape_code = False, shape_dim = 2, subtract_feats = False):
        super(PointNetPlusPlus_dense_attention, self).__init__()
        self.activation = torch.nn.ReLU()
        self.shape_code = shape_code
        self.mlp_units = [20, 40, 80, 60]
        if shape_code:
            self.shape_dim = shape_dim
            self.fc_1_1 = MLP_stacked(42 + shape_dim, self.mlp_units, normalize = True, activation = self.activation)
            # self.fc_1_1 = torch.nn.Linear(42 + shape_dim, 60)
        else:
            self.fc_1_1 = MLP_stacked(42, self.mlp_units, normalize = True, activation = self.activation)

        self.fc_3 = PointNetSetAbstraction(radius = 1.0, nsample = 4, in_channel = self.mlp_units[-1] * 2 + input_dim, mlp =[30, 60], group_all = False, subtract_feats = subtract_feats)
        self.fc_4 = MLP_stacked(60, self.mlp_units, normalize = True, activation = self.activation)
        self.fc_5 = Attention_block(self.mlp_units[-1])
        self.fc_6 = torch.nn.Linear(self.mlp_units[-1], out_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.sigmoid = torch.nn.Sigmoid()  # using [0,1]
        self.positional_embedding = Embedding(
            in_channels=input_dim, N_freqs=10)

    def forward(self, x):
        
        # print(x.device)
        # x = x.permute(0, 2, 1)
        if self.shape_code:
            x_shape = x[..., -self.shape_dim:]
        
        x_pose = x[..., :2]
        
        # x_pose - B, N, 2
        x = self.positional_embedding(x_pose)
        # x_pose - B, N, 42 = 2 * 10 * 2 + 2
        if self.shape_code:
            # print(x.shape)
            x = torch.cat([x, x_shape], axis = -1)
        # print(x.shape, "network dense")
        x = self.fc_1_1(x)
        x_global = torch.max(x, dim=1).values
        x = torch.cat([x, x_global.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x = x.permute(0, -1, 1)
        x_pose_in = x_pose.permute(0, -1, 1)
        x = self.fc_3(x_pose_in, x)
        # print(x.shape, "pointnet++")
        x = x.permute(0, -1, 1)
        x = self.fc_4(x)
        x = self.fc_5(x)
        x = self.fc_6(x)

        return x


class PointTransformer(torch.nn.Module):
    """Point Transformer network"""
    def __init__(self, input_dim=2, out_dim=2, device="cuda:0"):
        super(PointTransformer, self).__init__()
        self.PointTransformer = PointTransformerSeg(input_dim, out_dim)
        self.device = device
        
    
    def forward(self, pxo):
        # (B, 14, 4)
        p = pxo[:,:,:2]
        p = torch.nn.functional.pad(p, (0,1), "constant", 0)
        # (B, 14, 3)
        
        x = pxo 
        o = []
        for i in range(p.size(0)):
            
            o.append(p.size(1)*(i+1))
        o = torch.tensor(o, dtype=torch.int ,device=self.device).expand(2,-1)
        x = self.PointTransformer([p,x,o])
        
        return x


if __name__ == "__main__":

    x = torch.randn(2, 100, 2)
    net = PointNetPlusPlus_attention()
    out = net(x)
    # print(out.shape)