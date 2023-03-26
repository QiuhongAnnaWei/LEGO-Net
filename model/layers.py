import torch
import numpy as np

# Code borrowed and edited from: https://raw.githubusercontent.com/yanx27/Pointnet_Pointnet2_pytorch/master/models/pointnet2_utils.py



def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(radius, nsample, xyz, points, returnfps=False, subtract_feats = False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    # S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # new_xyz = index_points(xyz, fps_idx)
    new_xyz = xyz
    S = N
    # Sampling all neighbors
    idx = query_ball_point(radius, xyz.shape[1] - 1, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # [B, N, N-1, F=C=2], each obj's neighborhood's relative coord

    if points is not None:
        grouped_points = index_points(points, idx) # B, N, N-1, F=2or4
        if subtract_feats:
            # print("subtracting")
            grouped_points = grouped_points - points.unsqueeze(-2) # [B, N, N-1, F] - [B, N, 1, F]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        # relative coordinates, relative features
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(torch.nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, group_all, subtract_feats = False):
        super(PointNetSetAbstraction, self).__init__()
        # self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = torch.nn.ModuleList()
        self.mlp_bns = torch.nn.ModuleList()
        last_channel = in_channel
        self.subtract_feats = subtract_feats
        for out_channel in mlp:
            self.mlp_convs.append(torch.nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(torch.nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.radius, self.nsample, xyz, points, subtract_feats = self.subtract_feats)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample=N-1, C+D=(relative coord+feat)]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        # print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = torch.nn.functional.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0] # B, F, nsample, npoint -> B, F, npoint
        new_xyz = new_xyz.permute(0, 2, 1) # B, N, F
        return new_points




class Embedding(torch.nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class Attention_block(torch.nn.Module):

    def __init__(self, in_channels):
        super(Attention_block, self).__init__()

        self.f = torch.nn.Linear(in_channels, max(in_channels // 8, 1))
        self.g = torch.nn.Linear(in_channels, max(in_channels // 8, 1))
        self.h = torch.nn.Linear(in_channels, in_channels)

        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        x - B, N, D
        """

        f = self.f(x)  # B, in // 8, N
        g = self.g(x)  # B, in // 8, N
        h = self.h(x).permute((0, 2, 1))  # B, out, N

        attention = torch.einsum("bpd, bkd-> bpk", f, g)  # B, N, N
        attention = h @ self.softmax(attention)  # B, out, N
        attention = attention.permute((0, 2, 1))

        out = self.gamma * attention + x

        return out


class MLP_layer(torch.nn.Module):
    def __init__(self, input, output, normalize = True, activation = None):
        super(MLP_layer, self).__init__()

        self.mlp = torch.nn.Linear(input, output)
        
        self.normalize = normalize
        if normalize:
            self.bn = torch.nn.BatchNorm1d(output)

        self.activation = activation
    
    def forward(self, x):
        
        # print(x.shape, "check")
        out = self.mlp(x)
        
        if self.normalize:
            out = self.bn(out.transpose(-1, 1)).transpose(-1, 1)
        
        if self.activation is not None:
            out = self.activation(out)
        # print(out.shape, "check 2")
        
        return out
    
class MLP_stacked(torch.nn.Module):
    def __init__(self, input, mlp_units = [20, 40, 60], normalize = True, activation = None):
        super(MLP_stacked, self).__init__()
        

        self.mlps = []
        for unit_idx in range(len(mlp_units)):
            if unit_idx == 0:
                self.mlps.append(MLP_layer(input, mlp_units[unit_idx], normalize = normalize, activation = activation))
            else:
                # print(mlp_units[unit_idx], mlp_units[unit_idx + 1])
                self.mlps.append(MLP_layer(mlp_units[unit_idx - 1], mlp_units[unit_idx], normalize = normalize, activation = activation))

        self.mlps = torch.nn.Sequential(*self.mlps)
    
    def forward(self, x):

        out = self.mlps(x)
        
        
        return out
    


if __name__ == "__main__":

    pts = torch.randn(2, 50, 2).permute(0, 2, 1)
    feats = torch.randn(2, 50, 15).permute(0, 2, 1)
    # print(pts.shape)

    point_net_layer = PointNetSetAbstraction(radius = 0.2, nsample = 32, in_channel = 15 + 2, mlp = [6, 12], group_all = False)

    out = point_net_layer(pts, feats)
    # print(out[0].shape, out[1].shape)
    # print(out[0] - pts)