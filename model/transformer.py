import torch
import torch.nn as nn
from model.floorplan_encoder import PointNet_Line, ResNet18, PointNet_Point

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        self.freq_bands = freq_bands

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim, embedder_obj.freq_bands


class FixedPositionalEncoding(nn.Module):
    def __init__(self, numfreq, end=10):
        super().__init__()
        """for each frequency we have num_coord*2 (cos and sin) terms"""
        exb = torch.linspace(0, numfreq-1, numfreq) / (numfreq-1) #  [0, ..., 15]/15
        self.sigma = torch.pow(end, exb).view(1, -1)  # (1x16)
        # for end=10: [[1=10^(0/15), 1.1659=10^(1/15)..., 10=10^(15/15)]] # geometric with factor 10^(1/15)
            # tensor([[ 1.0000,  1.1659,  1.3594,  1.5849,  1.8478,  2.1544,  2.5119,  2.9286,
            #           3.4145,  3.9811,  4.6416,  5.4117,  6.3096,  7.3564,  8.5770, 10.0000]])
            # divide above tensor by 2 -> frequency (num of periods in [0,1])
        self.sigma = torch.pi * self.sigma # (1x16)
        # NOTE: have the first sigma term be pi ( so that cos(pi*norm_ang) = cos(theta) )

        # ORIGINAL:
            # [0, 2, ..., 30] / 32
            # [1=10^(0/32), 1.1548=10^(2/32) ..., 8.6596=10^(30/32)] # geometric with factor 10^(2/32)
            # 2pi(10^(2/32))^0, 2pi(10^(2/32))^1, ..., 2pi(10^(2/32))^15
            # tensor([[ 6.2832,  7.2557,  8.3788,  9.6756, 11.1733, 12.9027, 14.8998, 17.2060,
            #          19.8692, 22.9446, 26.4960, 30.5971, 35.3329, 40.8019, 47.1172, 54.4101]])

    def forward(self, x):
        # x: B x N x 2
        B,N,_ = x.shape # _=2/4
        x = x.unsqueeze(-1)  # B x N x 2/4 x 1
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1).reshape(B,N,-1)  # B x N x 2/4 x (16+16) --> B x N x 64/128


class Transformer(nn.Module):
    def __init__(self, d_model= 512, nhead= 8, num_encoder_layers= 6,
                 dim_feedforward= 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x, padding_mask=None):
        # batch x sequence x feature (N, S, E)
        return self.encoder(x, src_key_padding_mask=padding_mask)
        # src_key_padding_mask: (N,S)
            # If a BoolTensor is provided, positions with True not allowed to attend while False values will be unchanged: [False ..., True, ...]


class TransformerWrapper(nn.Module):
    """"""
    def __init__(self, pos_dim=2, ang_dim=2, siz_dim=2, cla_dim=19, maxnfpoc=25, nfpbpn=250, invsha_d=128, use_invariant_shape = False,
                 ang_initial_d = 128, siz_initial_unit = [64, 16], cla_initial_unit = [64], invsha_initial_unit = [128,128],
                 all_initial_unit = [512], final_lin_unit = [256, 4], use_two_branch = False,
                 pe_numfreq= 16, pe_end=128, use_floorplan = False, floorplan_encoder_type = 'pointnet',
                 
                 nhead= 8, num_encoder_layers= 6, dim_feedforward= 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5, batch_first: bool = True):
        """ * For each object's input: pos, ang - PE; siz - either PE or MLP; cla - MLP. 
            * Then the result from these 4 are concatenated (B x nobj x total_d) and passed into MLP all_initial. 
            * Finally, the (B x nobj x d_model) from all_initial is concatenated with floor plan to get
              (B x nobj+1 x d_model) and passed into the transformer (encoder only).
            * The transformer outputs (B x nobj+1 x d_model), which is then passed to all_final to produce
             (B x nobj+1 x out_dim) [:,:-1,:] to disregard floor plan.
                
            {}_unit: contains output dimension of each layer, input dim to first layer based on other params.  
            out_dim: final_lin_unit[-1]
            transformer_input_d: original d_model=512, feat_d for each obj, taken in by the transformer and outputted by the transformer. 
        """
        super().__init__()
        self.pos_dim, self.ang_dim, self.siz_dim, self.cla_dim = pos_dim, ang_dim, siz_dim, cla_dim
        all_initial_unit = [ e for e in all_initial_unit] # make copy as changed below
        transformer_input_d = all_initial_unit[-1]

        self.use_floorplan, self.floorplan_encoder_type, self.use_two_branch, self.use_invariant_shape = use_floorplan, floorplan_encoder_type, use_two_branch, use_invariant_shape
        if use_floorplan: all_initial_unit[-1]-=1 
            # dmodel to transformer must be divisible by numhead: save space for flag distinguishing floor plan token from obj token

        self.activation = torch.nn.LeakyReLU() # ReLU()
        self.pe = FixedPositionalEncoding(pe_numfreq, pe_end) 
        
        # 1a. pos (B x nobj x pos_d) -> (B x nobj x 2*pe_numfreq*(pos_dim)=128)

        # 1b: ang: (B x nobj x ang_d//2) -> (B x nobj x 2*pe_numfreq*(ang_dim//2) -> 128)
        self.ang_initial = torch.nn.Linear(2*pe_numfreq*ang_dim//2, ang_initial_d)

        # 1c. siz: (B x nobj x siz_d) -> (B x nobj x siz_initial_feat_d=128)
        if siz_initial_unit is None: # Positional encoding on size
            self.siz_initial = None
            siz_initial_feat_d = 2*pe_numfreq*siz_dim 
        else: # assume nonempty
            self.siz_initial = [torch.nn.Linear(siz_dim, siz_initial_unit[0])]
            for i in range(1, len(siz_initial_unit)):
                self.siz_initial.append(torch.nn.Linear(siz_initial_unit[i-1], siz_initial_unit[i]))
            self.siz_initial = torch.nn.ModuleList(self.siz_initial)
            siz_initial_feat_d = siz_initial_unit[-1]

        # 1d. cla: (B x nobj x cla_d) -> (B x nobj x cla_initial_unit[-1]=128)
        self.cla_initial = [torch.nn.Linear(cla_dim, cla_initial_unit[0])]
        for i in range(1, len(cla_initial_unit)):
            self.cla_initial.append(torch.nn.Linear(cla_initial_unit[i-1], cla_initial_unit[i]))
        self.cla_initial = torch.nn.ModuleList(self.cla_initial)
        
        # 1e: invsha (B x nobj x invsha_d) -> (B x nobj x 128) = invariant shape feature for object
        inv_sha_initial_d = 0
        if self.use_invariant_shape: 
            inv_sha_initial_d = invsha_initial_unit[-1]
            self.invsha_initial = [torch.nn.Linear(invsha_d, invsha_initial_unit[0])]
            for i in range(1, len(invsha_initial_unit)):
                self.invsha_initial.append(torch.nn.Linear(invsha_initial_unit[i-1], invsha_initial_unit[i]))
            self.invsha_initial = torch.nn.ModuleList(self.invsha_initial)

        # 2a. (B x nobj x initial_feat_input_d=128*5) -> (B x nobj x transformer_input_d) 
        initial_feat_input_d = 2*pe_numfreq*(pos_dim) + ang_initial_d + siz_initial_feat_d + cla_initial_unit[-1] + inv_sha_initial_d
        self.all_initial = [torch.nn.Linear(initial_feat_input_d, all_initial_unit[0])]
        for i in range(1, len(all_initial_unit)):
            self.all_initial.append(torch.nn.Linear(all_initial_unit[i-1], all_initial_unit[i]))
        self.all_initial = torch.nn.ModuleList(self.all_initial)

        # 2b. floor plan
        self.floorplan_encoder = None
        if self.use_floorplan: # self.floorplan = PointNetClass_Simple(input_dim=max, out_dim)
            if self.floorplan_encoder_type == 'pointnet':
                corner_feat_units = [self.pos_dim, 64, 128] # 2 layers
                line_feat_units = [corner_feat_units[-1]*2, 512, 1024]  # 2 layers
                fp_units = [line_feat_units[-1], 512, transformer_input_d-1] # 1024, 511/more (1 layer)

                self.floorplan_encoder = PointNet_Line(torch.nn.LeakyReLU(), maxnfpoc=maxnfpoc, corner_feat_units=corner_feat_units, line_feat_units=line_feat_units, fp_units=fp_units)
        
            elif self.floorplan_encoder_type == 'resnet':
                self.floorplan_encoder = ResNet18(freeze_bn=False, in_dim=3, out_dim=transformer_input_d-1)

            elif self.floorplan_encoder_type == 'pointnet_simple':
                self.floorplan_encoder = PointNet_Point(torch.nn.LeakyReLU(), nfpbpn=nfpbpn, feat_units=[4, 64, 64, 512, transformer_input_d-1])

        # 3. (B x nobj(+1) x transformer_input_d) ->  (B x nobj(+1) x transformer_input_d) 
        self.transformer = Transformer(transformer_input_d, nhead, num_encoder_layers, dim_feedforward, dropout, layer_norm_eps, batch_first)

        # 4. (B x nobj(+1) x transformer_input_d) -> (B x nobj x out_dim=pos_dim+ang_dim)
        if not self.use_two_branch:
            final_lin = [torch.nn.Linear(transformer_input_d, final_lin_unit[0])]
            for i in range(1, len(final_lin_unit)):
                final_lin.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
            self.final_lin = torch.nn.ModuleList(final_lin)
        else:
            final_lin_pos = [torch.nn.Linear(transformer_input_d, final_lin_unit[0])]
            final_lin_ang = [torch.nn.Linear(transformer_input_d, final_lin_unit[0])]
            for i in range(1, len(final_lin_unit)-1):
                final_lin_pos.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
                final_lin_ang.append(torch.nn.Linear(final_lin_unit[i-1], final_lin_unit[i]))
            final_lin_pos.append(torch.nn.Linear(final_lin_unit[-2], self.pos_dim))
            final_lin_ang.append(torch.nn.Linear(final_lin_unit[-2], self.ang_dim))

            self.final_lin_pos = torch.nn.ModuleList(final_lin_pos)
            self.final_lin_ang = torch.nn.ModuleList(final_lin_ang)



    def forward(self, x, padding_mask, device, fpoc=None, nfpc=None, fpmask=None, fpbpn=None):
        """ x           : [batch_size, maxnumobj, pos+ang+siz+cla]
            padding_mask: [batch_size, maxnumobj], for nn.TransformerEncoder (False: not masked, True=masked, not attended to)
            fpoc        : [batch_size, maxnfpoc, pos=2], with padded 0 beyond num_floor_plan_ordered_corners for each scene, for 'pointnet'
            nfpc        : [batch_size], for 'pointnet'
            fpmask      : [batch_size, 256, 256, 3], for 'resnet'
            fpbpn       : [batch_size, nfpbp=250, 4], for 'pointnet_simple'
        """

        # print(f"TransformerWrapper: x.shape={x.shape}")
        # 1a. pos (B x nobj x pos_d) -> (B x nobj x 2*pe_numfreq*(pos_dim)=128)
        pos = self.pe(x[:, :, :self.pos_dim])

        # 1b: ang: (B x nobj x ang_d//2) -> (B x nobj x 2*pe_numfreq*(ang_dim//2) -> 128)
        ang_rad_pi = torch.unsqueeze(torch.atan2(x[:,:,self.pos_dim+1], x[:,:,self.pos_dim]), 2) # [B, numobj, 1], [-pi, pi]
        ang = self.pe(ang_rad_pi/torch.pi) # in [-1, 1]
        ang = self.ang_initial(ang)

        # 1c. siz: (B x nobj x siz_d) ->  (B x nobj x siz_initial_feat_d=128)
        siz = x[:,:, self.pos_dim+self.ang_dim : self.pos_dim+self.ang_dim+self.siz_dim]
        if self.siz_initial is None:
            siz = self.pe(siz) # [B, nobj, pe_terms=2*2freq]
        else:
            for i in range(len(self.siz_initial)-1):
                siz = self.activation(self.siz_initial[i](siz)) # changes last dim
            siz = self.siz_initial[-1](siz)
            
        # 1d. cla: (B x nobj x cla_d) ->  (B x nobj x cla_initial_unit[-1]=128)
        cla =  x[:,:, self.pos_dim+self.ang_dim+self.siz_dim:self.pos_dim+self.ang_dim+self.siz_dim+self.cla_dim]
        for i in range(len(self.cla_initial)-1):
            cla = self.activation(self.cla_initial[i](cla)) # [B, nobj, cla_feat_d]
        cla = self.cla_initial[-1](cla)

        # 1e: invsha (B x nobj x invsha_d=128) -> (B x nobj x 128)   
        if self.use_invariant_shape:  
            invsha = x[:,:, self.pos_dim+self.ang_dim+self.siz_dim+self.cla_dim:]
            for i in range(len(self.invsha_initial)-1):
                invsha = self.activation(self.invsha_initial[i](invsha)) # [B, nobj, cla_feat_d]
            invsha = self.invsha_initial[-1](invsha)

        # 2a. (B x nobj x initial_feat_input_d=128*5) -> (B x nobj x transformer_input_d=d_model(-1)) 
        initial_feat = torch.cat([pos, ang, siz, cla], dim=-1)
        if self.use_invariant_shape: initial_feat = torch.cat([initial_feat, invsha], dim=-1)
        for i in range(len(self.all_initial)-1):
            initial_feat = self.activation(self.all_initial[i](initial_feat)) 
        initial_feat = self.all_initial[-1](initial_feat) # [B, nobj, 512(-1)]

        # 2b. floor plan: input -> (B x 1 x transformer_input_d=512(-1+1) )
        # Add floor plan as last obj along dim 1 + add binary flag as last col along dim 2 + add 1 entry to padding mask for floor plan object
        if self.use_floorplan:
            if self.floorplan_encoder_type == 'pointnet': # (B x nfpoc x 2 -> 128 -> 128*2 -> 512 -> 1024) -> (B x 1024 -> 512/511)
                scene_fp_input = self.floorplan_encoder(fpoc, nfpc, device)
                    
            elif self.floorplan_encoder_type == 'resnet':
                fpmask = fpmask.permute((0,3,1,2))  # [B, H, W, C] -> [B, C, H, W] 
                scene_fp_input = torch.unsqueeze(self.floorplan_encoder(fpmask), dim=1) # [B, None->1, transformer_input_d-1=511]

            elif self.floorplan_encoder_type == 'pointnet_simple':
                scene_fp_input = self.floorplan_encoder(fpbpn, device)  # [B, None->1, transformer_input_d-1]

            # Prepare overall input to transformer
            initial_feat = torch.cat([initial_feat, scene_fp_input], dim=1) # [B, nobj+1, transformer_input_d-1]
            flag = torch.zeros(initial_feat.shape[0], initial_feat.shape[1], 1).to(device) # [B, nobj+1, 1], is_floorplan 
            # NOTE: flag always apended at the end, with or without shape feature
            flag[:, -1, 0] = 1 # last row of every scene is 1
            initial_feat = torch.cat((initial_feat, flag), dim=-1) # [B, nobj+1, transformer_input_d-1+1 = transformer_input_d]
            
            # mask for transformer (False: not masked, True=masked/not attended to)
            padding_mask = torch.cat((padding_mask, torch.zeros(initial_feat.shape[0],1).to(device)), dim=1).bool()# [batch_size, maxnumobj+1]
        

        # 3. (B x nobj(+1) x transformer_input_d) ->  (B x nobj(+1) x transformer_input_d) 
        trans_out = self.transformer(initial_feat, padding_mask=padding_mask)

        # 4. (B x nobj(+1) x transformer_input_d) -> (B x nobj x out_dim) 
        if self.use_floorplan: trans_out = trans_out[:,:-1,:] # (B x nobj x transformer_input_d)
        
        if not self.use_two_branch:
            for i in range(len(self.final_lin)-1):
                trans_out = self.activation(self.final_lin[i](trans_out))
            out = self.final_lin[-1](trans_out) # (B x nobj x out_dim) 
            
            if out.shape[2] == 1: return torch.sigmoid(out) # classifier (classificaiton token is the last entry along dim1)
            else: return out
        else:
            pos_out = trans_out
            for i in range(len(self.final_lin_pos)-1):
                pos_out = self.activation(self.final_lin_pos[i](pos_out))
            pos_out = self.final_lin_pos[-1](pos_out) # (B x nobj x pos_dim) 

            ang_out = trans_out
            for i in range(len(self.final_lin_ang)-1):
                ang_out = self.activation(self.final_lin_ang[i](ang_out))
            ang_out = self.final_lin_ang[-1](ang_out) # (B x nobj x ang_dim) 
            
            return torch.cat([pos_out, ang_out], dim=2) #(B x nobj x pos_dim+ang_dim) 

 

class TransformerWrapper_Simple(nn.Module):
    def __init__(self, pos_dim=2, ang_dim=2, point_feat_dim=2, out_dim=2, 
                 pe_numfreq= 16, d_model= 512, nhead= 8, num_encoder_layers= 6,
                 dim_feedforward= 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        self.pos_dim = pos_dim
        self.ang_dim = ang_dim

        self.pe = FixedPositionalEncoding(pe_numfreq) 

        pe_term = pos_dim+ang_dim//2 # ang_dim either 0 or 2
        pe_dim = 2*pe_numfreq*pe_term
        self.initial_feat = nn.Linear(pe_dim + point_feat_dim, d_model)
        self.final_lin = nn.Linear(d_model, out_dim) 
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, layer_norm_eps, batch_first)
        

    def forward(self, x, padding_mask):
        # (B, numobj, 6), numobj <=14
        pos = x[:, :, :self.pos_dim] # [B, numobj, pos_dim]
        pe_input = pos
        if self.ang_dim > 0:
            ang_rad_pi = torch.unsqueeze(torch.atan2(x[:,:,self.pos_dim+1], x[:,:,self.pos_dim]), 2) # [B, numobj, 1], [-pi, pi]
            pe_input = torch.cat([pos, ang_rad_pi/torch.pi], dim=-1) # [B, numobj, pos_dim+1=3], all in [-1,1]

        input_feat = x[:,:,self.pos_dim+self.ang_dim:] # [B x numobj x point_feat_dim]
        input_x = torch.cat([self.pe(pe_input), input_feat], dim=-1)  # B x numobj x pe_dim+point_feat_dim
        input_x = self.initial_feat(input_x)  # B x numobj x d_model=512
        return self.final_lin(self.transformer(input_x, padding_mask=padding_mask))


class TransformerWrapper_SimpleSize(nn.Module):
    """Also pass siz in [pos, ang, siz, cla] into PE (compared to only passing pos+ang)."""
    def __init__(self, pos_dim=2, ang_dim=2, siz_dim=2, point_feat_dim=2, out_dim=2, 
                 pe_numfreq= 16, d_model= 512, nhead= 8, num_encoder_layers= 6,
                 dim_feedforward= 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        self.pos_dim = pos_dim
        self.ang_dim = ang_dim
        self.siz_dim = siz_dim

        self.pe = FixedPositionalEncoding(pe_numfreq) 

        pe_term = pos_dim+ang_dim//2+siz_dim # ang_dim either 0 or 2
        pe_dim = 2*pe_numfreq*pe_term #5*32 = 160
        self.initial_feat = nn.Linear(pe_dim + point_feat_dim, d_model) 
        self.final_lin = nn.Linear(d_model, out_dim) 
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, layer_norm_eps, batch_first)


    def forward(self, x, padding_mask):
        pos = x[:, :, :self.pos_dim] # [B, numobj, pos_dim]
        siz = x[:, :, self.pos_dim+self.ang_dim:self.pos_dim+self.ang_dim+self.siz_dim] 
        ang_rad_pi = torch.unsqueeze(torch.atan2(x[:,:,self.pos_dim+1], x[:,:,self.pos_dim]), 2) # [B, numobj, 1], [-pi, pi]
       
        pe_input = torch.cat([pos, ang_rad_pi/torch.pi, siz], dim=-1) # [B, numobj, 2+1+2=5], all in [-1,1]

        input_feat = x[:,:,self.pos_dim+self.ang_dim+self.siz_dim:] # [B x numobj x point_feat_dim]
        input_x = torch.cat([self.pe(pe_input), input_feat], dim=-1)  # B x numobj x pe_dim+point_feat_dim
        input_x = self.initial_feat(input_x)  # B x numobj x d_model=512
        return self.final_lin(self.transformer(input_x, padding_mask=padding_mask))


if __name__ == "__main__":
    # pe = FixedPositionalEncoding(32)
    maxnfpoc=25
    transformer_encoder = TransformerWrapper(use_floorplan=True, maxnfpoc=maxnfpoc, use_invariant_shape=True)
    src = torch.rand(32, 10, 25+128)
    padding_mask = torch.rand(32, 10)
    fpoc = torch.rand(32, maxnfpoc, 2) # [batch_size, maxnfpv, pos]
    nfpc = torch.randint(4, maxnfpoc, (32,)) 
    out = transformer_encoder(src, padding_mask, 'cpu', fpoc=fpoc, nfpc=nfpc)
    print(out.shape)


    # encoder_layer = nn.TransformerEncoderLayer(d_model=2, nhead=1, batch_first=True)
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    # src = torch.tensor([[ [1.,2.], [3.,4.], [5.,6.]]])
    # print(src)
    # src_key_padding_mask=torch.tensor([[False, True, True]])
    # out = transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    # print(f"{src_key_padding_mask}: {out}")
