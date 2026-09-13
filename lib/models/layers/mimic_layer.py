from torch import nn
import torch
from timm.models.layers import to_2tuple, Mlp, DropPath
from lib.models.layers.attn import Linear_Attention
from lib.models.layers.attn_blocks import Block
from lib.models.layers.adapter import Bi_direct_adapter
# from mamba_ssm import Mamba


class MIMICLayer(nn.Module):
    def __init__(self, dim=8, xavier_init=False, act_layer=nn.GELU):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim, bias=True)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # self.act = act_layer()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        # x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up


class MIMICLayerV2(nn.Module):
    def __init__(self, dim=768, act_layer=nn.Tanh, norm_layer=nn.LayerNorm):
        super().__init__()

        self.linear = nn.Linear(dim, dim, bias=True)  
        self.norm = norm_layer(dim) 

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear.weight)

        self.act = act_layer()
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear(self.norm(x))   
        x = self.act(x)
        return x
    

class DynamicMlp(nn.Module):
    """ DynamicMlp
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            layer_num=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)
        # drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.seq = None
        if layer_num > 1:
            self.seq = nn.Sequential(*[
                nn.Sequential(
                    linear_layer(in_features, in_features, bias=bias),
                    act_layer(),
                    nn.Dropout(drop),
                    norm_layer(in_features) if norm_layer is not None else nn.Identity()
                ) for _ in range(layer_num-1)]
            )

        self.fc = linear_layer(in_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.seq:
            x = self.seq(x)
        x = self.fc(x)
        x = self.drop(x)
        return x


class AttnMlp(nn.Module):
    def __init__(
            self,
            num_tokens,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()

        self.linear1 = nn.Linear(num_tokens, num_tokens, bias=bias)
        # self.linear2 = nn.Linear(num_tokens, num_tokens, bias=bias)
        # self.act = act_layer()
        self.drop = nn.Dropout(drop)
        # self.norm = norm_layer(num_tokens) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.linear1(x).softmax(dim=-1)
        # x = x.transpose(-1,-2)
        # x = self.drop(self.linear2(x)).transpose(-1,-2)
        return self.drop(x)

class DynamicMlpMimicAttn(nn.Module):
    """ DynamicMlp_MimicAttn
    """
    def __init__(
            self,
            num_tokens=320,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            layer_num=1,
    ):
        super().__init__()
        self.seq = nn.Sequential(*[
            AttnMlp(num_tokens, act_layer, norm_layer, bias, drop) for _ in range(layer_num)]
        )

    def forward(self, x):
        x = self.seq(x)
        return x
    

class DynamicLoRa(nn.Module):
    """ dynamic_lora 差的一批
    """
    def __init__(
            self,
            layer_num=1,
    ):
        super().__init__()

        self.seq = nn.Sequential(*[
            Bi_direct_adapter() for _ in range(layer_num)]
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class CrossMlp(nn.Module):
    def __init__(
            self,
            num_tokens,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.extract_linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.ext_norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()

        self.cross_linear = nn.Linear(num_tokens, num_tokens, bias=bias)
        # self.cross_norm = norm_layer(num_tokens) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.ext_norm(self.drop(self.act(self.extract_linear(x))))
        x = x.transpose(-1,-2)
        # cross
        x = self.cross_linear(x)
        # x = self.drop(x)
        return x.transpose(-1,-2)

class DynamicCrossMlp(nn.Module):
    """ DynamicCrossMlp
    """
    def __init__(
            self,
            num_tokens,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            layer_num=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.seq = None
        if layer_num > 1:
            self.seq = nn.Sequential(*[
                CrossMlp(
                    num_tokens, in_features, hidden_features, out_features, act_layer, norm_layer, bias, drop
                ) for _ in range(layer_num-1)]
            )

        self.last_mlp = CrossMlp(num_tokens, in_features, hidden_features, out_features, act_layer, norm_layer, bias, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.seq:
            x = self.seq(x)
        x = self.last_mlp(x)
        x = self.drop(x)
        return x


class DynamicViT(nn.Module):
    """ DynamicViT
    """
    def __init__(
            self, embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer,
            layer_num=1,
    ):
        super().__init__()
        self.seq = nn.Sequential(*[
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer
            ) for _ in range(layer_num)]
        )

    def forward(self, x):
        x = self.seq(x)
        return x
    
class DynamicMlp_DM(nn.Module):
    """ DynamicMlp_DistillMedium
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            layer_num=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.seq = None
        if layer_num > 1:
            self.seq = nn.Sequential(*[
                nn.Sequential(
                    linear_layer(in_features, in_features, bias=bias),
                    act_layer(),
                    nn.Dropout(drop),
                    norm_layer(in_features) if norm_layer is not None else nn.Identity()
                ) for _ in range(layer_num-1)]
            )

        self.fc = linear_layer(in_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        medium_fea = []
        if self.seq:
            for layer in self.seq:
                x = layer(x)
                medium_fea.append(x)
        x = self.fc(x)
        x = self.drop(x)
        return x, medium_fea


class Mimic_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Linear_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class DynamicMMBlock(nn.Module):
    """ DynamicMMBlock
    """
    def __init__(
            self,
            in_features,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            layer_num=1,
    ):
        super().__init__()

        # self.seq = None
        self.seq = nn.Sequential(*[
            Mimic_Block(
                dim=in_features, num_heads=8, attn_drop=attn_drop, drop=drop
            ) for _ in range(layer_num)]
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Policy(nn.Module):
    '''two_conv
    '''
    def __init__(self, n_observations=768, n_actions=1, search_size=16):
        super().__init__()

        self.conv1 = nn.Conv2d(n_observations, n_observations//2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 尺寸减半
        conv1_outsize = (search_size + 2*3 -7) // 2 + 1
        self.conv2 = nn.Conv2d(n_observations//2, n_actions, kernel_size=(conv1_outsize, conv1_outsize)) # 尺寸缩为1*1
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()

    def forward(self, x, lens_x=256):
        B = x.shape[0]

        w = int(lens_x**0.5)
        s = x[:,-lens_x:].transpose(-1,-2).reshape(B, -1, w, w)
        
        s = self.act1(self.conv1(s))
        s = self.act2(self.conv2(s)).squeeze(-1).squeeze(-1)
        return s
    
class Policy_v2(nn.Module):

    def __init__(self, n_observations, n_actions=1):
        """
        Args:
            n_observations (int): _description_
            n_actions (int, optional): number of actions, i.e. exit or continue. Defaults to 1.
        """
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, n_actions)
        self.gelu = nn.GELU()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.gelu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

class IOUPredictor(nn.Module):
    '''two_conv
    '''
    def __init__(self, embed_dim=768, search_size=16):
        super().__init__()

        self.conv1 = nn.Conv2d(embed_dim, embed_dim//2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 尺寸减半
        conv1_outsize = (search_size + 2*3 -7) // 2 + 1
        self.conv2 = nn.Conv2d(embed_dim//2, 1, kernel_size=(conv1_outsize, conv1_outsize)) # 尺寸缩为1*1
        self.act = nn.Sigmoid()

    def forward(self, x, lens_x=256):
        B = x.shape[0]

        w = int(lens_x**0.5)
        s = x[:,-lens_x:].transpose(-1,-2).reshape(B, -1, w, w)
        
        s = self.act(self.conv1(s))
        s = self.act(self.conv2(s)).squeeze(-1).squeeze(-1)
        return s
    
class IOUPredictor2(nn.Module):
    def __init__(self, embed_dim=768, search_size=16):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim//2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 尺寸减半
        conv1_outsize = (search_size + 2*3 -7) // 2 + 1
        self.conv2 = nn.Conv2d(embed_dim//2, 1, kernel_size=(conv1_outsize, conv1_outsize)) # 尺寸缩为1*1
        self.act = nn.Sigmoid()

    def forward(self, x, lens_x=256):
        x = self.norm(x)
        B = x.shape[0]

        w = int(lens_x**0.5)
        s = x[:,-lens_x:].transpose(-1,-2).reshape(B, -1, w, w)
        
        s = self.act(self.conv1(s))
        s = self.act(self.conv2(s)).squeeze(-1).squeeze(-1)
        return s
    
class IOUPredictor_v2(nn.Module):
    ''' v2: two_linear
    '''
    def __init__(self, embed_dim=768, search_size=16):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(search_size**2, 1)
        self.act = nn.GELU()
        self.act2 = nn.ReLU()

    def forward(self, x, lens_x=256):        
        x = self.act(self.linear(x[:,-lens_x:]).squeeze(-1))
        x = self.act2(self.linear2(x))
        return x

class IOUPredictor_v3(nn.Module):
    ''' v3: two_linear_sigmoid
    '''
    def __init__(self, embed_dim=768, search_size=16):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(search_size**2, 1)
        self.act = nn.GELU()
        self.act2 = nn.Sigmoid()

    def forward(self, x, lens_x=256):        
        x = self.act(self.linear(x[:,-lens_x:]).squeeze(-1))
        x = self.act2(self.linear2(x))
        return x
    
class IOUPredictor_Linear_Sig(nn.Module):
    ''' v4: Linear_Sig
    '''
    def __init__(self, embed_dim=768):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):        
        x = self.act(self.linear(x))
        return x
    
class IOUPredictor_Linear_Sig_avgdim(nn.Module):
    ''' v4: linear_sig_avgdim
    '''
    def __init__(self, token_num=320):
        super().__init__()

        self.linear = nn.Linear(token_num, 1)
        self.act = nn.Sigmoid()

    def forward(self, x, lens_x):
        x = x.mean(-1).squeeze(-1)
        x = self.act(self.linear(x))
        return x
    
class IOUPredictor_Linear_Sig_Dim(nn.Module):
    ''' v4: Linear_Sig_dim
    '''
    def __init__(self, token_num=320):
        super().__init__()

        self.linear = nn.Linear(token_num, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):        
        x = self.act(self.linear(x))
        return x

class IOUPredictor_Linear_Sig_Avg_Dim(nn.Module):
    ''' v4: Linear_Sig_Avg_Dim
    '''
    def __init__(self, token_num=320):
        super().__init__()

        self.linear = nn.Linear(token_num, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.mean(-1)
        x = self.act(self.linear(x))
        return x

class IOUPredictor_Linear_Sig_Cat(nn.Module):
    ''' v4: Linear_Sig_Cat
    '''
    def __init__(self, embed_dim=768):
        super().__init__()

        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(embed_dim, 1)
        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(-1,-2)
        x = self.act1(self.linear1(x)).transpose(-1,-2)
        x = self.act2(self.linear2(x)).squeeze(-1)
        return x
    
class IOUPredictor_v4(nn.Module):
    ''' v4: two_linear_sigmoid_full
    '''
    def __init__(self, embed_dim=768, token_num=320):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(token_num, 1)
        self.act = nn.GELU()
        self.act2 = nn.Sigmoid()

    def forward(self, x, lens_x):        
        x = self.act(self.linear(x).squeeze(-1))
        x = self.act2(self.linear2(x))
        return x
    
    
class ModEEPredictor(nn.Module):
    ''' linear_v1
        320*768     320*768
        320*1       320*1
                3*1
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(token_num*2, type_num)
        self.act = nn.Sigmoid()
        self.use_softmax = use_softmax
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x, xi):        
        x = self.act(self.linear(x))
        xi = self.act(self.linear(xi))
        if self.use_softmax:
            res = self.act2(self.linear2(torch.cat((x,xi),dim=1).squeeze(-1)))
        else:
            res = self.linear2(torch.cat((x,xi),dim=1).squeeze(-1))
        return res
    
class ModEEPredictor_token(nn.Module):
    ''' linear_v1
        2*768
        2*1
        3*1
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(2, type_num)
        self.act = nn.Sigmoid()
        self.use_softmax = use_softmax
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x):        
        x = self.act(self.linear(x))
        if self.use_softmax:
            res = self.act2(self.linear2(x.squeeze(-1)))
        else:
            res = self.linear2(x.squeeze(-1))
        return res

# class ModEEPredictor_avgdim(nn.Module):
#     ''' linear_v1
#         b*320*768 b*320*768
#         b*320*1   b*320*1
#         b*3*1
#     '''
#     def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
#         super().__init__()

#         self.linear = nn.Linear(embed_dim, 1)
#         self.linear2 = nn.Linear(2, type_num)
#         self.act = nn.Sigmoid()
#         self.use_softmax = use_softmax
#         if use_softmax:
#             self.act2 = nn.Softmax()

#     def forward(self, x):        
#         x = self.act(self.linear(x))
#         if self.use_softmax:
#             res = self.act2(self.linear2(x.squeeze(-1)))
#         else:
#             res = self.linear2(x.squeeze(-1))
#         return res
    
class ModEEPredictor_v2(nn.Module):
    ''' linear_v2
        320*768     320*768
        320*1       320*1
         |-----1*1----|
        1*1     |    1*1
         |-----3*1----|
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False, linear4=False):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(token_num*2, 1)
        self.linear3 = nn.Linear(token_num, 1)
        if linear4:
            self.linear4 = nn.Linear(type_num, type_num)
        self.gelu = nn.GELU()
        self.act = nn.Sigmoid()

        self.use_softmax = use_softmax
        self.linear4 = linear4
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x, xi):        
        x = self.gelu(self.linear(x))
        xi = self.gelu(self.linear(xi))
        rgbtir = self.act(self.linear2(torch.cat((x,xi),dim=1).squeeze(-1)))
        rgb = self.act(self.linear3(x.squeeze(-1)))
        tir = self.act(self.linear3(xi.squeeze(-1)))
        if self.linear4:
            if self.use_softmax:
                res = self.act2(self.linear4(torch.cat((rgb,rgbtir,tir),dim=1)))
            else:
                res = self.linear4(torch.cat((rgb,rgbtir,tir),dim=1))
        else:
            if self.use_softmax:
                res = self.act2(torch.cat((rgb,rgbtir,tir),dim=1))
            else:
                res = torch.cat((rgb,rgbtir,tir),dim=1)
        return res
    
class ModEEPredictor_v3_distill(nn.Module):
    ''' linear_v3_distill
        320*768     320*768
        320*768     320*768
        320*1       320*1
                3*1
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
        super().__init__()

        self.linear_distill = nn.Linear(embed_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(token_num*2, type_num)
        self.act_distill = nn.GELU()
        self.act = nn.Sigmoid()
        self.use_softmax = use_softmax
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x, xi):       
        x_dis = self.act_distill(self.linear_distill(x))
        xi_dis = self.act_distill(self.linear_distill(xi))
        x = self.act(self.linear(x_dis))
        xi = self.act(self.linear(xi_dis))
        if self.use_softmax:
            res = self.act2(self.linear2(torch.cat((x,xi),dim=1).squeeze(-1)))
        else:
            res = self.linear2(torch.cat((x,xi),dim=1).squeeze(-1))
        return res, [x_dis, xi_dis]
    
class ModEEPredictor_v4_distill(nn.Module):
    ''' linear_v4_distill
        320*768     320*768
                cat
              1* 768*2
                 3
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
        super().__init__()

        self.linear_distill_r = nn.Linear(embed_dim, embed_dim)
        self.linear_distill_t = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(token_num, 1)
        self.linear3 = nn.Linear(embed_dim*2, type_num)
        self.act1 = nn.GELU()
        # self.act = nn.Sigmoid()
        self.use_softmax = use_softmax
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x, xi):       
        x_dis = self.act1(self.linear_distill_r(x))
        xi_dis = self.act1(self.linear_distill_t(xi))
        x = self.act1(self.linear2(torch.cat((x_dis,xi_dis),dim=-1).transpose(-1,-2)))
        
        if self.use_softmax:
            res = self.act2(self.linear3(x.squeeze(-1)))
        else:
            res = self.linear3(x.squeeze(-1))
        return res, [x_dis, xi_dis]

class ModEEPredictor_v5_mamba(nn.Module):
    ''' linear_v5_mamba
        320*768     320*768
        320*768     320*768
        320*1       320*1
                3*1
    '''
    def __init__(self, embed_dim=768, type_num=3, token_num=320, use_softmax=False):
        super().__init__()

        self.mamba = Mamba2(d_model=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(token_num*2, type_num)
        self.act_m = nn.GELU()
        self.act = nn.Sigmoid()
        self.use_softmax = use_softmax
        if use_softmax:
            self.act2 = nn.Softmax()

    def forward(self, x, xi):        
        x = self.act_m(self.mamba(x))
        xi = self.act_m(self.mamba(xi))
        x = self.act(self.linear(x))
        xi = self.act(self.linear(xi))
        if self.use_softmax:
            res = self.act2(self.linear2(torch.cat((x,xi),dim=1).squeeze(-1)))
        else:
            res = self.linear2(torch.cat((x,xi),dim=1).squeeze(-1))
        return res


class Reconstructor_mamba(nn.Module):
    ''' Reconstructor_mamba
    '''
    def __init__(self, embed_dim=768):
        super().__init__()

        self.mamba = Mamba(d_model=embed_dim)

    def forward(self, x):        
        x = self.mamba(x)
        return x