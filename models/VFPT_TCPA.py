# VPT mathod
from utils.args import *
from torch import nn
from models.utils import get_backbone, SimamModule
import torch
from train.utils import device
from models.TCPA import Pool_TCPA, IPrompt_TCPA
    

    
class Model_VFPT_TCPA(nn.Module):
    def __init__(self, args):
        super(Model_VFPT_TCPA, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.mode = "train"
        # self.prompts = nn.Parameter(torch.randn(12, args.p_len_vpt, 768))
        # self.prompts.data.uniform_(-1, 1)
        # self.prompt_dropout = torch.nn.Dropout(self.args.prompt_dropout_vpt)
        
        if self.args.RDVP:
            self.p_len = self.args.pool_size_cls * self.args.len_prompts_cls + self.args.pool_size_image * self.args.len_prompts_image
            # self.p_len = self.args.topk_cls * self.args.len_prompts_cls + self.args.topk_image * self.args.len_prompts_image
            self.Pool = Pool_TCPA(args=args)
        if self.args.ADVP:
            self.ADVP_w = torch.randn(1, self.p_len, 768)
            self.ADVP_b = torch.randn(1, self.p_len, 768)
            torch.nn.init.kaiming_normal_(self.ADVP_w, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.ADVP_b, nonlinearity='relu')
            self.ADVP_w = self.ADVP_w * 0.1 + 1
            self.ADVP_b = self.ADVP_b * 0.1
            self.ADVP_w = nn.Parameter(self.ADVP_w)
            self.ADVP_b = nn.Parameter(self.ADVP_b)
        if self.args.IPrompt:
            self.Prompter = IPrompt_TCPA(args=args)
        if self.args.simam:
            self.simam = SimamModule() 
        
    def train(self):
        self.mode = "train"
        self.backbone.eval()
        self.backbone.head.train()
        self.Pool.train()
    
    def eval(self):
        self.mode = "eval"
        self.backbone.eval()
        self.backbone.head.eval()
        self.Pool.eval()
            
    def forward(self, x, random=False):
        if self.args.simam:
            x = self.simam(x)
        if self.args.IPrompt:
            x = x + self.Prompter(x)          
        B = x.shape[0]
        # p_len = self.args.p_len_vpt
        backbone = self.backbone
        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)
        x = backbone.norm_pre(x)
        dist = 0
        for i, block in enumerate(backbone.blocks):
            prompt = self.Pool.get_prompts(layer=i, B=B)
            self.mask, dist_layer = self.Pool.get_mask(layer=i, x=x, random=random)  
            if dist_layer is None:
                dist = None
            else:
                dist += dist_layer      
            prompt = torch.fft.fft(torch.fft.fft(prompt, dim=-1),dim=-2).real
            x = torch.cat((x[:, :1, :], prompt, x[:, 1:, :]), dim=1)
            if self.args.random_prompt and self.mode == "train":
                x = x + 0.001 * torch.randn(x.shape, requires_grad=False).to(device)
            x = self.forward_block(block, x)
        x = backbone.norm(x)
        x = backbone.forward_head(x)
        return x, dist
    
    def forward_block(self, block, x):
        # print(x.shape)
        res = block.norm1(x)
        # res = block.attn(res)
        res = self.forward_attn(block.attn, res)
        if self.args.RDVP:
            # print(x.shape)
            x = torch.cat((x[:, :1, :], x[:, (1+self.p_len):, :]), dim=1)
            # print(x.shape)
        x = x + block.drop_path1(block.ls1(res))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x
    
    
    def forward_attn(self, Attn, x):
        p_len = self.p_len
        B, N, C = x.shape
        qkv = Attn.qkv(x).reshape(B, N, 3, Attn.num_heads, C // Attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        if self.args.ADVP:
            v = v.permute(0, 2, 1, 3)
            v = v.reshape(B, -1, 768)   
            v_prompt = v[:, 1:(1+p_len), :]
            v_prompt = v_prompt * self.ADVP_w + self.ADVP_b
            v = torch.cat([v[:, :1, :], v_prompt, v[:, (1+p_len):, :]], dim=1)
            v = v.reshape(B, -1, 12, 64)
            v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * Attn.scale
        attn = attn.softmax(dim=-1)
        attn = Attn.attn_drop(attn)
        
        if self.args.RDVP:
            attn = torch.cat([attn[:, :, :1, :], attn[:, :, (1+p_len):, :]], dim=2)
            attn = attn * self.mask
            x = (attn @ v).transpose(1, 2).reshape(B, N-p_len, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = Attn.proj(x)
        x = Attn.proj_drop(x)
        return x

    
    def learnable_parameters(self):
        if self.args.arch == 'ViT/B-16':
            params = list(self.backbone.head.parameters())
            if self.args.ADVP:
                params += [self.ADVP_w]
                params += [self.ADVP_b]
            if self.args.RDVP:
                params += list(self.Pool.parameters())
            if self.args.IPrompt:
                params += list(self.Prompter.parameters())
        else: raise NotImplementedError()
        return params
    
    
    
