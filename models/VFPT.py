# VPT mathod
from utils.args import *
from torch import nn
from models.utils import get_backbone
import torch
from train.utils import device
    
    
class Model_VFPT(nn.Module):
    def __init__(self, args):
        super(Model_VFPT, self).__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.mode = "train"
        self.prompts = nn.Parameter(torch.randn(12, args.p_len_vpt, 768))
        self.prompts.data.uniform_(-1, 1)
        self.prompt_dropout = torch.nn.Dropout(self.args.prompt_dropout_vpt)
        
    def train(self):
        self.backbone.eval()
        self.backbone.head.train()
        self.prompt_dropout.train()
    
    def eval(self):
        self.backbone.eval()
        self.backbone.head.eval()
        self.prompt_dropout.eval()
            
    def forward(self, x):        
        B = x.shape[0]
        p_len = self.args.p_len_vpt
        backbone = self.backbone
        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)
        x = backbone.norm_pre(x)
        for i, block in enumerate(backbone.blocks):
            prompt = self.prompt_dropout(self.prompts[i].expand(B, -1, -1)) # [p_len, 768]
            prompt = torch.fft.fft(torch.fft.fft(prompt, dim=-1),dim=-2).real
            if i == 0:
                x = torch.cat((x[:, :1, :], prompt, x[:, 1:, :]), dim=1)
            else:
                x = torch.cat((x[:, :1, :], prompt, x[:, (1+p_len):, :]), dim=1)
            x = block.forward(x)
        x = backbone.norm(x)
        x = backbone.forward_head(x)
        return x
    
#     # Visual Prompts
# x = torch.cat((	x[:, :1, :],
#                 prompt_dropout(prompt_proj(prompt_embeddings).expand(B, -1, -1)), 
#                 x[:, 1:, :]), dim=1)

# # Visual Fourier Prompts (Fourier percentage equals 1.0)
# x = torch.cat((	x[:, :1, :],
#                 torch.fft.fft(torch.fft.fft(
#                 prompt_dropout(prompt_proj(prompt_embeddings).expand(B, -1, -1)), 
#                                             dim=-1),dim=-2).real,
#                 x[:, 1:, :]), dim=1)
    
    def learnable_parameters(self):
        if self.args.arch == 'ViT/B-16':
            params = list(self.backbone.head.parameters())
        else: raise NotImplementedError()
        params += [self.prompts]
        return params