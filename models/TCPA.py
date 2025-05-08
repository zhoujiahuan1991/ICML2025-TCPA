import torch
import torch.nn as nn
from train.utils import device

class IPrompt_TCPA(nn.Module):
    def __init__(self, args, dim=9, patch=11):
        super().__init__()
        self.Prompter = nn.Sequential(
            nn.Conv2d(3, dim, patch, stride=1, padding=int((patch-1)/2)),
            nn.PReLU(),
            nn.Conv2d(dim, 3, patch, stride=1, padding=int((patch-1)/2))
        )
    def forward(self, x):
        return self.Prompter(x)


class Pool_TCPA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.p_len_cls = self.args.pool_size_cls * self.args.len_prompts_cls
        self.p_len_image = self.args.pool_size_image * self.args.len_prompts_image
        self.p_len = self.p_len_cls + self.p_len_image
        self.Pool_cls = Pool(args=args, topk=args.topk_cls, pool_size=args.pool_size_cls, len_prompts=args.len_prompts_cls)
        self.Pool_image = Pool(args=args, topk=args.topk_image, pool_size=args.pool_size_image, len_prompts=args.len_prompts_image)
        self.prompt_dropout = torch.nn.Dropout(self.args.prompt_dropout_dvp)
        self.mode = "train"
        
    def eval(self):
        self.Pool_cls.eval()
        self.Pool_image.eval()
        self.prompt_dropout.eval()
    
    def train(self):
        self.Pool_cls.train()
        self.Pool_image.train()
        self.prompt_dropout.train()
    
    def get_prompts(self, layer, B):
        prompts_cls = self.Pool_cls.prompts[layer]
        prompts_cls = prompts_cls.reshape(-1, self.Pool_cls.dim)
        prompts_image = self.Pool_image.prompts[layer]
        prompts_image = prompts_image.reshape(-1, self.Pool_image.dim)
        # 将prompt_cls和prompt_image合并
        prompts = torch.cat([prompts_cls, prompts_image], dim=0)        
        prompts = prompts.expand(B, -1, -1)
        # if self.args.random_prompt :
        #     tmp = 0.01 * torch.randn(2, 2, 2)

        prompts = self.prompt_dropout(prompts) # [p_len, 768]
        return prompts

    def get_mask(self, layer, x, random=False):
        if self.args.TDVP:
            return self.get_mask_TDVP(layer, x, random=random)
        elif self.args.RDVP:
            return self.get_mask_RDVP(layer, x)
        
    def get_mask_RDVP(self, layer, x):
        mask_cls = torch.ones((197+self.p_len), device=device)
        mask_cls[(1+self.p_len_cls): (1+self.p_len)] = 0
        # 扩展1个维度
        mask_cls = mask_cls.unsqueeze(0)
        mask_image = torch.ones((197+self.p_len), device=device)
        mask_image[1:(1+self.p_len_cls)] = 0
        # 将mask_image expand 196份
        mask_image = mask_image.expand(196, -1)
        # 将mask_cls和mask_image合并
        mask = torch.cat([mask_cls, mask_image], dim=0)
        # 将mask 由 [197, 357] 扩展为 [12, 197, 357]
        mask = mask.expand(12, -1, -1)
        mask = mask.expand(x.shape[0], -1, -1, -1)
        # print(mask.shape) # torch.Size([128, 12, 197, 357])
        # input()
        return mask, None
    
    def get_mask_TDVP(self, layer, x, random=False):
        mask_cls = torch.ones((197+self.p_len), device=device)
        mask_cls[(1+self.p_len_cls): (1+self.p_len)] = 0
        # 扩展1个维度
        mask_cls = mask_cls.unsqueeze(0)        
        mask_image = torch.ones((197+self.p_len), device=device)
        mask_image[1:(1+self.p_len_cls)] = 0
        # 将mask_image expand 196份
        mask_image = mask_image.repeat(196, 1)
        # 将mask_cls和mask_image合并
        mask = torch.cat([mask_cls, mask_image], dim=0)
        # 将mask 由 [197, 357] 扩展为 [12, 197, 357]
        # print(mask.shape)
        # print(mask)
        # print(mask[0])
        # print(mask[1])
        # print(mask[2])
        # print(mask[3])
        # print(mask[4])
        # print(mask[5])
        # input()
        mask = mask.repeat(12, 1, 1)
        mask = mask.repeat(x.shape[0], 1, 1, 1)
        # print(mask.shape) # torch.Size([128, 12, 197, 357])
        # input()
        # x [128, 197, 768]
        # print(x.shape)
        cls_token = x[:,0,:]
        dist_cls, index_cls = self.Pool_cls(cls_token, layer, random=random, return_index=True)
        # print(dist_cls)
        # print(index_cls.shape)
        # print(index_cls)
        # input()
        index_cls = index_cls.to(device)
        tmp = torch.zeros((x.shape[0], self.args.pool_size_cls), device=device)
        # tmp[index_cls] = 1
        tmp.scatter_(dim=1, index=index_cls, value=1)
        # print(tmp)
        # input()

        # tmp = torch.zeros((2, 4), device=device)
        # index_cls = torch.tensor([[1],[3]], dtype=torch.long)
        # index_cls = index_cls.to(device)
        # tmp.scatter_(dim=1, index=index_cls, value=1)
        # print(tmp)
        # input()
        # tmp = torch.zeros((2, 4), device=device)
        # index_cls = torch.tensor([[1,2],[0,3]], dtype=torch.long)
        # index_cls = index_cls.to(device)
        # tmp.scatter_(dim=1, index=index_cls, value=1)
        # print(tmp)
        # input()
        # tmp = torch.zeros((2, 4), device=device)
        # index_cls = torch.tensor([[0,1,2,3],[0,1,2,3]], dtype=torch.long)
        # index_cls = index_cls.to(device)
        # tmp.scatter_(dim=1, index=index_cls, value=1)
        # print(tmp)
        # input()
        # print(tmp.shape)
        # print(tmp)
        # input()
        tmp = torch.repeat_interleave(tmp, repeats=self.args.len_prompts_cls, dim=1) # [B, p_len_cls]
        # print(tmp.shape)
        # print(tmp)
        # input()
        # print(tmp.shape)
        tmp = tmp.repeat(12, 1, 1) # [12, B, p_len_cls]
        # print(tmp.shape)
        tmp = tmp.permute(1, 0, 2) # [B, 12, p_len_cls]\
        # print(tmp.shape)
        mask[:,:,0,1:(1+self.p_len_cls)] = tmp
        
        image_token = x[:,1:,:]
        image_token = image_token.reshape(-1, 768)
        dist_image, index_image = self.Pool_image(image_token, layer, random=random, return_index=True)
        # print(dist_image)
        # print(index_image.shape)
        # print(index_image)
        index_image = index_image.to(device)
        tmp = torch.zeros((x.shape[0] * 196, self.args.pool_size_image), device=device)
        # tmp[index_image] = 1
        tmp.scatter_(dim=1, index=index_image, value=1)
        # print(tmp.shape)
        # print(tmp)
        # input()
        tmp = tmp.reshape(x.shape[0], 196, self.args.pool_size_image)
        # print(tmp.shape)
        # print(tmp)
        # input()
        tmp = torch.repeat_interleave(tmp, repeats=self.args.len_prompts_image, dim=2) # [B, 196, p_len_cls]
        # print(tmp.shape)
        # print(tmp)
        # input()
        tmp = tmp.repeat(12, 1, 1, 1) # [12, B, 196, p_len_cls]
        tmp = tmp.permute(1, 0, 2, 3) # [B, 12, 916, p_len_cls]
        mask[:,:,1:,(1+self.p_len_cls): (1+self.p_len)] = tmp
        # print(mask.shape)
        # # print(mask[0][0])
        # print(mask[0][0][0])
        # print(mask[0][0][1])
        # print(mask[0][0][2])
        # print(mask[0][0][3])
        # print(mask[0][0][4])
        # input()
        # print(mask[1][0][0])
        # print(mask[1][0][1])
        # print(mask[1][0][2])
        # print(mask[1][0][3])
        # print(mask[1][0][4])
        # print(mask[1][0][5])
        # input()
        # print(mask[2][0][0])
        # print(mask[2][0][1])
        # print(mask[2][0][2])
        # print(mask[2][0][3])
        # print(mask[2][0][4])
        # print(mask[2][0][5])
        # input()
        # print(mask[2][2][0])
        # print(mask[2][2][1])
        # print(mask[2][2][2])
        # print(mask[2][2][3])
        # print(mask[2][2][4])
        # print(mask[2][2][5])
        # input()
        return mask, dist_cls + dist_image
        
            
    
        



class Pool(nn.Module):
    def __init__(self, args, topk=9, num_layers=12, pool_size=20, len_prompts=1, dim=768):
        super().__init__()
        self.args = args
        self.topk = topk
        self.num_layers = num_layers
        self.pool_size = pool_size
        self.len_prompts = len_prompts
        self.dim = dim
        self.keys = nn.Parameter(torch.randn(num_layers, pool_size, dim))
        self.keys.to('cuda')
        self.keys.data.uniform_(-1, 1)
        self.prompts = nn.Parameter(torch.randn(num_layers, pool_size, len_prompts, dim))
        self.prompts.data.uniform_(-1, 1)
        self.prompts.to('cuda')
        
    def forward(self, x, layer, random=False, return_index=False):
        # x [64, 768]
        # print(x.shape)
        B, dim = x.shape
        # 取出当前层的key和prompt
        keys = self.keys[layer] # [20, 768]
        prompts = self.prompts[layer] # [20, 1, 768]
        # 对x和key进行归一化
        x = torch.nn.functional.normalize(x, dim=-1)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        # 计算当前层的key和x的余弦相似度
        sim = torch.matmul(x, keys.t()) # [64, 20]
        
        if random is True:
            # 对于batch中的B个样本,为每个样本随机选择topk个prompt
            # 随机选择topk个prompt的index
            # topk_index = torch.randint(0, self.pool_size, (B, self.topk), device=device)
            rand_values = torch.rand(B, self.pool_size, device=device)
            # print(rand_values.shape)
            # print(rand_values)
            # # input()
            # print(rand_values.argsort(dim=1).shape)
            # print(rand_values.argsort(dim=1))
            # # input()
            # # 对每行随机排序后取前topk个索引
            topk_index = rand_values.argsort(dim=1)[:, :self.topk]

            # topk_index = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
            # topk_index = topk_index.repeat(B, 1)
        else:    
            # 取出相似度最大的topk 个 prompt的index
            topk = torch.topk(sim, self.topk, dim=1)  # [64, 9]
            topk_index = topk.indices
        # print(topk_index[0])
        # 取出相似度最大的topk个prompt
        # prompts = prompts[topk_index] # [64, 9, 1, 768]
        # prompts = prompts.reshape(B, -1, dim) # [64, 9, 768]
        # 取出Topk个prompt的相似度
        # sim_topk = sim[topk_index] # [64, 9]
        sim_topk = torch.gather(sim, dim=1, index=topk_index)


        # sim = torch.tensor([[11,12,13,14],[21,22,23,24]],device=device)
        # topk_index = torch.tensor([[1],[3]],dtype=torch.long, device=device)
        # # sim_topk = sim[topk_index]
        # sim_topk = torch.gather(sim, dim=1, index=topk_index)
        # print(sim_topk.shape)
        # print(sim_topk)
        # input()
        # sim = torch.tensor([[11,12,13,14],[21,22,23,24]],device=device)
        # topk_index = torch.tensor([[1,2],[3,0]],dtype=torch.long, device=device)
        # # sim_topk = sim[topk_index]
        # sim_topk = torch.gather(sim, dim=1, index=topk_index)
        # print(sim_topk.shape)
        # print(sim_topk)
        # input()
        # sim = torch.tensor([[11,12,13,14],[21,22,23,24]],device=device)
        # topk_index = torch.tensor([[1,2,3,0],[3,0,2,1]],dtype=torch.long, device=device)
        # # sim_topk = sim[topk_index]
        # sim_topk = torch.gather(sim, dim=1, index=topk_index)
        # print(sim_topk.shape)
        # print(sim_topk)
        # input()

        # sim_topk = sim[topk_index] # [64, 9]
        dist = 1-sim_topk # [64, 9]
        # print(dist.shape)
        # print(dist[0])
        # 计算dist所有位置的平均值
        dist = dist.mean()
        # print(dist)
        # input()
        if return_index: 
            return dist, topk_index 
        else: return dist



# class Pool_2(nn.Module):
#     def __init__(self, args, topk=9, num_layers=12, pool_size=20, len_prompts=1, dim=768):
#         super().__init__()
#         self.args = args
#         self.topk = topk
#         self.num_layers = num_layers
#         self.pool_size = pool_size
#         self.len_prompts = len_prompts
#         self.dim = dim
#         self.keys = nn.Parameter(torch.randn(num_layers, pool_size, dim))
#         self.keys.to('cuda')
#         self.keys.data.uniform_(-1, 1)
#         self.prompts_net = nn.Sequential(
#             nn.Linear(dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, dim),
#         )
#         self.prompts_net.to('cuda')
#     def forward(self, x, layer, random=False, return_key=False):
#         # x [64, 768]
#         B, dim = x.shape
#         # 取出当前层的key和prompt
#         keys = self.keys[layer] # [20, 768]
#         prompts = self.prompts_net(keys) # [20, 1, 768]
#         # 对x和key进行归一化
#         x = torch.nn.functional.normalize(x, dim=-1)
#         keys = torch.nn.functional.normalize(keys, dim=-1)
#         # 计算当前层的key和x的余弦相似度
#         sim = torch.matmul(x, keys.t()) # [64, 20]
#         if random is True:
#             # 对于batch中的B个样本,为每个样本随机选择topk个prompt
#             # 随机选择topk个prompt的index
#             topk_index = torch.randint(0, self.pool_size, (B, self.topk))
#         else:    
#             # 取出相似度最大的topk 个 prompt的index
#             topk = torch.topk(sim, self.topk, dim=1)  # [64, 9]
#             topk_index = topk.indices
#         # print(topk_index[0])
#         # 取出相似度最大的topk个prompt
#         prompts = prompts[topk_index] # [64, 9, 1, 768]
#         prompts = prompts.reshape(B, -1, dim) # [64, 9, 768]
#         # 取出Topk个prompt的相似度
#         sim_topk = sim[topk_index] # [64, 9]
#         dist = 1-sim_topk # [64, 9]
#         # print(dist.shape)
#         # print(dist[0])
#         # 计算dist所有位置的平均值
#         dist = dist.mean()
#         # print(dist)
#         # input()
#         if return_key is True:
#             return prompts, dist, keys[topk_index]
#         return prompts, dist