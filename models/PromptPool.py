import torch
import torch.nn as nn









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
    def forward(self, x, layer, random=False, return_key=False):
        # x [64, 768]
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
            topk_index = torch.randint(0, self.pool_size, (B, self.topk))
        else:    
            # 取出相似度最大的topk 个 prompt的index
            topk = torch.topk(sim, self.topk, dim=1)  # [64, 9]
            topk_index = topk.indices
        # print(topk_index[0])
        # 取出相似度最大的topk个prompt
        prompts = prompts[topk_index] # [64, 9, 1, 768]
        prompts = prompts.reshape(B, -1, dim) # [64, 9, 768]
        # 取出Topk个prompt的相似度
        sim_topk = sim[topk_index] # [64, 9]
        dist = 1-sim_topk # [64, 9]
        # print(dist.shape)
        # print(dist[0])
        # 计算dist所有位置的平均值
        dist = dist.mean()
        # print(dist)
        # input()
        if return_key is True:
            return prompts, dist, keys[topk_index]
        return prompts, dist



class Pool_2(nn.Module):
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
        self.prompts_net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim),
        )
        self.prompts_net.to('cuda')
    def forward(self, x, layer, random=False, return_key=False):
        # x [64, 768]
        B, dim = x.shape
        # 取出当前层的key和prompt
        keys = self.keys[layer] # [20, 768]
        prompts = self.prompts_net(keys) # [20, 1, 768]
        # 对x和key进行归一化
        x = torch.nn.functional.normalize(x, dim=-1)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        # 计算当前层的key和x的余弦相似度
        sim = torch.matmul(x, keys.t()) # [64, 20]
        if random is True:
            # 对于batch中的B个样本,为每个样本随机选择topk个prompt
            # 随机选择topk个prompt的index
            topk_index = torch.randint(0, self.pool_size, (B, self.topk))
        else:    
            # 取出相似度最大的topk 个 prompt的index
            topk = torch.topk(sim, self.topk, dim=1)  # [64, 9]
            topk_index = topk.indices
        # print(topk_index[0])
        # 取出相似度最大的topk个prompt
        prompts = prompts[topk_index] # [64, 9, 1, 768]
        prompts = prompts.reshape(B, -1, dim) # [64, 9, 768]
        # 取出Topk个prompt的相似度
        sim_topk = sim[topk_index] # [64, 9]
        dist = 1-sim_topk # [64, 9]
        # print(dist.shape)
        # print(dist[0])
        # 计算dist所有位置的平均值
        dist = dist.mean()
        # print(dist)
        # input()
        if return_key is True:
            return prompts, dist, keys[topk_index]
        return prompts, dist
