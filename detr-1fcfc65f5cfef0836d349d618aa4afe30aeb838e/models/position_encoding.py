# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, tensor_list: NestedTensor):  #假设通道位数我们设为128
        x = tensor_list.tensors   #假设[batch_size,128,h/32=24,1/32=32]
        mask = tensor_list.mask   #[batch_size,24,32]
        assert mask is not None
        # padding=1,picture=0  取反之后，padding=0,picture=1
        # [0,0,0,1]
        # [0,0,0,1]
        # [0,0,0,1]
        # [1,1,1,1]
        # ~mask
        # [1,1,1,0]
        # [1,1,1,0]
        # [1,1,1,0]
        # [0,0,0,0]
        not_mask = ~mask
        # mask.shape=[batch_size,w/32,h/32]  这是因为主体经过backbone压缩过了          
        # 沿着axis=2 按列方向进行累加
        # x_embed
        # [1,2,3,3]
        # [1,2,3,3]
        # [1,2,3,3]
        # [0,0,0,0]
        # 沿着axis=1 按行方向进行累加
        # y_embed
        # [1,1,1,0]
        # [2,2,2,0]
        # [3,3,3,0]
        # [3,3,3,0]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   

        if self.normalize:
            eps = 1e-6
            # 取最后一行为基准进行缩放，归一化
            # [1/3，1/3, 1/3，0/0=1]*scale
            # [2/3, 2/3, 2/3, 1   ]*scale
            # [3/3, 3/3, 3/3  1   ]*scale
            # [3/3, 3/3, 3/3  1   ]*scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 取最后一列为基准进行缩放
            # [1/3, 2/3, 3/3, 3/3]*scale
            # [1/3, 2/3, 3/3, 3/3]*scale
            # [1/3, 2/3, 3/3  3/3]*scale
            # [0/0, 0/0, 0/0  0/0]*scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        # [0,1,2,....63]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # [dim_t//2]=[0,0,1,1,2,2,3,3....,31,31...,63] 
        # (2 * (dim_t // 2)  =[0,0,2,2,4,4,6,6,....62,62.....,126]
        # (2 * (dim_t // 2) / self.num_pos_feats)=[0,0,2,2,4,4,6,6,....62,62,...126]/128
        # self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) =10000^{[0,0,2,2,4,4,6,6,....62,62...126]/128}
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # a=np.arange(5)
        # a=[1,2,3,4]
        # a[:,None]
        # [[1],
        #  [2],
        #  [3],
        #  [4]]
        # pos_x和pos_y的如果看不懂可以看下论文解读里的两附图
        # 我们这里可见w/32=24 h/32=32
        # x_embed[:, :, :, None]=[bach_size,24,32,1]
        # y_embed[:, :, :, None]=[bach_size,24,32,1]
        # x_embed[:, :, :, None] / dim_t =[batch_size,24,32,128]
        pos_x = x_embed[:, :, :, None] / dim_t  # None作用就是增加一个新的维度，并且这个唯独应该实在最前面的
        pos_y = y_embed[:, :, :, None] / dim_t
        # 见位置编码公式  
        # [batch_size ,24,32 128]
        # 一副图，去除batch_size 维度，就是一个长方体，图像w=24,h=32
        # 还有一个深度维度128,那么这个唯独
        # pox_x=[1,2,3]  #这里我们为了简单起见设定w=3,h=3,dim=4
        #       [1,2,3]
        #       [1,2,3]
        # pox_x=1  pox_x[0][0]=[sin(1/(10000^{0/10}),cos(1/(10000^{0/10}),sin(1/(10000^{2/10}),cos(1/(10000^{2/10}))
        # pox_x其他坐标和pox_y都是一样的，位置与维度的耦合形成一个长方体
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 这边在 torch.cat((pos_y, pos_x), dim=3) 这边现在最后一个维度进行拼接，[batch_size,24,32,128]
        # 那么对应的pos[0][1]就是 pox_x=2  pox_y=1,那么，他们的深度唯独就i变成了 [{x部分}{sin(2/(10000^{0/10}),cos(2/(10000^{0/10}),sin(2/(10000^{2/10}),cos(2/(10000^{2/10}），(y部分)sin(1/(10000^{0/10}),cos(1/(10000^{0/10}),sin(1/(10000^{2/10}),cos(1/(10000^{2/10}]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 最后进行通道转换[batch_size,128*2,24,32]  与x在bachbone的输出维度是一直的，2i 为通道方向的变量，pos为图片像素索引变量
        return pos  


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)    
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors     # 读取图片tensor 
        h, w = x.shape[-2:]         # 读取图片
        i = torch.arange(w, device=x.device)   #这里有个问题，如果i大于50了会怎么样，会循环吗？   
        j = torch.arange(h, device=x.device)   #同上
        x_emb = self.col_embed(i)              #shape [w]  ,[w,num_pos_feature]
        y_emb = self.row_embed(j)              #shape [h]  ,[h,num_pos_feature]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),    #[w,num_pos_feature] =>[1,w,num_pos_feature]=> [h,w,num_pos_feature] =>w所在维度重复
            y_emb.unsqueeze(1).repeat(1, w, 1),    #[h,num_pos_feature] =>[h,1,num_pos_feature]=> [h,w,num_pos_feature] =>h所在维度重复
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  #在num_pos_feature*2，然后=>[num_pos_feature*2,h,w]=>[batch_size,num_pos_feature*2,h,w]  
        # 使用了nn.embeding，所以这个位置编码是训练出来的。
        return pos


def build_position_encoding(args):   # args={hidden_dim//2,position_embedding}
    N_steps = args.hidden_dim // 2   # 就是backbone输出的通道数的一半，为什么取得一半，因为，x，y各有自己的位置编码，经过cat之后就是两倍，128=>256 取一半正好
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
