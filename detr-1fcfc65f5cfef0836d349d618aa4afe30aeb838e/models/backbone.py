# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))    #应该就是在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出。
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)  #手动加载模型

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()   # 求平方根的倒数，然后直接元素相乘，然后，这里W除以的标准差
        bias = b - rm * scale            # 偏差乘以均值，但这个均值经历的缩放
        return x * scale + bias          # 然后在正式normal


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)   # 冻结layer2,layer3,layer4,如果train_backone标记为不训练，那么所有梯度都要冻结掉
        if return_interm_layers:   #这个展示不清楚，是用来干嘛的
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)  #IntermediateLayerGetter ,返回层，返回层的内容，取决于return_interm_layers，resnet50分为四个stage
        self.num_channels = num_channels                                            #通道数量
    '''
    class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        '''
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)   # 对多个tensors,进行封装，tensor[0]：[batch_size,C,W,H]   xs:[batch_size,2048, H/32, W/32]
        out: Dict[str, NestedTensor] = {}     # 产生一个dict，一个name，对应于一个NestedTensor
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 功能：利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
            # 输入：
            # input(Tensor)：需要进行采样处理的数组。
            # scale_factor(float或序列)：空间大小的乘数
            # mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
            # align_corners(bool)：在几何上，我们将输入和输出的像素视为正方形而不是点。如果设置为True，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值。如果设置为False，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于边界外值，使此操作在保持不变时独立于输入大小scale_factor。
            # recompute_scale_facto(bool)：重新计算用于插值计算的 scale_factor。当scale_factor作为参数传递时，它用于计算output_size。如果recompute_scale_factor的False或没有指定，传入的scale_factor将在插值计算中使用。否则，将根据用于插值计算的输出和输入大小计算新的scale_factor（即，如果计算的output_size显式传入，则计算将相同 ）。注意当scale_factor 是浮点数，由于舍入和精度问题，重新计算的 scale_factor 可能与传入的不同。
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]  #个人估计mask的维度应该和x是一样的
            out[name] = NestedTensor(x, mask)  #tensor_list,中，掩码是同一个吗？
        return out     #out ，是个字典，name就是resnet50的一个输出，key里面包含一个NestedTensor,这个NestedTensor中就一个tensor和对应的掩码流


class Backbone(BackboneBase):  # Base的子类，对base进行补充
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(   # 应该是其他代码里面已经加载好了，我们这边吧resnet50给拿出来，通过name
        """
        :param 传入参数 block: Bottleneck
        :param 传入参数 layers:[3, 4, 6, 3]
        :param num_classes: 分类数
        :param zero_init_residual: 零初始化
        :param groups: 分组数（暂时用不上，ResNeXt用）
        :param width_per_group: 每个分组的通道数（暂时用不上，ResNeXt用）
        :param replace_stride_with_dilation: 是否用空洞卷积替代stride（用不上），空洞卷及可以压缩步长，当操作内容只会被压缩到图片的核心区域，自己想想，但空洞卷，resnet 除了第一层，之外，一共有三层，这三个元素，代表了那些元素需要使用空洞卷及
        :param norm_layer:BatchNorm
        """
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
