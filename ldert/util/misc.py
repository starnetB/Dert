from ctypes import cast
from doctest import OutputChecker
from optparse import Option
import os
from re import L 
import subprocess
import time
from collections import defaultdict,deque
import datetime
import pickle
from tkinter import Scale
from typing import Optional,List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvisiong 0.5
import torchvision
if float(torchvision.__version__[:4])*10 < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

class NestedTensor(object):
    '''
    Optional[Tensor],mask默认参数类型是Tensor
    '''
    def __init__(self,tensors,mask: Optional[Tensor]):
        self.tensors=tensors
        self.mask=mask
    '''
    device to NestedTensor
    '''
    def to(self,device):
        cast_tensor=self.tensors.to(device)
        mask=self.mask
        if mask is not None:
            assert mask is not None
            cast_mask=mask.to(device)
        else:
            cast_mask=None
        return NestedTensor(cast_tensor,cast_mask)
    
    def decompose(self):
        return self.tensors,self.mask
    
    def __repr__(self):
        return str(self.tensors)

def is_dist_avail_and_initialized():
    # 如果没有GPU
    if not dist.is_available():
        return False
    # 如果没有初始化
    if not dist.is_initialized():
        return False
    return True 

def get_rank():
    # 如果设备没有获得GPU群，或者没有初始化就返回0
    # 否则返回当前的rank_num
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

'''
#paddle.nn.functional. interpolate ( 
# x, size=None, 
# scale_factor=None, 
# mode='nearest', 
# align_corners=False, 
# align_mode=0, 
# data_format='NCHW', name=None )
该OP用于调整一个batch中图片的大小。

输入为4-D Tensor时形状为(num_batches, channels, in_h, in_w)
或者(num_batches, in_h, in_w, channels)，
输入为5-D Tensor时形状为(num_batches, channels, in_d, in_h, in_w)
或者(num_batches, in_d, in_h, in_w, channels)，
并且调整大小只适用于深度，高度和宽度对应的维度。

# 插值
支持的插值方法:
    
    NEAREST：最近邻插值--赋值给最邻近的像素点值
    BILINEAR：双线性插值--在两个方向上进行线性插值，去中点什么的
    TRILINEAR：三线性插值
    BICUBIC：双三次插值
    LINEAR: 线性插值
    AREA: 面积插值
'''
def interpolate(input,size=None,scale_factor=None,mode="nearest",align_corners=None):
     # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:4])*10<0.7:
        if input.numel()>0:
            return torch.nn.functional.interpolate(
                input,size,scale_factor,mode,align_corners
            )
        output_shape=_output_size(2,input,size,scale_factor)
        output_shape=list(input.shape[:-2])+list(output_shape)
        return _new_empty_tensor(input,output_shape)
    else:
        return torchvision.ops.misc.interpolate(input,size,scale_factor,mode,align_corners)