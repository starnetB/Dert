import random  

import PIL
from numpy import FLOATING_POINT_SUPPORT
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
# 此类用于一个batch中图片的大小调整
from util.misc import interpolate

'''
@ image 图片
@ target 包含一些json读过来的参数
    @ 包含boxes   1对多的关系  [n,2,2]
    @ 包含masks   1对多的关系  [m,i,h] 
'''
def crop(image,target,region):
    #根据提供的区域裁剪图片
    #top，left，height，width
    # The image can be a PIL image or a Tensor 
    cropped_image=F.crop(image,*region)

    target=target.copy()
    i,j,h,w =region
    # should we do something wrt the original size
    # 所以target是从json读出来的图片信息，这里重新调整图片信息
    target["size"]=torch.tensor([h,w])
    # iscrowd 是什么意思，后面会用到
    # 首先，得分优先原则，即得分大的det先去匹配gt；
    # 每次匹配的时候，每次匹配时的gt candidates是未被匹配的gt并上已匹配但是属性crowd为true的gt；
    fields=["labels","area","iscrowd"]

    if "boxes" in target:
        boxes=target["boxes"]
        max_size=torch.as_tensor([w,h],dtype=torch.float32)
        #修改boxes的位置信息
        # boxes信息 ：[x_l,y_l,x_r,y_r]
        cropped_boxes=boxes-torch.as_tensor([j,i,j,i])
        # 防止box超出最大边界
        # 最大值不可以超出裁剪后的图片边界
        # 一副图片可能有多个box
        cropped_boxes=torch.min(cropped_boxes.reshape(-1,2,2),max_size)
        # 不得出现小于0的情况发生
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        # 重新裁剪后的参数
        target["boxes"]=cropped_boxes.reshape[-1,4]
        target["area"]=area
        fields.append("boxes") 
    
    if "mask" in target:
        # FIXME should we update the area here if there are no boxes?
        # 把裁剪部分的掩码流拿出来
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("mask")
    
    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes=target['boxes'].reshape(-1,2,2)
            # 仅仅保留那些right大于left坐标的正常boxes
            '''
            # a = torch.rand(4, 2).bool()
            # tensor([[True, True],
                      [True, False],
                      [True, True],
                      [True, True]], dtype=torch.bool)
            torch.all(a, dim=1)
            tensor([ True, False,  True,  True], dtype=torch.bool)
            '''
            keep=torch.all(cropped_boxes[:,1,:]>cropped_boxes[:,0,:])
        else:
            #保留那些存在True的对象
            #flatten(dim)表示，从第dim个维度开始展开，将后面的维度转化为一维
            #any(1)如果存在1就返回Ture
            #所以结果也是[true,true,false....]
            keep=target['masks'].flatten(1).any(1)
        
        for field in fields:
            target[field]=target[field][keep]
    return cropped_image,target

def hflip(image,target):
    # 将图片水平反转
    flipped_image=F.hflip(image)

    w,h=image.size

    target=target.copy()
    if "boxes" in target:
        boxes=target["boxes"]
        boxes = boxes[:, 
        # x_l,y_l,x_r,y_r ->x_r,y_l,x_l,y_r ，水平反转了呗
        [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"]=boxes
    
    if "masks" in target:
        target['masks']=target['masks'].flip(-1)
    
    return flipped_image,target

# 根据定义改变图片大小
def resize(image,target,size,max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size,size,max_size=None):
        w,h=image_size
        if max_size is not None:
            min_original_size=float(min(w,h))
            max_original_size=float(max(w,h))
            # 图的最小尺寸如果在按比例放大后超过了最大范围，那么就重新调整size
            if max_original_size/min_original_size*size>max_size:
                size=int(round(max_size*min_original_size/max_original_size))
        if(w<=h and w==size) or (h<=w and h==size):
            return (h,w)
        
        if w<h:
            ow=size
            oh=int(size* h / w)
        else:
            oh=size
            ow=int(size* w / h)

        return (oh,ow)
    
    def get_size(image_size,size,max_size=None):
        #如果size是个list or tuple那么直接反转(w,h)->(h,w)
        if isinstance(size,(list,tuple)):
            return size[::-1]
        else:
        #如果size是min_size,也就是最小边的值，那么返回运算的结果(h,w)
            return get_size_with_aspect_ratio(image_size,size,max_size)
    
    size=get_size(image.size,size,max_size)
    rescaled_image=F.resize(image.size)

    #如果没有target,那么就这么返回吧
    if target in None:
        return rescaled_image,None
    #放回比例tuple （h_r/h_o,w_r/h_o)
    ratios=tuple(float(s)/float(s_orig) for s,s_orig in zip(rescaled_image.size,image.size))
    ratio_width,ratio_height=ratios  
    target=target.copy()
    if "boxes" in target:
        boxes=target["boxes"]
        scale_boxes=boxes*torch.as_tensor([ratio_width,ratio_height,ratio_width,ratio_height])
        target["boxes"]=scale_boxes
    
    if "area" in target:
        area=target["area"]
        scaled_area=area*(ratio_width*ratio_height)
        target["area"]=scaled_area
    
    h,w=size  
    target["size"]=torch.size([h,w])

    #后面增加一个维度，因为，interploate 中调用的api支持4维度或5维度的tensor
    #采用最临近插值
    if "masks" in target:
        target['mask']=interpolate(
            target["masks"][:,None].float(),size,mode="nearest")[:,0]>0.5

    # 返回target与rescaled_image
    return rescaled_image,target

# 填充相关图片
def pad(image,target,padding):
    # assumes that we only pad on the bottom right corners
    # 在右边和下方填充我们的内容
    padded_image=F.pad(image,(0,0,padding[0],padding[1]))
    if target in None:
        return padded_image,None
    target=target.copy()
    #should we do something wrt the original size?
    #target['size']=torch.tensor(padded_image[::-1])
    target['size']=torch.tensor(padded_image.shape[::-1])
    if "masks" in target:
        target['masks']=torch.nn.functional.pad(target['masks'],(0,padding[0],0,padding[1]))
    return padded_image,target