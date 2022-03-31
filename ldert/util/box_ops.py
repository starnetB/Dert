import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    # torch.unbind(input ,dim=0)
    # remove a tensor dimension
    x_c,y_c,w,h=x.unbind(-1)
    b=[(x_c-0.5*w),(y_c-0.5*h),
       (x_c+0.5*w),(y_c+0.5*h)]
    
    return torch.stack(b,dim=1)

def box_xyxy_to_cxcywh(x):
    x0,y0,x1,y1=x.unbind(-1)
    b=[(x0+x1)/2,(y0+y1)/2,
       (x1-x0),(y1-y0)]
    return torch.stack(b,dim=1)

# modified from torchvision to also return the union
# boxes1 of dim [N,4]
# boxes2 of dim [M,4]


