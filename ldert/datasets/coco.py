from pathlib import Path
from statistics import mode

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self,img_folder,ann_file,transforms,return_masks):
        super(CocoDetection,self).__init__(img_folder,ann_file)
        self._transforms=transforms
        self.prepare=C

class ConvertCocoPolysToMask(object):
    def __init__(self,return_masks=False):
        self.return_masks=return_masks
    
    def __call__(self, image,target):
        w,h=image.size

        image_id=target["image_id"]
        image_id=torch.tensor([image_id])
        # 把target中的annotations拿出来
        anno=target["annotations"]
        # 把没有iscrowd 和iscrowd==0的obj拿出来
        anno=[obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"]==0]
        # 把boxes拿出来 x_l,y_t,x_r,y_b
        boxes=[obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes=torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
        # boxes.shape[0] 一张图片中boxes的数量 
        boxes[:,2:]+=boxes[:,:2]  #使得后面两个坐标变成，x_r,y_b
        boxes[:,0::2].clamp_(min=0,max=w) #不要超限
        boxes[:,1::2].clamp_(min=0,max=h)  

        classes =[obj["category_id"] for obj in anno]
        classes =torch.tensor(classes,dtype=torch.int64)

        if self.return_masks:
            # 把掩码拿出来
            segmentations=[obj["segmentation"] for obj in anno]
            masks=convert_coco_poly_to_mask(segmentations,h,w)

def convert_coco_poly_to_mask(segmentations,height,width):
    masks=[]
    for polygons in segmentations:
        rles=coco_mask.frPyObjects(polygons,height,width)
        mask=coco_mask.decode(rles)

def make_coco_transforms(image_set):
    normalize=T.Compose([ 
        T.ToTensor(),  #将img变成Tensor
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # mean,std
    ])

    scales=[480,512,544,576,608,640,672,704,736,768,800]

    if image_set=='train':
        return T.Compose([  
            # 随机水平反转
            T.RandomHorizontalFlip(),
            # 操作二选一
            # 
            T.RandomSelect(
                T.RandomResize(scales,max_size=1333),
                T.Compose([
                    # 随机改变大小
                    T.RandomResize([400,500,600]),
                    # 随机裁剪
                    T.RandomSizeCrop(384,600),
                    # 随机改变大小
                    T.RandomResize(scales,max_size=1333),
                ])
            ),
            # 归一化
            normalize,
        ])

    if image_set=='val':
        return T.Compose([  
            T.RandomResize([800],max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set,args):
    root=Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode ="instances" 
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder,ann_file=PATHS[image_set]
    dataset =CocoDetection(img_folder,ann_file,transformss=make_coco_transforms(image_set),return_masks=args.masks)
    return dataset