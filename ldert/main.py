import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch  
from  torch.utils.data import DataLoader,DistributedSampler

#import datasetes
import util.misc as utils
#from datasets import build_dataset,get_coco_api_from_dataset
#from engine import evaluate,train_one_epoch
#from models import build_model

def get_args_parser():
    parser=argparse.ArgumentParser('Set transformer detector',add_help=False)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--batch_size',default=1e-5,type=float)
    parser.add_argument('--weight_decay',default=1e-4,type=float)
    parser.add_argument('--epochs',default=300,type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_file', default='coco')
    return parser

def main(args):
    #utils.init_distributed_mode(args)
    #print("git:\n {}\n".format(utils.get_sha()))

    #cuda or cpu
    device=torch.device(args.device)
    # print(device)
    # 根据rank来获得当前进程的seed数
    seed=args.seed+utils.get_rank()
    print(seed)
    # 各个组件加载随机数种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    

if __name__ =='__main__':
    '''
    @ description 描述
    @ parents 提前包含的参数
    '''
    parser=argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args=parser.parse_args()
    
    if args.output_dir:
        '''
        @ parents：是否创建父目录，True等同mkdir -p;False时，父目录不存在，则抛出FileNotFoundError
        @ exist_ok：在3.5版本加入。False时，路径存在，抛出FileExistsError;True时，FileExistsError被忽略
        '''
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)