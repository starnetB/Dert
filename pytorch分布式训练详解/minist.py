from tkinter.tix import Tree
from turtle import back, forward
from sklearn.utils import shuffle
import torch 
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets,transforms
import argparse
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

'''
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
'''
DATA_DIR='./data/mnist'    
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        '''
        @ in_channels:
        @ out_channels:
        @ kernel_size=5
        @ stride=1
        '''
        self.conv1=nn.Conv2d(1,20,5,1)
        self.conv2=nn.Conv2d(20,50,5,1)
        self.fc1=nn.Linear(4*4*50,500)
        self.fc2=nn.Linear(500,10)
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        '''
        @ x input
        @ kernel_size  =2 /=(3,2)
        @ stride =2  /=(2,1)
        @ padding =0 /=(1,1)
        @ dilation =1 /=(2,2)  如果等于1,则卷积核则不变，如果等于2,则卷积分核插入一个空格
        '''
        x=F.max_pool2d(x,2,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2,2)
        x=x.view(-1,4*4*50)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)
    
def train(model,train_loader,optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target=data.to(DEVICE),target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
       

def test(model,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target =data.to(DEVICE),target.to(DEVICE)
            output=model(data)
            test_loss +=F.nll_loss(output,target,reduction='sum').item() #sum up batch loss
            # output.max(1,keepdim=True)[0]  tensor 返回每行最大的那个元素 size [64,1]
            # output.max(1,keepdim=True)[1]  LongTensor 返回每行最大那个元素的索引 size [64,1]
            pred=output.max(1,keepdim=True)[1]  #  get the index of the max log-probability
            correct +=pred.eq(target.view_as(pred)).sum().item()
    test_loss /=len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))



def load_data(dist,batch_size=64,test_batch_size=64):
    '''
    num_worker:1  同时可以启动的线程数量
    pin_memory:   是否将数据放在内存中
    '''
    train_kwargs={'num_workers':1,'pin_memory':True}
    test_kwargs={'num_workers':1,'pin_memory':True}

    '''
    @ datasets.MNINST()
     @ ROOT=DATA_DIR, 加载数据的根目录
     @ train=True,是否从train.py加载训练数据集
     @ download=True,如果没有数据是否从网上下载
     @ transform ,对数据做的操作
    @ transforms.Compose([])对数据集做的操作组合 
        @ transform.ToTensor()   
            @ 将输入数据shape W,H,C->C,W,H
            @ 将所有数除以255,将数据归一化到[0,1]
        @ transform.Normalize()
            @ output=(input-mean)/std
    '''
 
    train_data_set=datasets.MNIST(DATA_DIR,train=True,download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # 这里 Normalize()的使用方法，详情请见normalize_learn.py
                                transforms.Normalize((0.1307,),(0.3001,))
                            ]))
             
    if  dist.is_initialized():
        # 如果采用分布式训练, 使用DistributedSampler让每个worker拿到训练数据集不同的子集
        datasampler = DistributedSampler(train_data_set)
        # sampler shuffle must be False
        train_kwargs.update({'sampler':datasampler,
                             'shuffle':False})
    

    train_loader=torch.utils.data.DataLoader(train_data_set,batch_size,**train_kwargs)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR,train=False,transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,),(0.3001,))
            ])),
        batch_size=test_batch_size,shuffle=True,**test_kwargs)
    
    return train_loader,test_loader

def main():

    parser=argparse.ArgumentParser(description="pytorch MNINST Example")
    parser.add_argument('--backend',type=str,help='DIstributed backend',
                        choices=[dist.Backend.GLOO,dist.Backend.NCCL,dist.Backend.MPI],
                        default=dist.Backend.NCCL)
    parser.add_argument('--init-method', default=None, type=str,
                        help='Distributed init_method')
    parser.add_argument('--rank', default=-1, type=int,
                        help='Distributed rank')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='Distributed world_size')
    args = parser.parse_args()

    # 集群组初始化
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            rank=args.rank,
                            world_size=args.world_size
                            )
    # 如果采用分布式训练, 使用DistributedSampler让每个worker拿到训练数据集不同的子集
    train_loader,test_loader=load_data(dist)  
    model=Net().to(DEVICE)

    model=nn.parallel.DistributedDataParallel(model)
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(1,2):
        train(model,train_loader,optimizer,epoch)
        test(model,test_loader)

if __name__=='__main__':
    main()
    
            



            