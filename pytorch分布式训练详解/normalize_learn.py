# transforms.Normalzie()
# 看了许多文章，都是说：transform.Normalize()通过公式
# 即同一纬度的数据减去这一维度的平均值，再除以标准差，将归一化后的数据变换到【-1,1】之间。可真是这样吗？？
# 并非如此，而是通过这段代码，让数据变成正太分布


# 求解mean与std
# 我们需要求得一批数据的mean与std，代码如下
import torch
import numpy as np 
from torchvision import transforms


# 这里以上术创建的数据为例子
data = np.array([
                [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
                [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],
                [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]
        ],dtype='uint8')

# 将数据转为C,W,H，并归一化到[0,1]  
data=transforms.ToTensor()(data)

# 需要对数据进行扩维，增加batch维度
data=torch.unsqueeze(data,0)

nb_samples = 0.
# 创建3维的空列表
channel_mean=torch.zeros(3)
channel_std=torch.zeros(3)
print(data.shape)
N,C,H,W=data.shape[:4]
data = data.view(N, C, -1)     #将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
print(data.shape)
#展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
channel_mean += data.mean(2).sum(0)  
#展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
channel_std += data.std(2).sum(0)
#获取所有batch的数据，这里为1
nb_samples += N
channel_mean /= nb_samples
channel_std /= nb_samples
print(channel_mean, channel_std)

for i in range(3):
    data[i,:,:]=(data[i,:,:]-channel_mean[i])/channel_std[i]
print(data)
data = transforms.Normalize(channel_mean, channel_std)(data)
print(data)

# 结论
# 经过这样处理后的数据符合标准正态分布，即均值为0，标准差为1。使模型更容易收敛。并非是归于【-1，1】！！









