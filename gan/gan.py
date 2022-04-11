from fcntl import DN_RENAME
from tkinter.dialog import DIALOG_ICON
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F   
import torch.optim as optim   

from torch.autograd import Variable
  
matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False

# Data_params 
# 用于生成正太分布曲线
data_mean=4
data_stddev=1.25  

# Model params 
g_input_size=1   #Random noise dimension coming into generator, per output vector
g_hidden_size=50  #Generator complexity
g_output_size = 1    # size of generated output vector
d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'

minibatch_size=d_input_size

d_learning_rate=2e-4
g_learning_rate=2e-4

optim_betas=(0.9,0.999)  
num_epochs=30000
print_interval=200
d_steps=1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator 
g_steps=1  

# ## uncomment only one of theses
## ### Uncomment only one of these
#(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" %(name))



def get_distribution_sampler(mu,sigma):
    return lambda n:torch.Tensor(np.random.normal(mu,sigma,(1,n))) #Gaussin

def get_generator_input_sampler():
    return lambda m,n:torch.randn(m,n)

class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Generator,self).__init__()
        self.map1=nn.Linear(input_size,hidden_size)
        self.map2=nn.Linear(hidden_size,hidden_size)
        self.map3=nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x=F.elu(self.map1(x))
        x=F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Discriminator,self).__init__()
        self.map1=nn.Linear(input_size,hidden_size)
        self.map2=nn.Linear(hidden_size,hidden_size)
        self.map3=nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x=F.elu(self.map1(x))
        x=F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d),np.std(d)]
    

# #### DATA:Target data and generator input data
def decorate_with_diffs(data,exponent):
    mean=torch.mean(data.data,1)  #data [batch_size,data_d],[batch_size,1]
    #广播乘法
    mean_broadcast=torch.mul(torch.ones(data.size()),mean.tolist()[0])
    diffs=torch.pow(data-Variable(mean_broadcast),exponent)
    return torch.cat([data,diffs],1)


d_sampler=get_distribution_sampler(data_mean,data_stddev) #[1:100]
gi_sampler=get_generator_input_sampler()

# g_input_size 1
# g_hidden_size 50
# g_output_size 1
# bs=100
G=Generator(input_size=g_input_size,hidden_size=g_hidden_size,output_size=g_output_size)

# d_input_size 100
# d_hidden_size 50
# d_output_size 1
D=Discriminator(input_size=d_input_func(d_input_size),hidden_size=d_hidden_size,output_size=g_output_size)


criterion=nn.BCELoss()

d_optimizer=optim.Adam(D.parameters(),lr=d_learning_rate,betas=optim_betas)
g_optimizer=optim.Adam(G.parameters(),lr=g_learning_rate,betas=optim_betas)


for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # I,Train D on real +fake
        D.zero_grad()

        # 1A:Train D on real
        d_real_data=Variable(d_sampler(d_input_size))  #[1:100]
        d_real_dicision=D(preprocess(d_real_data))  #[1-200]->[1,1]
        d_real_error=criterion(d_real_dicision,Variable(torch.ones((1,1)))) #ones=True
        d_real_error.backward() # compute/store graddients,but don't change params

        # 18:Train D on fake
        d_gen_input=Variable(gi_sampler(minibatch_size,g_input_size))  #  [100,1]
        # [100,1]
        d_fake_data=G(d_gen_input).detach()  # 避免G被训练  # detach to aovid trainning G on thes labels
        d_fake_decision=D(preprocess(d_fake_data.t()))  #[1-100]->[1-200]->[1.1]
        d_fake_error=criterion(d_fake_decision,Variable(torch.zeros((1,1))))
        d_fake_error.backward()
        d_optimizer.step()
    
    for g_index in range(g_steps):
        # Z.Train G on D's response (but Do Not train D on these lables)
        G.zero_grad()

        gen_input=Variable(gi_sampler(minibatch_size,g_input_size))
        g_fake_data=G(gen_input)
        dg_fake_decision=D(preprocess(g_fake_data.t()))
        g_error=criterion(dg_fake_decision,Variable(torch.ones((1,1))))
        g_error.backward()
        g_optimizer.step  #

    if epoch % print_interval==0:
        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                            extract(d_real_error)[0],
                                                            extract(d_fake_error)[0],
                                                            extract(g_error)[0],
                                                            stats(extract(d_real_data)),
                                                            stats(extract(d_fake_data))))


if matplotlib_is_available:
    print("Plotting the generated distribution...")
    values = extract(g_fake_data)
    print(" Values: %s" % (str(values)))
    plt.hist(values, bins=50)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram of Generated Distribution')
    plt.grid(True)
    plt.show()