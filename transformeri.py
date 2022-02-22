from asyncio.proactor_events import _ProactorBaseWritePipeTransport
from curses import KEY_REPLACE
import math
from tkinter import PROJECTING
from turtle import forward, pos
from sympy import N  
import torch  
import numpy as np 
import torch.nn as nn   
import torch.optim as optim
import torch.utils.data as Data  

#S:Symbol that shows starting of decoding input
#E:Symbol that shows end of decoding output
#P:Symbol that will fill in blank sequence if current batch data size is short that time steps

sentences=[
    #enc_input                #dec_input                  #dec_ouput
    ['ich mochte ein bier P','S i want a beer .','i want a beer .E'],
    ['ich mochte ein cola P','S i want a coke .','i want a coke .E']
]

# Padding Should be Zero
# P的词语索引应该是零    
src_vocab={'P':0,'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
src_vocab_size=len(src_vocab)
# encode input 

tgt_vocab={'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
idx2word={i:w for i,w in enumerate(tgt_vocab)} # i is key of dict  #w is key of dict
tgt_vocab_size=len(tgt_vocab)

src_len=5 #enc_input max sequence length
tgt_len=6 #dec input(=dec_output) max sequence length  

# Transformer Parameters
d_model=512 #Embedding size 
d_ff=2048   #FeedForward dimensiong 前馈网络维度
d_k=d_v=64  # dimension of k(=Q),V
n_layers=6  # number of Encoder of Decoder Layer
n_heads=8   # number of heads in Multi-Head Attention  #多少个头

def make_data(sentences):  # 放入句子集，生成数据集合
    enc_inputs,dec_inputs,dec_outputs=[],[],[]
    # 从下面语句中，我们可以发现句子集的shape
    # shape[0]  就是句子的数量，
    # shape[1]  就是代表句子的三种格式：
        # 0 就是句子原来的样子，后面填充P
        # 1 就是句子前面加S，后面家点号 8
        # 2 就是句子后面加个点 8，加上结束符
    for i in range(len(sentences)):
        enc_input=[[src_vocab(n) for n in sentences[i][0].split()]]  #[[1,2,3,4,0],[1,2,3,5,0]]
        dec_input=[[tgt_vocab(n) for n in sentences[i][1].split()]]   #[[6,1,2,3,4,8],[6,1,2,3,5,8]]
        dec_output=[[tgt_vocab(n) for n in sentences[i][2].split()]]  #[[1,2,3,4,8,7],[1,2,3,5,8,7]]

        # n代表句子的数量
        # d代表句子的维度
        enc_inputs.extend(enc_input)  #n,编码输入句子的长度
        dec_inputs.extend(dec_input)   #n,解码输入的句子长度
        dec_outputs.extend(dec_output)  #n,解码输出的句子长度  

    return torch.LongTensor(enc_inputs),torch.LongTensor(dec_inputs),torch.LongTensor(dec_outputs)
        

class MyDataSet(Data.Dataset):
    def __init__(self,enc_inputs,dec_inputs,dec_outputs) -> None:
        super(MyDataSet,self).__init__()
        self.enc_inputs=enc_inputs
        self.dec_inputs=dec_inputs
        self.dec_outputs=dec_outputs
    
    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx],self.dec_inputs[idx],self.dec_outputs[idx]

enc_inputs,dec_inputs,dec_outputs=make_data(sentences)

# 2就是batch_size
# True 就是 Shuffle
loader=Data.DataLoader(MyDataSet(enc_inputs,dec_inputs,dec_outputs),2,True)

# 下面是位置编码公式
# 公式内容请见readme.md
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropuout=nn.Dropout(p=dropout)  #用于丢弃向量中的一部分内容，把这部分内容变成0.00

        pe=torch.zeros(max_len,d_model)  #一个zeros的Tensor，序列的最大长度是5000，这个序列位置的最大长度，d_model就是embedding_size=512
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #生成5000长度的初始化 位置，然后后面在扩展一个维度，用于填充
        div_term=torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)  #维度发生变化[seq,batch_size=1,embedding_size=512]
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        '''
        x:[seq_len,batch_size,d_model]
        '''
        x=x+self.pe[:x.size(0),:]  #利用位置编码进行相加，最后，我们只去我们想要的那部分编码，多于的我们不要
        return self.dropout(x)     #每次生成的数据中，dropout变成0，这样不会出现问题吗？
    
def get_attn_pad_mask(seq_q,seq_k):
    '''
    seq_q:[batch_size,seq_len]
    seq_k:[batch_size,seq_len]
    seq_len could be src_len or it could be tgt len 
    seq_len in seq_q and seq_len in seq_k maybe not equal
    也就是目标字符与元字符集合的长度可能不相等
    '''
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()

    #eq(zero) is PAD token
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)  #[batch_size,1,len_k] False is masked 
    return pad_attn_mask.expand(batch_size,len_q,len_k) # [batch_size,len_q,len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq:[batch_size,tgt_len]
    '''
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)] # attn_shape=[seq.size(0),seq.size(1),seq.size(1)]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1) #upper triangular matrix  k=1不保留对角线，对角线上的内容为0
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size,tgt_len,tgt_len]

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
    
    def forward(self,Q,K,V,attn_mask):
        '''
        Q:[batch_size,n_heads,len_q,d_k]  #查询向量集合
        K:[batch_size,n_heads,len_k,d_k]  #key向量集合
        V:[batch_size,n_heads,len_v,d_v]  #值向量集合
        attn_mask:[batch_size,n_heads,seq_len,seq_len]
        '''
        #scores:[batch_size,n_heads,len_q,len_k]
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k) 
        scores.mask_fill_(attn_mask,-1e9)  #Fill elements of self tensor with value where mask i 在对应的地方填充掩码

        attn=nn.Softmax(dim=-1)(scores)  # 在最后一个维度上计算sotfmax
        context =torch.matmul(attn,V)  #计算每一列的最终结果，一行对应与一个单词的计算结果
        return context,attn  

# 上面那一步为计算多头的计算结果，我们这里的这一步是前面那一部的前一部，也就是计算多头的Q，K，V

class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadAttention,self).__init__()
        self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(n_heads*d_v,d_model,bias=False)
    
    def forward(self,input_Q,input_K,input_V,attn_mask):
        '''
        input_Q:[batch_size,len_q,d_model
        input_K:[batch_size,len_k,d_model
        input_V:[batch_size,len_v(=len_k),d_model]
        # 这里我的理解就是他把一个enc_input，分成了三分，然后在计算对应的词
        attn_mask:[batch_size,seq_len,seq_len]
        '''
        residual,batch_size=input_Q,input_Q.size(0)
        #(B,S,D)-proj->(B,S,D_new)-split->(B,S,H,W)-trans->(B,H,S,W)
        Q=self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)  #Q:[batch_size,n_head,len_q,d_k]
        K=self.W_K(input_K).view(batch_size,-1,n_heads,d_k).transpose(1,2)  #K:[batch_size,n_head,len_k,d_k]
        V=self.W_V(input_V).view(batch_size,-1,n_heads,d_v).transpose(1,2)  #V:[batch_size,n_head,len_v,d_v]

        #attn_mask:[batch_size,n_heads,seq_len,seq_len]
        #repeat(1,n_heads,1,1)
        # 第二个维度赋值n_heads个头，其他维度不变
        attn_mask=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)

        #context:[batch_size,n_heads,len_q,d_v],
        #attn:[batch_size,n_heads,len_q,len_k]  #值没有加之前，但softmask已经计算好的中间状态
        context,attn=ScaleDotProductAttention()(Q,K,V,attn_mask)
        #context:[batch_size,len_q,n_heads*d_v]
        context=context.transpose(1,2).reshape(batch_size,-1,n_heads*d_v) 
        output=self.fc(context)  #[batch_size,len_q,d_model]
        return nn.LayerNorm(d_model).cuda()(output+residual),attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )

    
    def forward(self,inputs):
        '''
        inputs:[batch_size,seq_len,d_model]
        '''
        residual=inputs
        output=self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output+residual)  #[batch_size,seq_len,d_model]
    

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.pos_fnn=PoswiseFeedForwardNet()  

    def forward(self,enc_inputs,enc_self_attn_mask):
        '''
        enc_inputs:[batch_size,src_len,d_model]
        enc_self_attn_mask:[batch_size,src_len,src_len]
        '''
        # enc_outputs:[batch_size,src_len,d_model];
        # attn:[batch_size,n_heads,src_len,src_len];
        # enc_inputs to same Q,K,V
        enc_outputs,attn=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.pos_fnn(enc_outputs)  # enc_outputs:[batch_size,src_len,d_model]
        return enc_outputs,attn
    
class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super(DecoderLayer,self).__init__()
        self.dec_self_attn=MultiHeadAttention()
        self.dec_enc_attn=MultiHeadAttention()
        self.pos_fnn=PoswiseFeedForwardNet()

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        '''
        dec_inputs:[batch_size,tgt_len,d_model]      #解码输入
        enc_outputs:[batch_size,src_len,d_model]     #编码输出，用于编码解码注意力层的使用
        dec_self_attn_mask:[batch_size,tgt_len,tgt_len]  #解码注意力层的掩吗
        dec_enc_attn_mask:[batch_size,tgt_len,src_len]   #编码解码注意力层的掩码
        '''
        # dec_outputs:[batch_size,tgt_len,d_model]
        # dec_self_attn:[batch_size,n_heads,tgt_len,tgt_len]
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        # dec_outputs:[batch_size,tgt_len,d_model]
        # dec_enc_attn:[batch_size,h_heads,tgt_len,src_len]
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_outputs=self.pos_fnn(dec_outputs)  # [batch_size,tgt_len,d_model]
        return dec_outputs,dec_self_attn,dec_enc_attn
    
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(self).__init__()
        self.src_emb=nn.Embedding(src_vocab,d_model)  #将one-hot 变成vocab_size
        self.pos_emb=PositionalEncoding(d_model)   #产生位置编码，然后赋值给你的inputs
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        '''
        enc_inputs:[batch_size,src_len]
        '''
        enc_outputs=self.src_emb(enc_inputs) #[batch_size,src_len,d_model]
        enc_outputs=self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)  #[batch_size,src_len,d_model]
        enc_self_attn_mask=get_attn_pad_mask(enc_inputs,enc_inputs)  #[batch_size,src_len,src_len]
        enc_self_attns=[]
        for layer in self.layers:
            #enc_outputs:[batch_size,src_len,d_model]
            #enc_self_attn:[batch_size,n_heads,src_len,src_len]
            enc_outputs,enc_self_attn=layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs,enc_self_attns

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()
        self.tgt_emb=nn.Embedding(tgt_vocab,d_model)
        self.pos_emb=PositionalEncoding()
        self.layers=nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    
    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        '''
        dec_inputs:[batch_size,tgt_len]  #embedding 可以直接转化
        enc_inputs:[batch_size,src_len]
        ent_outputs:[batch_size,src_len,d_model]
        '''
        dec_outputs=self.tgt_emb(dec_inputs)   #[batch_size,tgt_len,d_model]
        dec_outputs=self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1).cuda()  #[batch_size,tgt_len,d_model]
        dec_self_attn_pad_mask=get_attn_pad_mask(dec_inputs,dec_inputs).cuda()  #·[batch_size,tgt_len,tgt_len]  #计算自注意力层的掩码
        dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size,tgt_len,tgt_len]  #输出一个对应的上三角
        dec_self_attn_mask=torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0)  #[batch_size,tgt_len,tgt_len]
        # 以上解码器的自掩码层，不像编码器的掩码层，
        # 只要关注过去的信息就可以了，因此叠加一个上三角的掩码层
        dec_enc_attn_mask=get_attn_pad_mask(dec_inputs,enc_inputs)  #[batch_size,tgt_len,src_len]

        dec_self_attns,dec_enc_attns=[],[]
        for layer in self.layers:
            #dec_outputs:[batch_size,tgt_len,d_model]
            #dec_self_attn:[batch_size,n_heads,tgt_len,tgt_len]
            #dec_enc_attn:[batch_size,n_heads,tgt_len,src_len]
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns   

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer,self).__init__()
        self.encoder=Encoder().cuda()
        self.decoder=Decoder().cuda()
        self.projection=nn.Linear(d_model,tgt_vocab,bias=False).cuda()
    
    def forward(self,enc_inputs,dec_inputs):
        '''
        enc_inputs:[batch_size,src_len]
        dec_inputs:[batch_size,tgt_len]
        '''
        #enc_outputs:[batch_size,src_len,d_model]
        #enc_self_attns:[n_layer,batch_size,n_heads,src_len,src_len]
        enc_outputs,enc_self_attns=self.encoder(enc_inputs)
        #dec_outputs:[batch_size,tgt_len,d_model]
        #dec_self_attns:[n_layer,batch_size,n_heads,tgt_len,tgt_len]
        #dec_enc_attn:[n_layer,batch_size,n_heads,tgt_len,src_len]
        dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits=self.projection(dec_outputs)  #dec_logits:[batch_size,tgt_len,tgt_vocab_size]
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns

model=Transformer().cuda()
criterion=nn.CrossEntropyLoss(ignore_index=0)
optimizer=optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)   #landa=0.99 用前100个梯度周期来更新

for epoch in range(1000):
    for enc_inputs,dec_inputs,dec_outputs in loader:
        '''
        enc_inputs:[batch_size,src_len]
        dec_inputs:[batch_size,rgt_len]
        dec_outputs:[batch_size,tgt_len]
        '''
        enc_inputs,dec_inputs,dec_outputs=enc_inputs.cuda(),dec_inputs.cuda(),dec_outputs.cuda()
        #output:[batch_size*tgt_len,tgt_vocab_size]
        outputs,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs,dec_inputs)
        loss=criterion(outputs,dec_outputs.view(-1)) 
        print('Epoch:','%04d'%(epoch+1),'loss=','{:,6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def greedy_decoder(model,enc_input,start_symbol):
    """
    为了简单期间，贪婪解码器是k=1的束搜素，当我们不知道目标序列输入的时候，这是十分必要的。
    因此我们尝试形成一个一个的输入目标单词，然后喂给transformer\
    @ parm model ,Transformer Model
    @ param enc_input:The encode input
    @ param start symbol :The start symbol,In this example it is "S" which corresponds to index 4
    @ return:The target input
    """
    # enc_input [batch_size=1,seq]
    # enc_outputs [batch_size=1,seq,512]
    # enc_self_attns [nLayer,batch_size=1,nheads,seq,seq]
    enc_outputs,enc_self_attns=model.encoder(enc_input) 
    # dec_input []  
    # torch.zeros(1,0) 是一个空的tensor
    dec_input=torch.zeros(1,0).type_as(enc_input.data)
    terminal=False
    next_symbol=start_symbol  #start_symbol =6
    # dec_input.detach()  剥离梯度计算
    while not terminal:
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype)],-1)
        # dec_input [batch_size=1,seq=1]
        
        dec_outputs,_,_=model.decoder(dec_input,enc_input,enc_outputs)
        #dec_outputs [batch_size=1,seq=1，512]

        projected=model.projection(dec_outputs)
        #projected [batch_size=1,seq=1,vocab_size]

        prob=projected.squeeze(0).max(dim=-1.keepdim=False)
        #prob [size=1]

        next_word=prob.data[-1]
        #输出预测词的vocab索引

        next_symbol=next_word
        if next_symbol==tgt_vocab["."]:
            terminal=True
        print(next_word)
    return dec_input

# 取出一个batch中的数据
enc_inputs,_,_=next(iter(loader))
# [batch_size,seq]
# [seq].view(-1,1)  ->[batch_size=1,seq]
for i in range(len(enc_inputs)):
    greedy_dec_input=greedy_decoder(model,enc_inputs[i].view(1,-1),start_symbol=tgt_vocab["S"])
    # greedy_dec_input [batch_size=1,tgt_seq]
    predict,_,_,_=model(enc_inputs[i].view(1,-1),greedy_dec_input)
    # predice [batch_size=1*seq,vocab_size]
    predict=predict.data(1,keepdim=True)[1]  # predict=[batch_size=1*seq,1]
    
    #
    print(enc_inputs[i],"->",[idx2word[n.item()] for n in predict.squeeze()])

# tensor([1,2,3,4,0]) -> ['i','want','a','beer',','.'E']