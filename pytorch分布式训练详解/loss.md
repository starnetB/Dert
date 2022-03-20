# NLLLoss
负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。     

对于包含N个样本的batch数据 D(x, y)，x 是神经网络的输出，并进行归一化和对数化处理。y是样本对应的类别标签，每个样本可能是C种类别中的一个。 

$l_n 为第n个样本对应的loss,0\leq y_n \leq C-1 $    
$ l_n=-w_{y_n}x_{n,y_n}$      
* 这里的x进行过归一化和对数处理     

$ w$用于多个类别之间的样本不平衡问题      
$ w_c =weight[c] \cdot 1$ &nbsp;  &nbsp;  &nbsp; &nbsp; $\left\{c \neq ignore_index \right\}$      

pytorch中通过torch.nn.NLLLoss类实现，也可以直接调用F.nll_loss函数，代码中的weight即是$w$。size_average与reduce已经弃用。reduction有三种取值mean,sum,none,对应不同的$l(x,y)$,默认为mean,对应与一般情况下整体loss的计算。     
$$ L=\left\{ l_1,\cdots,l_N \right\}$$
$$ l(x,y)=\begin{cases} L                  & if \ reduction ='none'  \\
\sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n&if \ reduction='mean' \\
\sum_{n-1}^N l_n& if\ reduction='sum' \end{cases}
$$
参数ignore_idex对应与忽视的类别，即该类别的误差不计loss,默许-100,例如，将padding处的类别设置为ignore_index

***   

# LogSoftmax   
pytorch中使用torch.nn.LogSoftmax函数对神经网络的输出进行归一化和对数化    
$$ LogSoftmax(x_i)=log(\frac{exp(x_i)}{\sum_j exp(x_j)}) $$

# CrossEntropyLoss
交叉熵损失函数，用于处理多分类问题，输入是未归一化神经网络输出。    
$$ CrossEntropyLoss(x,y)=NLLLoss(logSoftmax(x),y$$ 
对于包含N个样本的batch数据D(x,y),x是神经网络未归一化的输出。y是样本对应的类别标签，每个样本可能是C种类别中的一个。      
$l_n 为第n个样本对应的loss$,$\ \ \ 0 \leq y_n \leq C-1$    

$ l_n=-w_{y_n}(log \frac{exp(x_{n,y_n})}{\sum_j^C exp(x_{n,j})}) $     

$ l_n=-w_{y_n}(x_{n,y_n} + log(\sum_j^C exp(x_{n,j})))$   

pytorch中通过torch.nn.CrossEntropyLoss类实现，也可以直接调用F.cross_entropy 函数，代码中的weight即是w。size_average与reduce已经弃用。reduction有三种取值mean, sum, none，对应不同的返回$\ell(x, y).$ 默认为mean，对应于一般情况下整体loss的计算。      

$$ L=\left\{l_1,l_2,\cdots,l_N \right\}$$

$$\ell(x,y) = \begin{cases} L & if \ reduction='none'   \\
\sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}}  l_n & if\ reduction='mean' \\
\sum_{n=1}^N l_n & if\ reduction='sum' \end{cases}$$   
验证CrossEntropyLoss(x,y)=NLLLoss(logSoftmax(x),y)    
* 这里的x没有进行归一化



