# 本教程致力于transformer的pytorch实现   
* BERT和TransFormer在2018年被提出，引起了自然语言处理方面革命性的变革  
* transformer从翻译问题开始，已经被用于自然语言处理的各个方面  

## 位置编码的公式 
$$PE_{(pos,2i)}=\sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$
$$PE_{(pos,2i+1)}=\cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$$
* 这里需要解释的参数，包括以下内容
  * pos表示在seq中的位置
  * i则表示杂embed_size中的位置 
* transformer.py中位置编码有点不一样  
$$PE_{(pos,2i)}=\sin(pos \times e^{\frac{-2i \log{10000.0}}{d_{model}}})$$
$$PE_{(pos,2i)}=\cos(pos \times e^{\frac{-2i \log{10000.0}}{d_{model}}})$$