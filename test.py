
import torch 
import torch.nn
import numpy as np

'''
x=torch.rand(2,3,4)
y=torch.tensor([1,2,3,3,2,1],dtype=torch.long)
print(x.shape)
for i, c in enumerate(x.split([1,1,2], -1)):
    print("------------------------------------")
    print(i)
    print(c)
assert True
'''
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)
print(col_ind)
print(row_ind)
cost[row_ind, col_ind].sum()
print(cost[row_ind, col_ind])