# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:34:43 2019

@author: Mattis
"""

import numpy as np
import scipy as sp
from scipy.linalg import solve_banded

a = np.array([[5, 2, -1, 0, 0],
              [1, 4, 2, -1, 0],
              [0, 1, 3, 2, -1],
              [0, 0, 1, 2, 2],
              [0, 0, 0, 1, 1]])


print(a)

l = len(a)
u = 2
d = 1

ab = np.zeros((u+d+1,l))

#for i in range(l):
#    for j in range(u+d+1):
#        j += i
#        if j > l-1: j = l-1
#        if j-d < 0: j = d
#        ab[u+i+d-j,j-d] = a[i,j]
#    for j in range(d):
#        j += i
##        if j < 1: j = 1
#        ab[u+1+j,j] = a[i,j]
        
#for i in range(l-3):
#    for j in range(u+1):
##        if j > l-1: j = l-1
#        ab[u+i-j,j] = a[i,j]
#        print(ab)
        
for i in range(l):
    for j in range(l):
        if i-j < d+1 and j-i < u+1:
            ab[u+i-j,j] = a[i,j]
print(ab)

b = np.array([[0],[1],[2],[2],[3]])

x = solve_banded((d,u),ab,b)
print(x)




        






#ab[u + i - j, j] == a[i,j]