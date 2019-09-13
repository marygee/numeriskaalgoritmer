# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:29:18 2019
@author: selle
"""
from  scipy import *
from  pylab import *

class bspline:
    def __init__(self, coord):
        self.coord = coord
        k = len(self.coord)
        self.u_grid = linspace(0, 10, k+1)
#        N0 = empty(k)
#        N1 = empty(k)
#        N2 = empty(k)
#        N3 = empty(k)
#        for i in range([1, k]): # den ska inte inkludera 0??
#            N0[i] = lambda x : heaviside(x - u_grid[i-1], 1) - heaviside(x - u_grid[i], 1)
#        for i in range([1, k]):
#            N1[i] = lambda x : (x - u_grid[i-1])/(u_grid[i+1-1]-u_grid[i-1])*N0[i] + (u_grid[i+1]-x)/(u_grid[i+1]-u_grid[i])*N0[i+1]
#        for i in range([1, k]):
#            N2[i] = lambda x : (x - u_grid[i-1])/(u_grid[i+2-1]-u_grid[i-1])*N1[i] + (u_grid[i+2]-x)/(u_grid[i+2]-u_grid[i])*N1[i+1]
#        for i in range([1, k]):
#            N3[i] = lambda x : (x - u_grid[i-1])/(u_grid[i+3-1]-u_grid[i-1])*N2[i] + (u_grid[i+3]-x)/(u_grid[i+3]-u_grid[i])*N2[i+1]
        
    def __call__(self, u, d):
        self.u = u
        self.d = d
        i_hot = self.findhot(u)
        s = self.blossom(self.d, self.u, i_hot)
        self.plot()
        return s
    
    def findhot(self, u):
        i_hot = (self.u_grid > u).argmax() - 1
        return i_hot
    
    def blossom(self, d, u, i_hot):
        d_hotx = zeros((4, 4))
        d_hoty = zeros((4, 4))
        d_x = [x[0] for x in d] 
        d_y = [x[1] for x in d] 
        hotstart = i_hot-2
        hotend = i_hot+2
        d_hotx[0,:] = d_x[hotstart:hotend]
        d_hoty[0,:] = d_y[hotstart:hotend]
        for t in range(0, 3): # (0, 1, 2)
            for index in range(0, 3-t): # (0, 1, 2) for t = 0; (0, 1) for t = 1; 0 for t = 2
                alpha = self.alpha(u, i_hot, t, index)
                d_hotx[t+1, index] = alpha*d_hotx[t, index] + (1-alpha)*d_hotx[t, index+1]
                d_hoty[t+1, index] = alpha*d_hoty[t, index] + (1-alpha)*d_hoty[t, index+1]
        sx = d_hotx[3, 0]
        sy = d_hoty[3, 0]
        return (sx, sy)
        
    def alpha(self, u, i_hot, t, index):
        i_leftmost = i_hot - 2 + index + t
        u_leftmost = self.u_grid[i_leftmost]
        i_rightmost = i_hot +1 + index
        u_rightmost = self.u_grid[i_rightmost]
        return (u_rightmost - u) / (u_rightmost - u_leftmost)
    
    def plot(self):
        import matplotlib.pyplot as plt
        data =  self.coord
        x_val = [x[0] for x in data]
        y_val = [x[1] for x in data]
        plt.plot(x_val,y_val,'or')
        u = linspace(0, 10, 100)
        s = zeros(100)
        for j in range(0, 100):
            s[j] = self.__call__(u[j], self.d)
        plt.plot(u, s)
        plt.show()
        
        
d = [(1, 2), (2, 3), (4, 6), (8, 4), (4, 4), (3, 3)]
b = bspline(d)
s = b(5, d)

# Vid körning failar koden pga att i_hot blir skum vilket smittar av sig på hotstart och hotend...