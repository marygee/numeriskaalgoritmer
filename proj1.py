# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:29:18 2019
@author: selle
"""
from  scipy import *
from  pylab import *
import  unittest

class bspline:
    def __init__(self, coord, u_grid):
        self.coord = coord
        k = len(self.coord)
        self.u_grid = u_grid

    def __call__(self, u, d, blossom_i):
        self.u = u
        self.d = d
        i_hot = self.findhot(u)
        [d_hotx, d_hoty] = self.blossom(self.d, self.u, i_hot)
        sx = d_hotx[3, 0]
        sy = d_hoty[3, 0]
        s = (sx, sy)
        self.plot(blossom_i)
        print('s(', self.u, ') =', s)
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
        return [d_hotx, d_hoty]
#        sx = d_hotx[3, 0]
#        sy = d_hoty[3, 0]
#        return (sx, sy)
        
    def alpha(self, u, i_hot, t, index):
        i_leftmost = i_hot - 2 + index + t
        u_leftmost = self.u_grid[i_leftmost]
        i_rightmost = i_hot +1 + index
        u_rightmost = self.u_grid[i_rightmost]
        return (u_rightmost - u) / (u_rightmost - u_leftmost)
    
    def plot(self, blossom_i):
        import matplotlib.pyplot as plt
        x_val = [x[0] for x in self.coord]
        y_val = [x[1] for x in self.coord]
        plt.plot(x_val,y_val,'or')
        u = linspace(0.0000001, 0.999999, 1000)
        s = zeros((1000, 2))
        blossom_pointsx = zeros(4)
        blossom_pointsy = zeros(4)
        for j in range(0, 1000):
            i_hot = self.findhot(u[j])
            [d_hotx, d_hoty] = self.blossom(self.d, u[j], i_hot)
            sx = d_hotx[3, 0]
            sy = d_hoty[3, 0]
            s[j, :] = (sx, sy)
        plt.plot(s[:,0], s[:,1])
        self.plotblossom(blossom_i)
        plt.show()
        
    def plotblossom(self, blossom_i): # Add-on 1
        i_hot = self.findhot(self.u_grid[blossom_i])
        u = linspace(self.u_grid[i_hot]*0.95, self.u_grid[i_hot+1]*1.05, 100)
        d1 = zeros((100, 2))
        d2 = zeros((100, 2))
        d3 = zeros((100, 2))
        for j in range(0, 100):
            [d_hotx, d_hoty] = self.blossom(self.d, u[j], i_hot)
            d1x = d_hotx[1, 0]
            d1y = d_hoty[1, 0]
            d1[j, :] = (d1x, d1y)
            d2x = d_hotx[2, 0]
            d2y = d_hoty[2, 0]
            d2[j, :] = (d2x, d2y)
            d3x = d_hotx[3, 0]
            d3y = d_hoty[3, 0]
            d3[j, :] = (d3x, d3y)
        plt.plot(d1[:,0], d1[:,1])
        plt.plot(d2[:,0], d2[:,1])
        plt.plot(d3[:,0], d3[:,1])
        
#        def ControlPoints(self):
#        """
#        Here we create a matrix with all the control points, d_i
#        """
#        N_matrix = zeros((self.L+1, self.L+1))
#        print(shape(N_matrix))
#        for i in range(self.L+1):
#            for j in range(self.L+1):
#                N_matrix[i,j] = self.Basis(self.xi[i],3,j)
#                
#        " TODO: Implement scipy.linalg.solve_banded() correctly in order to solve Nd = sx and Nd = sy "
#        " Requires creation of ab-matrix: ab[1 + i - j, j] == a[i,j]"
#                
#                
#        return
#            
#    
#    def basis(self,k=3,j):
#        """
#        Runs the recursive algorithm for the basis functions
#        """
#        u_grid = self.u_grid
#        
#        def N(self, u):        
#            j = self.findhot(u)
#            
#            if k == 0:
#                if u[j-1] == u[j]:
#                    return 0
#                elif u >= u[j-1] and u > u[j]:
#                    return 1
#                else:
#                    return 0
#                
#            if u_grid[j+k-1] == u_grid[j-1]:
#                a = 0
#            else:
#                a = (u-u_grid[j-1])/(u_grid[j+k-1]-u_grid[j-1])
#            if u_grid[j+k] == u_grid[j]:
#                b = 0
#            else:
#                b = (u_grid[j+k]-u)/(u_grid[j+k]-u_grid[j])  
#  
#                return a*basis(k-1,j) + b*basis(k-1,j+1)
#  
#        return N
#    
#    
#class  TestBasis(unittest.TestCase):
#    def setUp(self):
#        self.spline = bspline(d, u_grid)
#    def TestPositive(self):
#        for j in range(len(self.spline.d)):
#            N = self.spline.basis(j)
#            for i in self.spline.u_grid:
#                self.assertFalse( N(i) < 0 )
#    

d = [(-12.73564, 9.03455),
(-26.77725, 15.89208),
(-42.12487, 20.57261),
(-15.34799, 4.57169),
(-31.72987, 6.85753),
(-49.14568, 6.85754),
(-38.09753, -1e-05),
(-67.92234, -11.10268),
(-89.47453, -33.30804),
(-21.44344, -22.31416),
(-32.16513, -53.33632),
(-32.16511, -93.06657),
(-2e-05, -39.83887),
(10.72167, -70.86103),
(32.16511, -93.06658),
(21.55219, -22.31397),
(51.377, -33.47106),
(89.47453, -33.47131),
(15.89191, 0.00025),
(30.9676, 1.95954),
(45.22709, 5.87789),
(14.36797, 3.91883),
(27.59321, 9.68786),
(39.67575, 17.30712)]
u_grid = linspace(0, 1, 26)
u_grid[ 1] = u_grid[ 2] = u_grid[ 0]
u_grid[-3] = u_grid[-2] = u_grid[-1]

b = bspline(d, u_grid)
s = b(0.2, d, 15)


# Test that blossom_i < len(d)
