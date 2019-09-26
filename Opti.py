# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:07:02 2019

@author: mans_
"""

from scipy import *
from pylab import *

#class problem(func):
#    
#    return x0

class Optimization:
    
    def __init__(self, func):
        
        self.func = func

    def newton(self,x0):

        tol =  10**(-7)
        xold=x0
        for i in range(0,1000):
            delta = solve(self.Hessian(xold),self.grad(xold))
            xnew = xold + delta
            if abs(xold-xnew) < tol:
                return xold
            xold = xnew
        return none
         
    def grad(self,x):
        h = 10**(-8)
        
        return (self.func(x+h) - self.func(x))/h
    
    def Hessian(self,x):
        eps = 10**-8
        Hessian = zeros((len(x),len(x)))
        fun1 = self.grad(x)
        for j in range(len(x)):
            xx0 = x[j]
            x[j] = xx0 + eps
            fun2 = self.grad(x)
            hessian[:,j] = (fun2-fun1)/eps
            x[j] = xx0
        return (hessian + hessian.T)/2
          
def f(x1,x2):
    return 100*(x2-x1)**(2)+(1-x1)**2
O = Optimization(f)          
O.newton((1,1))