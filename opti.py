# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:07:02 2019
@author: mans_
"""

from scipy import *
from pylab import *
from scipy import optimize
import matplotlib.pyplot as plt

class Optimization:
    
    def __init__(self, func):
        self.func = func
        
    def __call__(self):
        pass

    def newton(self,x0, method='exact', newton = 'classical'):

        tol =  10**(-7)
        xold=x0
        for i in range(0,1000):
            g = self.grad(xold, self.func) 
            if newton == 'classical':
                G = self.hessian(xold)
                s = solve(G,g)
            elif newton == 'badbroyden':
                xolder=[0.95*x for x in x0]
                try:
                    Qold = identity(len(x0))
                except TypeError:
                    Qold = 1
                Q = self.broyden(xold, xolder, Qold, 'bad')
                s = solve(Q,g)
                xolder = xold
            elif newton == 'goodbroyden':
                xolder=[0.95*x for x in x0]
                try:
                    Hold = identity(len(x0))
                except TypeError:
                    Hold = 1
                H = self.broyden(xold, xolder, Hold, 'good')
                s = H@g
                xolder = xold
            
            fa = lambda a: self.func(xold - a*s)
            alpha = self.linesearch(fa, method, xold)
            
            xnew = xold-alpha*s
            
            if norm(xold-xnew) < tol:
                return xold
            xold = xnew
        return None
         
    def grad(self,x, func):
        h = 10**(-8)
        
        
        try:
            f = zeros(len(x))
            for i in range(len(x)):
                ei = zeros(len(x))
                ei[i] = h
                f[i] = (func(x+ei) - func(x))/h
            return f
        except TypeError:
            return (func(x+h) - func(x))/h
        
        # This is in case x isn't a list
#        if type(x) == int or type(x) == float:
#            return (func(x+h) - func(x))/h
#        
#        f = zeros(len(x))
#        for i in range(len(x)):
#            ei = zeros(len(x))
#            ei[i] = h
#            f[i] = (func(x+ei) - func(x))/h
#        return f
    
    def hessian(self,x):
        hessian = zeros((len(x),len(x)))
        eps = 10**-8
        func = self.func
        
        for i in range(len(x)):     # 0, 1
            ei = zeros(len(x))
            ei[i] = eps
            
            for j in range(len(x)): # 0, 1
                ej = zeros(len(x))
                ej[j] = eps
                hessian[i,j] = (func(x+ei+ej) - func(x+ei) - func(x+ej) + func(x))/eps**2
        
        return hessian
    
    def broyden(self, xold, xolder, Qold, quality = 'good'):
        delta = array(xold) - array(xolder)
        gamma = self.grad(xold, self.func) - self.grad(xolder, self.func)
        if quality == 'bad':
            v = (gamma - Qold*delta)/(delta.T*delta)
            w = delta
            Q = Qold + v*w.T
            return (Q + Q.T)/2
        elif quality == 'good':
            Hold = Qold
            u = delta - Hold*gamma
            a = 1/(u.T*gamma)
            H = Hold + a*u*u.T
            return (H + H.T)/2
            
            
    
    def condition(self, method, fa, a0, aU, aL):
        """
        This is a method to calculate the left and right conditions. 
        
        --------------
        Imput:
            
        
        -------------
        Return:
            [LC, RC] = [Boolean, Boolean]
        """
        ra = 0.25
        sigma = 0.5 # Needs to be greater than ra
        
        dfaL = self.grad(aL,fa) # This is f_a prime of aL
        if method=='g':
            LC = (fa(a0) >= fa(aL) + (1-ra)*(a0-aL)*dfaL)
            RC = (fa(a0) <= fa(aL) + ra*(a0-aL)*dfaL)
        elif method=='w':
            LC = (self.grad(a0,fa) >= sigma*dfaL)
            RC = (fa(a0) <= fa(aL) + ra*(a0-aL)*dfaL)
        return [LC, RC]
    
    
    def linesearch(self, fa, method, xold):
        if method=='exact':
            a0 = optimize.fmin(fa,xold)
            return a0

        a0 = 1.0
        aL = 0.
        aU = 10**9
        
        [LC, RC] = self.condition(method, fa, a0, aU, aL)
        
        while not (LC and RC):
            if  not LC:
                [a0, aU, aL] = self.block1(a0, aU, aL, fa)
            else:
                [a0, aU, aL] = self.block2(a0, aU, aL, fa)
            [LC, RC] = self.condition(method, fa, a0, aU, aL)
        return a0
        
    
    def block1(self, a0, aU, aL, fa):
        tau = 0.1 # These should probably be changed
        xi = 0.2
        
        delta_a0 = self.extrapol(a0, aU, aL, fa)
        delta_a0 = max(delta_a0, tau*(a0-aL))
        delta_a0 = min(delta_a0, xi*(a0-aL))
        aL = a0
        a0 = a0 + delta_a0
        return [a0, aU, aL]
    
    def block2(self, a0, aU, aL, fa):
        tau = 0.1 # This should probably be changed
        
        aU = min(a0,aU)
        bar_a0 = self.interpol(a0, aU, aL, fa)
        bar_a0 = max(bar_a0, aL + tau*(aU-aL))
        bar_a0 = min(bar_a0, aU - tau*(aU-aL))
        a0 = bar_a0
        return [a0, aU, aL]
    
    def extrapol(self, a0, aU, aL, fa):
        return (a0-aL)*self.grad(a0,fa)/(self.grad(aL,fa) - self.grad(a0,fa))
    
    def interpol(self, a0, aU, aL, fa):
        return (a0-aL)**2*self.grad(aL,fa)/(2*(fa(aL) - fa(a0) + (a0-aL)*self.grad(aL,fa)))
        
    def plot(self):
        x1 = linspace(-0.5,2,1000)
        x2 = linspace(-1.5,4,1000)
        X1, X2 = meshgrid(x1, x2)
        plt.figure()
        #print(self.func((X1,X2)))
        cp = plt.contour(X1,X2,self.func((X1,X2)),[0,1,3,10,50,100,500,800],colors='black')
        plt.clabel(cp, inline=True, fontsize=10)
        plt.plot(self.xx[0::2],self.xx[1::2],'o',color='black')
        plt.show()
        pass      
    
    
def f(x):
    return 100*(x[1]-x[0]**2)**(2) + (1-x[0])**2
O = Optimization(f) 
#H_matrix = O.Hessian(array([0.4,0.6]))         
x = O.newton([0.8,0.8], method = 'g', newton = 'goodbroyden')
