# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:34:27 2019
@author: selle
"""
from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod
from room import LeftSideRoom, RightSideRoom, MiddleRoom
from scipy.linalg import toeplitz
from numpy.linalg import solve

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocessors = comm.Get_size()
#print('helloworldfromprocess' , rank)

# Initialize the three rooms
LeftSideRoom = LeftSideRoom(height = 1, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)
RightSideRoom = RightSideRoom(height = 1, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)
MiddleRoom = MiddleRoom(height = 2, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)

class Solver:
    
    def __init__(self):
        pass
    
    def __call__(self, room, leftBoundaryCond = None, rightBoundaryCond = None):
        K = self.Kmatrix(room)
        F = self.Fvector(room, leftBoundaryCond, rightBoundaryCond)
        newroom, leftBC, rightBC = self.timeStep(room, K, F, leftBoundaryCond, rightBoundaryCond)
        return newroom, leftBC, rightBC
        
    def Fvector(self, room, leftBoundaryCond, rightBoundaryCond):
        roomType = type(room).__name__
        
        if roomType == 'LeftSideRoom': #Normal uppe och nere, H till vänster, leftBC till höger
            F = zeros((room.meshW-1)*(room.meshH-2)) 
            for i in range(room.meshW-1):
                F[i] = room.gammaNormal
                F[-i-1] = room.gammaNormal
            for i in range(room.meshH-2):
                F[i*(room.meshW-1)] += room.gammaH #Vänster sida
                F[i*(room.meshW-1) + room.meshW-2] += leftBoundaryCond[i] #Höger sida
        
        elif roomType == 'RightSideRoom': #Normal uppe och nere, H till höger, rightBC till vänster
            F = zeros((room.meshW-1)*(room.meshH-2))            
            for i in range(room.meshW-1):
                F[i] = room.gammaNormal
                F[-i-1] = room.gammaNormal
            for i in range(room.meshH-2):
                F[i*(room.meshW-1)] += rightBoundaryCond[i]  #Vänster sida
                F[i*(room.meshW-1) + room.meshW-2] += room.gammaH #Höger sida

        elif roomType == 'MiddleRoom':
            F = zeros((room.meshW-2)*(room.meshH-2))
            gammaNormal = room.gammaNormal*ones(int((room.meshH-1)/2))
            leftSide = append(gammaNormal, leftBoundaryCond)
            rightSide = append(rightBoundaryCond, gammaNormal)
            for i in range(room.meshW-2):
                F[i] = room.gammaH
                F[-i-1] = room.gammaWF
            for i in range(room.meshH-2):
                F[i*(room.meshW-2)] += leftSide[i]
                F[i*(room.meshW-2) + room.meshW-3] += rightSide[i]  
            F = -F
        else:
            print('Room type unknown.')
        return F
    
    def Kmatrix(self, room):
        roomType = type(room).__name__
        if roomType == 'LeftSideRoom':
            a = zeros((room.meshW-1)*(room.meshH-2))
            a[0] = -4
            a[1] = 1
            a[room.meshW-1] = 1
            K = toeplitz(a)
            for i in range(room.meshH-2): 
                K[i*(room.meshW-1)+room.meshW-2, i*(room.meshW-1)+room.meshW-3] = 1 # Redandant for Maria method but crucial in case of Wiki method (if so, change to 0)
                K[i*(room.meshW-1)+room.meshW-2, i*(room.meshW-1)+room.meshW-2] = -3 # In case of Wiki method, replace -3 with -2
            for i in range(room.meshH-3): 
                K[i*(room.meshW-1)+room.meshW-2, i*(room.meshW-1)+room.meshW-1] = 0
                K[i*(room.meshW-1)+room.meshW-1, i*(room.meshW-1)+room.meshW-2] = 0
        elif roomType == 'RightSideRoom':
            a = zeros((room.meshW-1)*(room.meshH-2))
            a[0] = -4
            a[1] = 1
            a[room.meshW-1] = 1
            K = toeplitz(a)
            for i in range(room.meshH-2):
                K[i*(room.meshW-1), i*(room.meshW-1)+1] = 1 # Redandant for Maria method but crucial in case of Wiki method (if so, change to 0)
                K[i*(room.meshW-1), i*(room.meshW-1)] = -3 # In case of Wiki method, replace -3 with -2
            for i in range(room.meshH-3):
                K[i*(room.meshW-1)+room.meshW-2, i*(room.meshW-1)+room.meshW-1] = 0
                K[i*(room.meshW-1)+room.meshW-1, i*(room.meshW-1)+room.meshW-2] = 0
        elif roomType == 'MiddleRoom':
            a = zeros((room.meshW-2)*(room.meshH-2))
            a[0] = -4
            a[1] = 1
            a[room.meshW-2] = 1
            K = toeplitz(a)
            for i in range(room.meshH-3):
                K[i*(room.meshW-2)+room.meshW-3, i*(room.meshW-2)+room.meshW-2] = 0
                K[i*(room.meshW-2)+room.meshW-2, i*(room.meshW-2)+room.meshW-3] = 0
        else:
            print('Room type unknown.')
        return K

    def timeStep(self, room, K, F, leftBoundaryCond, rightBoundaryCond, omega = 0.8):
        roomType = type(room).__name__
        
        u = solve(K, F)
        
        if roomType == 'LeftSideRoom':
            u = reshape(u, (room.meshH-2, room.meshW-1)) 
            u_relax = omega*u + (1-omega)*room.mesh[1:-1,1:] #Relaxation
            
            room.UpdateMesh(u_relax)
            
            leftBoundaryCond = u_relax[:,-1]
            rightBoundaryCond = None
            
        if roomType == 'RightSideRoom':
            u = reshape(u, (room.meshH-2, room.meshW-1)) 
            u_relax = omega*u + (1-omega)*room.mesh[1:-1,:-1] #Relaxation
            
            room.UpdateMesh(u_relax)
                    
            leftBoundaryCond = None
            rightBoundaryCond = u_relax[:,0]
            
        if roomType == 'MiddleRoom':
            u = reshape(u, (room.meshH-2, room.meshW-2))
            u_relax = omega*u + (1-omega)*room.mesh[1:-1,1:-1] #Relaxation
            leftBC_relax = omega*leftBoundaryCond + (1-omega)*room.mesh[-len(leftBoundaryCond):,0]
            rightBC_relax = omega*rightBoundaryCond + (1-omega)*room.mesh[:len(rightBoundaryCond),-1]
        
            room.UpdateMesh(u_relax, leftBC_relax, rightBC_relax)         
            
            leftBoundaryCond -= u_relax[-len(leftBoundaryCond):,0] #Neumann condition
            rightBoundaryCond -= u_relax[0:len(rightBoundaryCond),-1] #Neumann condition
           
        return room, leftBoundaryCond, rightBoundaryCond     

testSolver = Solver()

gamma1Middle = 22*ones(int((MiddleRoom.meshH-3)/2))
gamma2Middle = 23*ones(int((MiddleRoom.meshH-3)/2))
newMiddleRoom = testSolver(MiddleRoom, leftBoundaryCond = gamma1Middle, rightBoundaryCond = gamma2Middle)

gammaLeft = -1*ones(int(LeftSideRoom.meshH-2))
newLeftSideRoom = testSolver(LeftSideRoom, leftBoundaryCond = gammaLeft)

gammaRight = -1*ones(int(RightSideRoom.meshH-2))
newRightSideRoom = testSolver(RightSideRoom, rightBoundaryCond = gammaRight)
