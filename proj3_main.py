#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:12:19 2019

@author: mariagunnarsson
"""


from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod
from room import LeftSideRoom, RightSideRoom, MiddleRoom
from scipy.linalg import toeplitz
from numpy.linalg import solve
from proj3 import Solver

LeftSideRoom = LeftSideRoom(height = 1, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)
RightSideRoom = RightSideRoom(height = 1, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)
MiddleRoom = MiddleRoom(height = 2, width = 1, deltaX = 1/20, gammaH = 40, gammaWF = 5, gammaNormal = 15)

leftBC = 22*ones(int((MiddleRoom.meshH-3)/2))
rightBC = 23*ones(int((MiddleRoom.meshH-3)/2))



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocessors = comm.Get_size()
    testSolver = Solver()
    print(rank)
    print(nprocessors)
    print('helloworldfromprocess' , rank)

    
    for i in range(10):
    
        if rank == 0:            
            leftBC = comm.recv(source=1)            
        
        elif rank == 1:
            MiddleRoom, leftBC, rightBC = testSolver(MiddleRoom,leftBC, rightBC)
            data_left = leftBC
            data_right = rightBC
            comm.send(data_left, dest=0)
            comm.send(data_right, dest=2)
            
        elif rank == 2:
            rightBC = comm.recv(source=1)

        
        
        
        if rank == 0:            
            LeftSideRoom, leftBC, rightBC = testSolver(LeftSideRoom, leftBoundaryCond = leftBC, rightBoundaryCond = None)
            data_left = leftBC
            comm.send(data_left, dest=1)
        
        elif rank == 1:       
            leftBC = comm.recv(source=0)
            rightBC = comm.recv(source=2)       
            
        elif rank == 2:
            RightSideRoom, leftBC, rightBC = testSolver(RightSideRoom, leftBoundaryCond = None, rightBoundaryCond = rightBC)
            data_right = rightBC
            comm.send(data_right, dest=1)

        
if __name__ == "__main__": 
    main()
        
    