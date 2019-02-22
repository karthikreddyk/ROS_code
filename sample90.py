#!/usr/bin/env python2.7
from __future__ import division
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
from numpy import linalg as LA
import numpy as np
from time import gmtime, strftime
from scipy.linalg import block_diag
import time, sys, signal, atexit
import math
import pdb
import scipy.io
from numpy.linalg import multi_dot
import zmq
import sys
import time
import rospy
import os
from vicon_bridge.msg import Markers
from geometry_msgs.msg import Point

moment=strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
matfile1 = 'states'+moment+'.mat' # to write states to a matlab file
matfile2 = 'LearntR'+moment+'.mat'  # to write learnt R as 
                                    #it evolves to a matrix
matfile3 = 'AppliedControl'+moment+'.mat' # to write control inputs actually
                                          # applied to the robots
matfile4 = 'Outputs'+moment+'.mat' # to write outputs to a matlab file



messageProcessingIndex = 0
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)    
socket.bind("tcp://192.168.1.4:%s" % port)

A = np.array([[0.8804,0.0],[0.0,0.8804]])
B = np.array([[0.009884, 0.0],[0.0, 0.009884]])
B1 = np.array([[0.0116],[0.0]])
B2 = np.array([[0.0],[0.0116]])

C = np.array([[0.1001,  0.1001],[ -0.2003, 0.2003]])
D = np.array([[0.0,0.0],[0.0,0.0]])
P = 300*dot(C.T,C)
Cinv = inv(C) # will be used in extracting X from Y, the observations
T1 = np.array([[1.0,0.0]])
T2 =  np.array([[0.0,1.0]])
X_iter = [] # to store states iteratively
U_iter = [] # to store control input, in this case velocity
U_hat_iter = [] # to store virtual nash equilibrium values
u1_hat_iter = [] # to store virtual nash equilibrium values
u2_hat_iter = [] # to store virtual nash equilibrium values
# Parameters
theta1 = .2
theta2 = .4

theta=np.array([[theta1],[theta2]])
theta1_hat = 1
theta2_hat = 1

R = block_diag(theta1 * np.eye(1) , theta2 * np.eye(1))
R_hat = block_diag(theta1_hat * np.eye(1) , theta2_hat * np.eye(1))


G1 = dot(B.T, P)
G2 = dot(G1, B) 
G3 = dot(B1.T, P)
G4 = dot(G3, B1)
G5 = dot(B2.T, P)
G6 = dot(G5, B2)
G7 = B - dot(B1, T1)
G8 = B - dot(B2, T2)
#traMat = np.eye(1)
Occlusioncounter = 0




def findTransfromationMatrix(robot1MocapCoordinates\
    ,robot2MocapCoordinates):
# multiply all incoming position coordinates with the transformaton matrix
    global traMat
    global messageProcessingIndex    

    r1M = robot1MocapCoordinates # assume r1M to be a 2x1 vector 
    r2M = robot2MocapCoordinates

    try:

        tempMat = inv(np.array([[r1M[0,0], r1M[1,0]],                  
                            [r2M[0,0], r2M[1,0]]]))
        dInt = LA.norm(r1M - r2M)
        ab1 = dot(tempMat, np.array([[-60],[-60]]))
        ab2 = dot(tempMat, np.array([[dInt/2],[-dInt/2]]))   
        transfromationMatrix = np.array([[ab1[0,0], ab1[1,0]],                  
                            [ab2[0,0], ab2[1,0]]])

    except IndexError as ie:
        print ('transformation matrix cant be found since input data\
            is empty')
        messageProcessingIndex = 0
        raise ie

    print('transfromationMatrix:')
    print(transfromationMatrix)
    return transfromationMatrix

# def findTransfromationMatrixdiff(robot1MocapCoordinates\
#     , robot2MocapCoordinates):
#     # multiply all incoming position coordinates with the transformaton matrix
#     global traMat
#     global messageProcessingIndex

#     r1M = robot1MocapCoordinates # assume r1M to be a 2x1 vector 
#     r2M = robot2MocapCoordinates

#     try:

#         tempMat = inv(np.array([[r1M[0,0], r1M[1,0]],                  
#                             [r2M[0,0], r2M[1,0]]]))
# #        dInt = LA.norm(r1M - r2M) # distance between two points
#         dInt = abs(r1M[0,0] - r2M[0,0]) # distance between two points
#         ab1 = dot(tempMat, np.array([[-65],[-55]]))
#         ab2 = dot(tempMat, np.array([[dInt/2],[-dInt/2]]))   
#         transfromationMatrix = np.array([[ab1[0,0], ab1[1,0]],                  
#                             [ab2[0,0], ab2[1,0]]])

#     except IndexError as ie:
#         print ('transformation matrix cant be found since\
#          input data is empty')
#         messageProcessingIndex = 0
#         raise ie

#     print('transfromationMatrix:')
#     print(transfromationMatrix)
#     return transfromationMatrix    


def callback(Data):
    global messageProcessingIndex
    global Occlusioncounter    
    global X_iter
    global U_iter
    global U_hat_iter
    global R_hat
    global theta1_hat
    global theta2_hat
    global traMat
    global u1_hat_iter
    global u2_hat_iter
    global matfile1
    global matfile2
    global matfile3
    global matfile4            

    if (0 <= messageProcessingIndex <= 3) or (980 <= messageProcessingIndex <= 1040) :
        print('messageProcessingIndex:')
        print(messageProcessingIndex)
        print('theta1_hat:')
        print(theta1_hat)    
        print('theta2_hat:')
        print(theta2_hat)            

    try:
        robot1Pose = np.array([[Data.markers[0].translation.x]\
#            ,[Data.markers[0].translation.y]])
            ,[Data.markers[0].translation.y + 1000]])
        robot2Pose = np.array([[Data.markers[1].translation.x]\
#            ,[Data.markers[1].translation.y]])
            ,[Data.markers[1].translation.y + 1000]])

        robot1Pose = robot1Pose*0.001 #to convert mocap coordinates to meters
        robot2Pose = robot2Pose*0.001 #to convert mocap coordinates to meters
        messageProcessingIndex = messageProcessingIndex + 1
        if messageProcessingIndex == 1:            
            # find the transformation matrix
            traMat = findTransfromationMatrix(robot1Pose, robot2Pose)
#            traMat = findTransfromationMatrixdiff(robot1Pose, robot2Pose)
#            findTransfromationMatrixdiff
#Karthik
#        if messageProcessingIndex == 30:
#            raise KeyboardInterrupt

#        if messageProcessingIndex == 15:
#            raise IndexError

        # robot one pose in Experiment coordinate system:
        ro1ExpPos = dot(traMat, robot1Pose)

        # robot one pose in Experiment coordinate system:
        ro2ExpPos = dot(traMat, robot2Pose)

        Y = np.array([[0.5*(ro1ExpPos[0,0] + ro2ExpPos[0,0])],
                      [(ro2ExpPos[0,0] - ro1ExpPos[0,0])]])
        X = dot(Cinv, Y)        

        X_iter.append(X.T)

        f_X = dot(A,X)   # f(X(k)) defined in the paper  

        #  virtual nash equilibrium calculation
        U_hat = -1 * multi_dot([inv(R_hat + G2) , G1, f_X])
        U_hat_iter.append(U_hat.T)
        local1 = (f_X + dot(G7,U_hat))
        local2 = (f_X + dot(G8,U_hat))
        
        # calculating corresponding control inputs for agent1 & 2
        u1 = -1 * multi_dot([inv((theta1 * np.eye(1)) + G4), G3,local1])
        u1_hat_iter.append(u1)


        u2 = -1 * multi_dot([inv((theta2 * np.eye(1)) + G6), G5,local2])
    #    print('u2:')
    #    print(u2)
        u2_hat_iter.append(u2)


        socket.send_string("%4.3f %4.3f" % (u1[0,0], u2[0,0]))

        
        e1  = (-1 * multi_dot([u1.T, G3, (local1 + dot(B1,u1))]))\
         - theta1_hat * dot(u1.T, u1)
        theta1_hat = theta1_hat + e1 * pow(dot(u1.T, u1), -1)
        
        # update of paramter estimates in robot 1
        e2 = (-1 * multi_dot([u2.T, G5, (local1 + dot(B2,u2))]))\
         - theta2_hat * dot(u2.T, u2)
        theta2_hat = theta2_hat + e2 * pow(dot(u2.T, u2), -1)

        # updated parameter block diagonal matrix        
        R_hat = block_diag(theta1_hat * np.eye(1) , theta2_hat\
         * np.eye(1))
        
#        if messageProcessingIndex == 1000:
        if (1000 <= messageProcessingIndex <= 1020):
            print('Sending disturbance to robots')            
            socket.send_string("%4.3f %4.3f" % (-30, 0)) # sending control to robots                

        if  abs(Y[1,0]) > 60:
            print ('Stopping since the system has become unstable')
            raise KeyboardInterrupt            

    except IndexError:
        print('Occlusion happened at messageProcessingIndex:')
        print(messageProcessingIndex)
        Occlusioncounter = Occlusioncounter + 1
        occlusionflag = 1
    # below piece of code can be removed. It doesn't serve any purpose

    except KeyboardInterrupt:        
        print ('Interrupted due to abnormal robot behaviour')
        scipy.io.savemat(matfile1, mdict={'states' : X_iter}\
            , oned_as = 'row')
        scipy.io.savemat(matfile2, mdict={'u1hat' : u1_hat_iter}\
            , oned_as = 'row')
        scipy.io.savemat(matfile4, mdict={'u2hat' : u2_hat_iter}\
            , oned_as = 'row')         
        scipy.io.savemat(matfile3, mdict={'UhatIter' : U_hat_iter}\
            , oned_as = 'row')
        for dummy in range(50):
            socket.send_string("%4.3f %4.3f" % (0.0, 0.0))
            socket.send_string("0.0 0.0")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


#    X_iter.append(X.T)



def main():   

    time.sleep(90)

    try:
        rospy.init_node('subscriber', anonymous =True)
        sub = rospy.Subscriber('/vicon/markers',Markers,callback\
            ,queue_size=1)
        rospy.spin()
        print ('Interrupted due to abnormal robot behaviour')
        for dummy in range(50):
            socket.send_string("%4.3f %4.3f" % (0, 0))        
            socket.send_string("0.0 0.0")        
        scipy.io.savemat(matfile1, mdict={'states' : X_iter}, \
            oned_as = 'row')
        scipy.io.savemat(matfile2, mdict={'u1hat' : u1_hat_iter}\
            , oned_as = 'row')
        scipy.io.savemat(matfile4, mdict={'u2hat' : u2_hat_iter}\
            , oned_as = 'row')         
        scipy.io.savemat(matfile3, mdict={'UhatIter' : U_hat_iter}\
            , oned_as = 'row')
        print ('Occlusioncounter:')
        print (Occlusioncounter)

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    except rospy.ROSInterruptException:
        print('scrwed')
        pass
                        
if __name__ == '__main__':
    main()
