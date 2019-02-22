#!/usr/bin/env python2.7
from __future__ import division
from control.matlab import dare
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
from numpy import linalg as LA
import numpy as np
from time import gmtime, strftime
from scipy.linalg import block_diag
import time, sys, signal, atexit
import math
import pdb
import scipy.io as sio
from numpy.linalg import multi_dot
import zmq
import rospy
import os
from vicon_bridge.msg import Markers
from geometry_msgs.msg import Point


moment=strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
matfile1 = 'states'+moment+'.mat' # to write states to a matlab file
matfile2 = 'LearntR'+moment+'.mat'  # to write learnt R as it evolves to a matrix
matfile3 = 'AppliedControl'+moment+'.mat' # to write control inputs actually applied to the robots
matfile4 = 'Outputs'+moment+'.mat' # to write outputs to a matlab file

messageProcessingIndex = 0
trueIndex = 0
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)    
socket.bind("tcp://192.168.1.4:%s" % port) #needs to be the same as current IP
dt = 1/70 # sampling rate - chosen to be 30 hz
pwmTovelConversion = 0.01676357142 # linear relationship between PWM and SS vel attained
pwm2VelFactor = pwmTovelConversion

# States are mid point of the robots and x2 - x1, the lateral position difference between both robots
# System Matrices
#B  = (pwm2VelFactor * dt/2)* np.array([[1.0,1.0],[-2.0,2.0]]) # considering x2 - x1 for d state
#A  =  np.eye(2)

#A = np.array([[0.8597,0.0],[0.0,0.8597]])
#B = np.array([[0.0116, 0.0],[0.0, 0.0116]])

A = np.array([[0.8804,0.0],[0.0,0.8804]])
B = np.array([[0.009884, 0.0],[0.0, 0.009884]])
C = np.array([[0.1001,  0.1001],[ -0.2003, 0.2003]])
D = np.array([[0.0,0.0],[0.0,0.0]])
Q = 100*dot(C.T,C)

Cinv = inv(C) # will be used in extracting X from Y, the observations


Learnt_Rstorage = [np.array([[1.0,1.0]])] # variable to store learnt R matrices
X = np.array([[0.0],[0.0]])
#Y = dot(C,X)
u1_observed = 0
u2_observed = 0

X_iter = [] # to store states iteratively as a list
Y_iter = [] # to store states iteratively as a list
U_applied = [] # to store the sequence of U's applied to the system as a list

#Q = 300 * np.array([[4,0.3,1],[0.3,1,0.5],[1,0.5,0.7]]) # a random positive definite matrix
#Q = 300 * np.array([[1,0,0],[0,1,0],[0,0,2]]) # a random positive definite matrix

#T1 = np.array([[1.0,0,0,0],[0,1,0,0]])
#T2 = np.array([[0,0,1,0],[0,0,0,1]])

traMat = np.eye(2)
Occlusioncounter = 0 # to count the times when occlusion happened because of MOCAP system


def findTransfromationMatrix(robot1MocapCoordinates, robot2MocapCoordinates):
    # multiply all incoming position coordinates with the transformaton matrix
    global traMat
    global messageProcessingIndex

    r1M = robot1MocapCoordinates # assume r1M to be a 2x1 vector 
    r2M = robot2MocapCoordinates

    try:

        tempMat = inv(np.array([[r1M[0,0], r1M[1,0]],                  
                            [r2M[0,0], r2M[1,0]]]))
        dInt = LA.norm(r1M - r2M) # distance between two points
        ab1 = dot(tempMat, np.array([[-60],[-60]]))
        ab2 = dot(tempMat, np.array([[dInt/2],[-dInt/2]]))   
        transfromationMatrix = np.array([[ab1[0,0], ab1[1,0]],                  
                            [ab2[0,0], ab2[1,0]]])

    except IndexError as ie:
        print ('transformation matrix cant be found since input data is empty')
        messageProcessingIndex = 0
        raise ie

    print('transfromationMatrix:')
    print(transfromationMatrix)
    return transfromationMatrix

def findTransfromationMatrixdiff(robot1MocapCoordinates, robot2MocapCoordinates):
    # multiply all incoming position coordinates with the transformaton matrix
    global traMat
    global messageProcessingIndex

    r1M = robot1MocapCoordinates # assume r1M to be a 2x1 vector 
    r2M = robot2MocapCoordinates

    try:

        tempMat = inv(np.array([[r1M[0,0], r1M[1,0]],                  
                            [r2M[0,0], r2M[1,0]]]))
#        dInt = LA.norm(r1M - r2M) # distance between two points
        dInt = abs(r1M[0,0] - r2M[0,0]) # distance between two points
        ab1 = dot(tempMat, np.array([[-65],[-55]]))
        ab2 = dot(tempMat, np.array([[dInt/2],[-dInt/2]]))   
        transfromationMatrix = np.array([[ab1[0,0], ab1[1,0]],                  
                            [ab2[0,0], ab2[1,0]]])

    except IndexError as ie:
        print ('transformation matrix cant be found since input data is empty')
        messageProcessingIndex = 0
        raise ie

    print('transfromationMatrix:')
    print(transfromationMatrix)
    return transfromationMatrix    

def agentlearning(A,B,Q,R_est,u_obs,X,theta_max,theta_min,ForAgent):

    # for now considering only one input at each agent
    # this function provides an update for learnt theta's for each agent
    # this is the crux of learning algorithm
    # ForAgent - for whom learning is taking place
    # estimation for u1 is done at agent 1 as well, it doesn't matter where the
    # learning is being carried out.
    # running the learning loop 10 times per one iteration of control loop
    # u_obs - observed value of (ForAgent) control by (atAgent). Observation will be same everywhere

    N_iter = 10 # It starts from 0. (N_iter+1) times learning loop runs per a run of control loop

    Est_R = np.copy(R_est, order='k')  # Est_R is the R matrix used for estimation in the function
    [Sinf_thetaEst,L_thetaEst,G_thetaEst] = dare(A,B,Q,Est_R,S=None, E=None)
    U_est = - dot(G_thetaEst , X)
    u_est = U_est[ForAgent,0]
    
    if (abs(u_est - u_obs) < 1e-5): # set precision for estimation
        return Est_R    


    Est_R_MAX = np.copy(R_est, order='k')
    Est_R_MAX[ForAgent,ForAgent] = theta_max # agent indexing should start from 0
    [Sinf_thetaMax,L_thetaMax,G_thetaMax] = dare(A,B,Q,Est_R_MAX,S=None, E=None)    

    Est_R_MIN = np.copy(R_est, order='k')
    Est_R_MIN[ForAgent,ForAgent] = theta_min
    [Sinf_thetaMin,L_thetaMin,G_thetaMin] = dare(A,B,Q,Est_R_MIN,S=None, E=None)    
    
    # finding exteremes and intial slope


    
    U_max = - dot(G_thetaMax , X)
    u_max = U_max[ForAgent,0]
    U_min = - dot(G_thetaMin , X)
    u_min = U_min[ForAgent,0]

    mulfac = 1 # multiplication factor, takes values 1 or -1 dep on some rules
    theta_est = R_est[ForAgent,ForAgent]



    slope = (u_max - u_min)/(theta_max - theta_min)
    endpoint1 = np.array([[theta_est],[u_est]])

    if (slope > 0) and (u_obs < u_est):
        mulfac = -1

    if (slope < 0) and (u_obs > u_est):
        mulfac = -1

    if (mulfac == 1):
        endpoint2 = np.array([[theta_max],[u_max]])

    if (mulfac == -1):
        endpoint2 = np.array([[theta_min],[u_min]])

    for i in range(0, N_iter):
        Invslope =  (endpoint2[0,0] - endpoint1[0,0])/(endpoint2[1,0] - endpoint1[1,0])
        theta_est_new = endpoint1[0,0] +  Invslope * (u_obs - endpoint1[1,0])
        Est_R[ForAgent,ForAgent] = theta_est_new
        [Sinf_thetaEst,L_thetaEst,G_thetaEst] = dare(A,B,Q,Est_R,S=None, E=None)
        U_est_new = - dot(G_thetaEst , X)
        u_est_new = U_est_new[ForAgent,0]

        if (((u_est_new) - (u_obs)) * ((u_obs) - (endpoint2[1,0]))) > 0 :
            endpoint1 = np.array([[theta_est_new],[u_est_new]])
        else:
            endpoint2 = np.array([[theta_est_new], [u_est_new]])

        theta_est = theta_est_new   # not being used anywhere

    return Est_R


def callback(Data):
    global messageProcessingIndex    
    global Occlusioncounter    
    global X_iter
    global Y_iter
    global Learnt_Rstorage
    global U_applied
    global traMat
    global matfile1
    global matfile2
    global matfile3
    global X
    global Y
    global u1_observed
    global u2_observed
    global trueIndex
    occlusionflag = 0

    theta_max = 100
    theta_min = 0.001;
    theta1 = 0.6
    theta2 = 0.3
    R1_control = block_diag(theta1, 1) # for the control matrix at agent 1
    R2_control = block_diag(1, theta2) # for the control matrix at agent 2    

    #todo ensure that marker 0 is always one robot and marker 1 is the other one

    try:
        # robot1Pose = np.array([[Data.markers[0].translation.x],[Data.markers[0].translation.y]])
        # robot2Pose = np.array([[Data.markers[1].translation.x],[Data.markers[1].translation.y]])
        robot1Pose = np.array([[Data.markers[0].translation.x]\
#            ,[Data.markers[0].translation.y]])
            ,[Data.markers[0].translation.y + 1000]])  # for change in origin
        robot2Pose = np.array([[Data.markers[1].translation.x]\
#            ,[Data.markers[1].translation.y]])
            ,[Data.markers[1].translation.y + 1000]])        



        robot1Pose = robot1Pose*0.001 #to convert mocap coordinates to meters
        robot2Pose = robot2Pose*0.001
        messageProcessingIndex = messageProcessingIndex + 1
#        trueIndex = messageProcessingIndex - 1 # the index of the message (sample) being processed                

    
    except IndexError:
        print('Occlusion happened. Ensure markers are in the  field of view of VICON')
        occlusionflag = 1
        Occlusioncounter = Occlusioncounter + 1            



    try:
        if messageProcessingIndex == 1:
            # find the transformation matrix
            traMat = findTransfromationMatrix(robot1Pose, robot2Pose)

        if messageProcessingIndex > 1:  # perform learning as soon as you observe
            R_hat = block_diag(Learnt_Rstorage[(trueIndex -1)][0,0],Learnt_Rstorage[(trueIndex -1)][0,1])
            # Theta 1 learning, agents start at 0. So agent 0
            Rhat_theta1_update = agentlearning(A,B,Q,R_hat,u1_observed,X,theta_max,theta_min, 0)
            # Theta 2 learning, agents start at 0. So agent 1
            Rhat_theta2_update = agentlearning(A,B,Q,R_hat,u2_observed,X,theta_max,theta_min, 1)
    #            Learnt_Rstorage = np.concatenate(
    #                (Learnt_Rstorage,np.array([[Rhat_theta1_update[0,0],Rhat_theta2_update[1,1]]]) ))
            Learnt_Rstorage.append(np.array([[Rhat_theta1_update[0,0],Rhat_theta2_update[1,1]]]))
            #X_iter = np.concatenate((X_iter, X))
            X_iter.append(X.T)
            #Y_iter = np.concatenate((Y_iter, Y))
            Y_iter.append(Y.T)
            # form correct control matrices
            R1_control[1,1] = Rhat_theta2_update[1,1]
            R2_control[0,0] = Rhat_theta1_update[0,0]

        if occlusionflag == 1 and messageProcessingIndex > 1:
            X = dot(A,X) + dot(B,np.array([[u1_observed],[u2_observed]])) # using predicted value for the state
            Y = dot(C,X)
            
        if occlusionflag == 0:
            ro1ExpPos = dot(traMat, robot1Pose)  # robot one pose in Experiment coordinate system
            ro2ExpPos = dot(traMat, robot2Pose)  # robot two pose in Experiment coordinate system
            Y = np.array([[0.5*(ro1ExpPos[0,0] + ro2ExpPos[0,0])],
                          [(ro2ExpPos[0,0] - ro1ExpPos[0,0])]])
            X = dot(Cinv, Y)


        if messageProcessingIndex > 0:
            [Sinf_ag1,L_ag1,G_ag1] = dare(A,B,Q,R1_control,S=None, E=None)
            [Sinf_ag2,L_ag2,G_ag2] = dare(A,B,Q,R2_control,S=None, E=None)
            U1_observed = - dot(G_ag1 , X) # 2 x 1 vector
            U2_observed = - dot(G_ag2 ,X) # 2 x 1 vector
            u1_observed = U1_observed[0,0] # control to be applied at agent 1
            u2_observed = U2_observed[1,0] # control to be applied at agent 2
            socket.send_string("%4.3f %4.3f" % (u1_observed, u2_observed)) # sending control to robots
            U_applied.append(np.array([[u1_observed, u2_observed]]))
            trueIndex = trueIndex + 1

#      adding disturbance to robots        
#        if trueIndex == 400:   
        if (1000 <= messageProcessingIndex <= 1020):
            print ('applying disturbance')
            socket.send_string("%4.3f %4.3f" % (-30, 0)) # sending control to robots


    except KeyboardInterrupt:
        print ('Interrupted due to abnormal robot behaviour or intentionally')
        sio.savemat(matfile1, mdict={'states' : X_iter}, oned_as = 'row')
        sio.savemat(matfile4, mdict={'Outputs' : Y_iter}, oned_as = 'row')        
        sio.savemat(matfile2, mdict={'LearntR' : Learnt_Rstorage}, oned_as = 'row')
        sio.savemat(matfile3, mdict={'AppliedControl' : U_applied}, oned_as = 'row')
        for dummy in range(50):
            socket.send_string("%4.3f %4.3f " % (0, 0))        
            socket.send_string("0.0 0.0")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def main():   

    time.sleep(90) # to start camera and motion capture system for recording the trial

    try:
        rospy.init_node('subscriber', anonymous =True)
        sub = rospy.Subscriber('/vicon/markers',Markers,callback,queue_size=1)
        rospy.spin()
        print ('Interrupted due to abnormal robot behaviour')
        for dummy in range(50):
            socket.send_string("%4.3f %4.3f " % (0.0, 0.0))
            socket.send_string("0.0 0.0")

        print ('Occlusioncounter:')
        print (Occlusioncounter)
        sio.savemat(matfile1, mdict={'states' : X_iter}, oned_as = 'row')
        sio.savemat(matfile2, mdict={'LearntR' : Learnt_Rstorage}, oned_as = 'row')
        sio.savemat(matfile3, mdict={'AppliedControl' : U_applied}, oned_as = 'row')
        sio.savemat(matfile4, mdict={'Outputs' : Y_iter}, oned_as = 'row')


        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    except rospy.ROSInterruptException:
        print('scrwed')
        pass
                        
if __name__ == '__main__':
    main()
