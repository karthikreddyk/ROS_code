
from numpy import dot
from numpy import sum, tile, linalg
from numpy.linalg import inv
import numpy as np
#import matplotlib.pyplot as plt
#from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import time, sys, signal, atexit
from upm import pyupm_bno055 as sensorObj
#from commandRobot import commandRobot
from cmdf import commandRobot
from actuate import actuateR
import math
import pickle
#import multiprocessing 
#import os
#import pdb
import scipy.io
import mraa
from _cffi_backend import string

#R_std = 0.08  # Karthik - needs to be tuned for different values
#Q_std_x = 0.1  # Karthik - needs to be tuned for different values
#Q_std_y = 0.17  # Karthik - needs to be tuned for different values

R_std = 0.02  # Karthik - needs to be tuned for different values
Q_std_x = 0.8  # Karthik - needs to be tuned for different values
Q_std_y = 0.8  # Karthik - needs to be tuned for different valu
#xOffset =  0.077952000262216 # offset to make process noise mean zero
#yOffset =  0.009179999827221 # offset to make process noise mean zero
#zOffset =  9.800646015357971 # offset to make process noise mean zero
#offset_2d = np.array([[xOffset , yOffset]]).T



def getProcessCovarianceMatrix(dt):
    
    a_X_var = Q_std_x # variance of random part of acceleration's X component
    a_Y_var = Q_std_y # variance of random part of acceleration's X component
    w = np.zeros((4,1))
    
    w[0,0] = a_X_var * dt
    w[1,0] = 0.5 * a_X_var * pow(dt,2)
    w[2,0] = a_Y_var * dt
    w[3,0] = 0.5 * a_Y_var * pow(dt,2)
    
    Q = dot(w,w.T) 
            
    Q[0,2] = 0
    Q[0,3] = 0
    Q[1,2] = 0
    Q[1,3] = 0

    Q[2,0] = 0
    Q[3,0] = 0
    Q[2,1] = 0
    Q[3,1] = 0
    
    return(Q)

def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P)

def setupUart():
    u = mraa.Uart(0)
    # Set UART parameters
    u.setBaudRate(115200)
    u.setMode(8, mraa.UART_PARITY_NONE, 1)
    u.setFlowcontrol(False, False)
    return u    
    
def encoderData():    
    u.writeStr("<E>")
    if u.dataAvailable(5):
        return(u.readStr(20))
    else:
        return("ND")

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
#    LH = gauss_pdf(Y, IM, IS)
#    return (X,P,K,IM,IS,LH)
    return (X,P,K,IM,IS)    

def gauss_pdf(X, M, S):
    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E=E+0.5*M.shape()[0]*log(2*pi)+0.5*log(det(S))
        P=exp(-E)
    else:
        DX=X-M
        E=0.5*dot(DX.T,dot(inv(S),DX))
        E=E+0.5*M.shape()[0]*log(2*pi)+0.5*log(det(S))
        P=exp(-E)
    return(P[0],E[0])

def imuSetup():
    # Instantiate an BNO055 using default parameters (bus 0, addr
    # 0x28).  The default running mode is NDOF absolute orientation
    # mode.
    sensor = sensorObj.BNO055()
    # setting the desired accelerometer configuration
    #3=> 16g, 4=> 125 hz bw, 0 = normal power mode
    # in fusion mode config values are ignored
    sensor.setAccelerationConfig(0,3,0) # Karthik - Might need to change this configuration
    sensor.setAccelerometerUnits(False) # to set the units to m/s^2

    print("First we need to calibrate.  4 numbers will be output every")
    print("second for each sensor.  0 means uncalibrated, and 3 means")
    print("fully calibrated.")
    print("See the UPM documentation on this sensor for instructions on")
    print("what actions are required to calibrate.")
    print()

    while (not sensor.isFullyCalibrated()):
        intData = sensor.getCalibrationStatus()
#        print("Magnetometer:", intData[0], end = ' ')
#        print(" Accelerometer:", intData[1], end = ' ')
#        print(" Gyroscope:", intData[2], end = ' ')
#        print(" System:", intData[3])
        print("Magnetometer:", intData[0])
        print(" Accelerometer:", intData[1])
        print(" Gyroscope:", intData[2])
        print(" System:", intData[3])
        time.sleep(0.1)


    print("Calibration complete.")
    time.sleep(60)
    byteData = sensor.readCalibrationData()
    print("Read data successfully.")
    print("Writing calibration data...")
    print("Reading calibration data....")
    time.sleep(1)    
    return sensor

def imuSetup1():
    # reads calibration data from a file to device registers
    sensor = sensorObj.BNO055()
    # setting the desired accelerometer configuration
    #3=> 16g, 4=> 125 hz bw, 0 = normal power mode
    # in fusion mode config values are ignored
    sensor.setAccelerationConfig(0,3,0) # Karthik - Might need to change this configuration
    sensor.setAccelerometerUnits(False) # to set the units to m/s^2
    byteData = pickle.load(open('calibRelative.dump' , 'rb'))
    sensor.writeCalibrationData(byteData)
    print("IMU ready for measurements...")
    time.sleep(60)
    return sensor    
    
    

    

def getNewRaidalDistance(Y_robot,angleData,angleData_prev_iter,Y_robot_prev,enc_radial_distance_prev_iter,i):
    enc_conv_distance = 0.007290240285580 # used to convert encoder reading to distance in meters
    
    if i != 0:

        delta_Y_robot = Y_robot - Y_robot_prev  #Difference in current and past encoder readings
        delta_Y_robot = delta_Y_robot * enc_conv_distance # to get the exact distance travelled during the iteration
        delta_yaw = 180 - abs(angleData_prev_iter[0] - angleData[0]) # change in yaw angle absolute value
        a =  delta_Y_robot[0,0] # side 1 of triangle # might
        b = enc_radial_distance_prev_iter # side 2 of the traingle 
        c = math.sqrt(math.pow(a,2) + math.pow(b,2) - 2*a*b*math.cos(math.radians(delta_yaw)))
        return c
    else:
        return (Y_robot[0,0] * enc_conv_distance)
    
def getEncoderIncrement(Y_robot,Y_robot_prev,relative_heading,i):
    enc_conv_distance = 0.007290240285580 # used to convert encoder reading to distance in meters
    if i != 0:

        delta_Y_robot = Y_robot - Y_robot_prev  #Difference in current and past encoder readings
        delta_Y_robot = delta_Y_robot * enc_conv_distance # to get the exact distance travelled during the iteration
        return (np.array([[delta_Y_robot[0,0] * (math.cos(math.radians(relative_heading))), delta_Y_robot[0,0] * (math.sin(math.radians(-1 * relative_heading)))]]).T)
    else:
        return (Y_robot * enc_conv_distance)
    
def getAmatrix(dt): # to ensure A updates with changing dt
        
    A = np.array([[1, 0, 0,  0],
                  [dt,  1, 0, 0],
                  [0, 0, 1,  0],
                  [0,  0, dt,  1]])
    
    return(A)

def getBmatrix(dt): # to ensure B updates with changing dt
    
    B = np.array([[dt, 0],
                  [0.5*dt*dt, 0],
                  [0,  dt],
                  [0, 0.5*dt*dt]])
    return(B)    
    
def computeAccelDrift(imuSensor):
    driftTemp = np.array([[0,0]])
    for i in range(0, 99):
        imuSensor.update()
        accelDriftData =  imuSensor.getLinearAcceleration()[0:2]
        driftTemp = np.add (driftTemp , np.array([[accelDriftData[0],accelDriftData[1]]]))
        time.sleep(0.01)
    print('drift calculated:')
    print((driftTemp/100))
    return((driftTemp/100).T)
    
    

def main():
    # for making the robot go in a square begin
    crossed360Count = 0 # to track as and when filtered gyro o/p  for angular displacement crosses 360. This happens when the accumulated angle crosses 360.
    nthTime = -1 # delete - just to count how many times the robot went in a square
    happen1 = False
    happen2 = False
    happen3 = False
    happen4 = False
    happen5 = False
    happen6 = False
    happen7 = False
    happen8 = False
    happen9 = False
    happen10 = False
    happen11 = False
    happen12 = False
    happen13 = False
    happen14 = False
    happen15 = False
    happen16 = False    
    
    # for making the robot go in a square end
    enc_conv_distance = 0.007290240285580 # used to convert encoder reading to distance in meters
    count4successiveSameEncoders = 0
    succNochange = np.array([[0,0,0,0]]) # for checking if there's no encoder count change in 4 successive iterations
    sNcCount = 0 # loop counter for the above list
    count4successiveSamePast = 0    
    
    matfile1 = 'states.mat' # to write states to a matlab file
    matfile2 = 'covar.mat'  # to write covariance matrix to a matlab file
    matfile3 = 'acceleration.mat' # to write states to a matlab file
    matfile4 = 'relative_Heading.mat' # to write states to a matlab file
    matfile5 = 'radial_encoder_dis.mat'  # to write covariance matrix to a matlab file
    matfile6 = 'Y_UpdateInput.mat'  # to write Y 
    
    missedEncoderCount = 0 # To measure any issues coming out of 
    missedIMUCount = 0
    longerIterationCount = 0    
       

    dt = 0.01   # time step
    X_iter = [] # to store states iteratively
    P_iter = []  # to store covariance matrices iteratively
    U_iter = [] # to store acceleration input
    angle_iteration = [] # to store orientation data
    encoder_measurement = [] # to store encoder input
    Y_iter = [] #for storing Y's passed to Kalman
    Y_velocity = 0 # for velocity from encoders
    Y_pulsesPerSec = 0
    relative_heading = 0
    
    #queue = multiprocessing.Queue()

    # order is V_x, X, V_y, Y - reason mentioned in the turotial
    A = np.array([[1, 0, 0,  0],
                  [dt,  1, 0, 0],
                  [0, 0, 1,  0],
                  [0,  0, dt,  1]])
    B = np.array([[dt, 0],
                  [0.5*dt*dt, 0],
                  [0,  dt],
                  [0, 0.5*dt*dt]])

#    H = np.array([[0, 1, 0, 0],
#                  [0, 0, 0, 1]])

    H = np.eye(4)

    # Error covariance of noise
    R = np.eye(4) * R_std**2  # assumption # karthik # might # need to change
#    R[0,0] = 0.0 * 0.03**2
#    R[2,2] = 0.0 * 0.03**2
    

    # Error covariance of the process
#    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    Q = getProcessCovarianceMatrix(dt)

    # Initialization # Karthik- when robot is starting from any point other than the origin
    # Karthik - at any different pose appropriate orientation ought to be set
    X = np.array([[0, 0, 0, 0]]).T  # .T implies transpose
  #  P = np.eye(4) * 500.    
    P = np.eye(4) * 6    # 6 - initial covariance we assume
    U = np.array([[0,0]]).T
    Y = np.array([[0,0,0,0]]).T    

    imuSensor = imuSetup1()
    uart = setupUart();
    uart.writeStr("<r>")

    offset_2d = computeAccelDrift(imuSensor)
    imuSensor.update()
    angleData = imuSensor.getEulerAngles()
    print('angleData:') # removeprint
    print(angleData) # removeprint
    
    initialHeading = angleData[0]  # to remove as offset
    print('initialHeading:') # removeprint
    print(initialHeading) # removeprint    
    
#    imuSensor = imuSetup1()    # directly read off the data
 

    #encoder_trigger = 0 # used to know when to get encoder reading from arduino inside
    N_iter = 8000 # number of iterations
    Y_new = np.array([[0,0]]).T
    angleData_prev_iter = np.array([[0,0,0]]).T # to store prev iteration's angles
    enc_radial_distance_prev_iter = 0 # to store previous  radial distance from origin 
                                                                            #travelled by bot from encoder readings
    Y_robot_prev = np.array([[0,0]]).T # to store previous encoder reading returned from robot
#    os.system("python Chk1test.py '<r>'")
    # Stoting initial values in arrays - begin:
    Y_robot_prev_4_iter = np.array([[0,0]]).T
    X_iter.append(X.T)
    P_iter.append(P.T)
    U_iter.append(U.T)
    angle_iteration.append(relative_heading)
    encoder_measurement.append(Y_new)
    # Stoting initial values in arrays - end:    
    
    
    begin = time.time()
    for i in range(0, N_iter):
        
#        start = time.time()
        
        
#        if i == 0:
#            prevTime = start # intialize prevTime for the first time 
        
#        A = getAmatrix(dt)
#        B = getBmatrix(dt)
        
    
        # get input data from IMU - begin
        # Karthik - as of now I get this data (orientation) directly from IMU (Fused o.p)
        # can modify this to make use of encoder measurement
        # Karthik-might modify

    #Karthik Undo until here
#        print('current iteration:')
#        print(i)
#        print(X)
        
        # Code to move the robot  begin  
        
        if ((i == 40) or (happen16 == True)) and happen1 == False:
            uart.writeStr("<s,20,F,20,F>")
            happen1 = True
            nthTime = nthTime + 1
            print(1)
            print('iteration:')
            print(i)
            print(X)
            happen2 = False
            happen3 = False
            happen4 = False
            happen5 = False
            happen6 = False
            happen7 = False
            happen8 = False
            happen9 = False
            happen10 = False
            happen11 = False
            happen12 = False
            happen13 = False
            happen14 = False
            happen15 = False
            happen16 = False
            

            
        elif X[1,0] >= 1.5 and happen2 == False and happen1 == True:
            uart.writeStr("<S>")
            happen2 = True
            print(2)
            print('iteration:')
            print(i)
            print(X)
        elif happen2 == True and happen3 == False:
            uart.writeStr("<s,20,F,20,B>")
            happen3 = True
            print(3)            
            print('iteration:')
            print(i)
            print(X)                        
                        
        elif relative_heading >= (90 + 360 * nthTime)  and happen4 == False and happen3 == True:            
            uart.writeStr("<S>")
            happen4 = True
            print(4)
            print('iteration:')
            print(i)
            print(X)                                                
        
        elif happen5 == False and happen4 == True:
            uart.writeStr("<s,20,F,20,F>")
            happen5 = True
            print(5)
            print('iteration:')
            print(i)
            print(X)                                    

        elif abs(X[3,0]) >= 1.0 and happen6 == False and happen5 == True:
            uart.writeStr("<S>")
            happen6 = True
            print(6)
            print('iteration:')
            print(i)
            print(X)                                               

        elif happen7 == False and happen6 == True:
            uart.writeStr("<s,20,F,20,B>")
            happen7 = True
            print(7)
            print('iteration:')
            print(i)
            print(X)                                    
                        
        elif relative_heading >= (180 + 360 * nthTime) and happen8 == False and happen7 == True:
            uart.writeStr("<S>")
            happen8 = True
            print(8)
            print('iteration:')
            print(i)
            print(X)                                    

        elif happen9 == False and happen8 == True:
            uart.writeStr("<s,20,F,20,F>")
            happen9 = True
            print(9)
            print('iteration:')
            print(i)
            print(X)                                            
        
        elif (X[1,0] <= 0) and happen9 == True and happen10 == False:
            uart.writeStr("<S>")
            happen10 = True
            print(10)
            print('iteration:')
            print(i)
            print(X)
        elif happen11 == False and happen10 == True:
            uart.writeStr("<s,20,F,20,B>")
            happen11 = True
            print(11)
            print('iteration:')
            print(i)
            print(X)
        elif (relative_heading >= (270 + 360 * nthTime)) and happen12 == False and happen11 == True:
            uart.writeStr("<S>")
            happen12 = True
            print(12)
            print('iteration:')
            print(i)
            print(X)
        elif happen13 == False and happen12 == True:
            uart.writeStr("<s,20,F,20,F>")
            happen13 = True
            print(13)
            print('iteration:')
            print(i)
            print(X)
        elif (X[3,0]) >= 0 and happen14 == False and happen13 == True:
            uart.writeStr("<S>")
            happen14 = True
            print(14)
            print('iteration:')
            print(i)
            print(X)    
        elif happen15 == False and happen14 == True:
            uart.writeStr("<s,20,F,20,B>")
            happen15 = True
            print(15)
            print('iteration:')
            print(i)
            print(X)
        elif (relative_heading >= (360 + 360 * nthTime)) and happen16 == False and happen15 == True:
            uart.writeStr("<S>")
            happen16 = True
            happen1 = False
            print(16)
            print('iteration:')
            print(i)
            print(X)
                                       
                                                            
            
                              
                                    
            
            
                 
        
#        if i == 701:
#            uart.writeStr("<s,20,F,20,F>")                    
#        if i == 1000:
#            uart.writeStr("<S>")
        if i == 7900:
            uart.writeStr("<S>")
        
        # Code to move the robot  end             
                        
            
            # p = multiprocessing.Process(target=actuateR, args=(queue,))
            # p.start()
            # queue.put("<u,1000,F,1000,F>")
            
        imuSensor.update()
        angleData = imuSensor.getEulerAngles()
        
        # to check if heading angle crossedover 360 to 0
        if (abs(angleData[0] - angleData_prev_iter[0]) >= 300): 
            crossed360Count = crossed360Count + 1 # hold howmany times cross over took place
        
        accel_Data =  imuSensor.getLinearAcceleration()[0:2] # apparently this is the correct syntax        
    # heading is always
        relative_heading = (360 * crossed360Count) + angleData[0] - initialHeading #- experiment with this
        # assuming that the robot always rotates clockwise

            
            
        rh = -1 * relative_heading # shorter variable name # because here cwise rotatom increases yaw
        U_temp = np.array([[accel_Data[0],accel_Data[1]]]).T - offset_2d
    #print('relative_heading:') # removeprint
    #print(relative_heading) # removeprint    
        if (angleData[0] > 360) or (abs(U_temp[0,0]) >= 10) or (abs(U_temp[1,0]) >= 10) : 
            # discarding i2c read fails/invalid measurements. need to check and see if this captures everything
            missedIMUCount += 1
            continue

                
        cs = math.cos(math.radians(rh))
        sn = math.sin(math.radians(rh))
        transform_to_initial_frame =  np.array([[cs, -sn],[sn, cs]])        

    #print('accel_Data:') # removeprint
    #print(accel_Data) # removeprint            
        

        #  - Preprocess U to get a useful signal - Karthik need to look in to this



#        U = U - offset_2d # removing offset to make noise mean zero - Karthik                
        U = dot(transform_to_initial_frame , U_temp)

        # get input data from IMU - end
        # 
    #print('transformed U:') # removeprint
    #print(U) # removeprint
        start = time.time()
        if i == 0:
            prevTime = start # intialize prevTime for the first time        
                        
        timeElapsed = start - prevTime
#        print('timeElapsed:')
#        print(timeElapsed)       
     
        
        if timeElapsed < 0.01 :
            requiredDelay = 0.01 - timeElapsed
#            print('requiredDelay:')
#            print(requiredDelay)          
            time.sleep(requiredDelay)
        elif timeElapsed > 0.02 :
            longerIterationCount = longerIterationCount + 1
            # making sure appropriate dt is selected
            #dt = timeElapsed
            #A = getAmatrix(dt)
            #B = getBmatrix(dt)
            #Q = getProcessCovarianceMatrix(dt)            
     

        (X, P) = kf_predict(X, P, A, Q, B, U)
        prevTime = time.time()

        
#        if (i % 2) == 0:    # might want to verify this but me thinks it works
                                                                        # determines the frequency of encoder data sampling    
                #get input from Encoder
#            Y_robot_unicode = commandRobot('<E>') # to store current encoder reading returned from robot
        uart.writeStr("<E>")
        if uart.dataAvailable(5):
            encoder_Data = uart.readStr(20)
        else:
            encoder_Data = "ND" # No Data
#            Y_robot_unicode = encoderData().decode().split(',') # to store current encoder reading returned from robot
        Y_robot_unicode = encoder_Data.decode().split(',') # to store current encoder reading returned from robot

              #handle cases where data is not returned from encoders begin
        if Y_robot_unicode == "ND":
            missedEncoderCount += 1
#                time.sleep(dt)        
#                time.sleep(0.01)
            angleData_prev_iter = angleData
            X_iter.append(X.T)
            P_iter.append(P.T)
            continue
    # handle no data from encoder - end
        try:

            Y_robot = np.array([[int(Y_robot_unicode[0]), int(Y_robot_unicode[1])]]).T
            Y_pulsesPerSec = float(Y_robot_unicode[2])
            if Y_pulsesPerSec != 0:
                Y_velocity =  0.262448650280891/(36/Y_pulsesPerSec) # velocity in the body frame of the robot 
            else:
                Y_velocity = 0           
            
        except Exception as w1:
            print('issue with encoder data')
            print(w1)
            print("Y_robot_unicode:")
            print(Y_robot_unicode)
    
        if ((Y_robot != Y_robot_prev).any()):   
            # when the robot is purely rotating about a point, its encoder doesn't change. Hence it's Y should not change
            #other wise unnecessary updates will be made. Hence the current if
#            Y_new = getNewRaidalDistance(Y_robot,angleData,angleData_prev_iter,Y_robot_prev,enc_radial_distance_prev_iter,i)
            #print('Distance from origin from encoder: ') # removeprint
            #print(Y_new) # removeprint
            Y_new = Y_new + getEncoderIncrement(Y_robot,Y_robot_prev,relative_heading,i)
            Y = np.array([[Y_velocity*math.cos(math.radians(relative_heading)),Y_new[0,0], Y_velocity*math.sin(math.radians(-1*relative_heading)),Y_new[1,0]]]).T # might
#        elif (i % 4 == 0) and ((Y_robot == Y_robot_prev_4_iter).all()):
        else:
            # for some wierd reason arduino is unable to provide zero velocity. Hence 
            # Zero velocity is being handled at edison level            
            succNochange[0,sNcCount] = i
            sNcCount = sNcCount + 1               
                
            if ((sNcCount == 4) and (succNochange[0,3] - succNochange[0,0] == 3)):
                Y[0,0]= 0  # making the linear velocity zero when the robot is stationary
                Y[2,0]= 0  # making the linear velocity zero when the robot is stationary
                
            if sNcCount == 4:
                succNochange = succNochange * 0
                sNcCount = 0                 

                 
#            and (succNochange[2] - succNochange[1] == 1)
#            and (succNochange[3] - succNochange[2] == 1)):
            # for some wierd reason arduino is unable to provide zero velocity. Hence 
            # Zero velocity is being handled at edison level
            # i%4 is to ensure faster sampling doesn't yield aero velocity                 
#            Y[0,0]= 0  # making the linear velocity zero when the robot is stationary
#            Y[2,0]= 0  # making the linear velocity zero when the robot is stationary
#            Y_robot_prev_4_iter = Y_robot
            
#        enc_radial_distance_prev_iter = Y_new
        #print('measurement input:')
        #print(Y)

    #Y[0][0] = Y[0]*cos(math.radians(angleData[0])) # might
    #Y[1][0] = Y[1]*sin(math.radians(angleData[0]))
    # encoder input end
#        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
        (X, P, K, IM, IS) = kf_update(X, P, Y, H, R)
        Y_iter.append(Y.T)            
        Y_robot_prev = Y_robot
#        if (i % 2) == 1:
#            time.sleep(0.04) # ensure sampling at apprx every 0.01 seconds        
        angleData_prev_iter = angleData
#        pdb.set_trace()
        X_iter.append(X.T)
        P_iter.append(P.T)
        U_iter.append(U.T)
        angle_iteration.append(relative_heading)
        encoder_measurement.append(Y_new)
#        end = time.time()
#        print('iteration time:')
#        print(end - start)
        
        
#        dt = end - start
        
      #    mat_X_states = np.matrix([X_iter])
#    mat_P_cov = np.matrix([P_iter])

#    with open('state.txt') as f:
#        for line in mat_X_states:
#        for line in X_iter:
#            np.savetxt(f, line, fmt='%.2f')
    #f = open("states.txt", "w")
    #pickle.dump(f,X_iter)    
    #f.close()    
    
    #f = open("Y_to_update.txt", "w")
    #pickle.dump(f,Y_iter)
    #f.close()

    #f = open("accelInput.txt", "w")
    #pickle.dump(f,U_iter)
    #f.close()    

    #f = open("encoder_measurement.txt", "w")
    #pickle.dump(f,encoder_measurement)
    #f.close()                
    
    #plt.figure(1)
    #plt.plot(X_iter[:,1])
    #plt.ylabel('X-coordinate')
    #plt.show()
    
    #plt.figure(2)
    #plt.plot(X_iter[:,3])
    #plt.ylabel('Y-coordinate')
    #plt.show()
    
    #plt.figure(3)
    #plt.plot(U_iter[:,0])
    #plt.ylabel('Acceleration X-coordinate')
    #plt.show()
    
    #plt.figure(4)
    #plt.plot(U_iter[:,1])
    #plt.ylabel('Acceleration Y-coordinate')
    #plt.show()
    
    #plt.figure(5)
    #plt.plot(Y_iter[:,0])
    #plt.ylabel('X encoder measurement')
    #plt.show()
    
    #plt.figure(6)
    #plt.plot(Y_iter[:,1])
    #plt.ylabel('Y encoder measurement')
    #plt.show()    
    
    
    
    
    end1 = time.time()
    print('iteration time:')
    print(end1 - begin)    
    
    scipy.io.savemat(matfile1, mdict={'states' : X_iter}, oned_as = 'row')

    scipy.io.savemat(matfile2, mdict={'covar' : P_iter}, oned_as = 'row')
  #  matfile1 = states.mat
    scipy.io.savemat(matfile3, mdict={'acceleration' : U_iter}, oned_as = 'row')

    scipy.io.savemat(matfile4, mdict={'relative_Heading' : angle_iteration}, oned_as = 'row')

    scipy.io.savemat(matfile5, mdict={'radial_encoder_dis' : encoder_measurement}, oned_as = 'row')
    
    scipy.io.savemat(matfile6, mdict={'Y_UpdateInput' : Y_iter}, oned_as = 'row')
    print("missedEncoderCount:")
    print(missedEncoderCount)
    print("missedIMUCount:")
    print(missedIMUCount)    
    print("longerIterationCount:")
    print(longerIterationCount)    
  #  f = open('states.txt','w')
  #  f1 = open('cov.txt','w')
  #  np.savetxt(f, X_iter, fmt='%.2f')
        
#    with open('cov.txt') as f1:
#        for line in mat_P_cov:
#        for line in P_iter:
#            np.savetxt(f1, line, fmt='%.2f')    
  #  np.savetxt(f1, P_iter, fmt='%.2f')    
  #  f.close()
  #  f1.close()
   # queue.close()


if __name__ == '__main__':
    main()
