# -*- coding: utf-8 -*-
"""

@author: Sushant

"""

import numpy as np
import scipy

######################################## Least Square Linear Regression ####################################
def LinearRegression(data,labels):
    numdata = int( np.size(data,0) )
    b1 = np.hstack(( data,np.ones((numdata,1)) ) ) 
    XXT = np.matmul(b1.T,b1)
    invXXT = np.linalg.pinv(XXT)
    b2 = np.matmul(b1.T,labels)
    w = np.matmul(invXXT,b2)
    return w

###################################### Ridges Regression ###################################################

def RidgeRegression(data,labels,lambda):
    numdata = int( np.size(data,0) )
    b1 = np.hstack(( data,np.ones((numdata,1)) ) ) 
    XXT = np.matmul(b1.T,b1) + lambda*np.identity(np.size(b1,1))
    b2 = np.matmul(b1.T,labels)
    #solved using Cholesky decompostion Ax = b
    b3 = scipy.linalg.cho_factor( XXT )
    w = scipy.linalg.cho_solve(b3,b2)
    return w

###################################### Predict values and Least Square Error###############################
def PredictLabels(testdata,w):
    numdata = int( np.size(testdata,0) )
    b3 = np.hstack( ( testdata,np.ones((numdata,1)) ) )
    pred = np.matmul(b3,w)
    return pred
        
def ltsqerror(prelabels,actlabels):
    return np.sum((prelabels-actlabels)**2)/int(np.size(prelabels,0))

####################################### Kernel Ridges Regression ########################################

def linear(x1,x2,p = None):
    return np.dot(x1,x2)

def polynomial(x1,x2,d):
    return ( 1+np.dot(x1,x2) )**d

def rbf(x1,x2,l):
    return  np.exp( -np.divide(np.dot(x1-x2,x1-x2), 2*(l**2 ) )    )

def KernelRidgeRegression(data,labels,lamda,kernel,p):
    numdata =  int( np.size(data,0) )
    traindata = np.asarray(data)
    #=========Kernel matrix======================
    K =  np.zeros((numdata,numdata))
    for i in range(0,numdata):
        for j in range(0,numdata):
            K[i,j] = kernel(traindata[i,:],traindata[j,:],p)
    #solved using Cholesky decompostion Ax = b
    b1 = scipy.linalg.cho_factor( K + lamda*np.identity(numdata) )
    alphas = scipy.linalg.cho_solve(b1,labels)
    return alphas

def KernelRidgesRegression_predict(traindata1,alphas,testdata1,kernel,p):
    numtraindata =  int( np.size(traindata1,0) )
    numtestdata =  int( np.size(testdata1,0) )
    traindata = np.asarray(traindata1)
    testdata = np.asarray(testdata1)
    predlabels = np.zeros((numtestdata,1))
    K =  np.zeros((numtestdata,numtraindata))
    for j in range(0,numtestdata):    
        for i in range(0,numtraindata):
            K[j,i] = kernel(traindata[i,:],testdata[j,:],p)
    predlabels = np.dot(K,alphas)

    return predlabels
            
