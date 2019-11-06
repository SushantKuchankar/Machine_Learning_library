# -*- coding: utf-8 -*-

#==============================================================================
# Author:     Sushant Kuchankar 
#     data   - m X n+1 matrix, where m is the number of training points and n+1 th column corresponds to vector of training labels for the training data
#==============================================================================


import numpy as np
from qpsolvers import solve_qp


def linear(x1,x2,p = None):
    return np.dot(x1,x2)

def polynomial(x1,x2,d):
    return ( 1+np.dot(x1,x2) )**d

def rbf(x1,x2,l):
    return  np.exp( -np.divide(np.dot(x1-x2,x1-x2), 2*(l**2 ) )    )

def ND_hyperplane(x1,svectors,labels,alphas,kernel,p=None):
    output = 0
    num = int( np.size(svectors,0) )
    for i in range(0,num):
        output += alphas[i]*labels[i]*kernel(x1,svectors[i],p)
    return output

#################################################OVA#########################################

def Grammatrix(data,kernel,p=1):
    
#==============================================================================
    traindata = np.asarray(data)
    trainlabels = np.asarray(labels)
    
    numdata =  int( np.size(traindata,0) )
    #=========Gram matrix======================
    X =  np.zeros((numdata,numdata))
    for i in range(0,numdata):
        print(i)
        for j in range(0,numdata):
            X[i,j] = kernel(traindata[i,:],traindata[j,:],p)
    
    return X


def SVM_learner_OVA(gram,data,labels,C,kernel,p = None ):
    
#==============================================================================    
    X = np.asarray(gram)
    traindata = np.asarray(data)
    trainlabels = np.asarray(labels)
    
    numdata =  int( np.size(gram,0) )
    #==========Kernal matrix==================
    Y = np.outer(trainlabels,trainlabels)
    P = np.multiply(X,Y)
        
    #==========minus of p1 norm===================================
    q = np.ones( numdata )*(-1)
    
    #============equality constraits==============================
    A = trainlabels.reshape( (1,numdata) )
    b = np.zeros(1)
    
    #================ineqaulity constraits========================
    
    G = np.vstack( (np.identity(numdata)*(-1),np.identity(numdata) ) )
    h = np.hstack( (np.zeros(numdata),np.ones(numdata)*C) )
    
    #=================quadratic minimization======================
    
    try:
        alphas = solve_qp(P, q, G, h, A, b)
    except ValueError:
        P = P + (np.identity(numdata))*(1e-5)
        alphas = solve_qp(P, q, G, h, A, b)
    
    #all alphas not approximately equal to zero are support vectors
    
    index = np.where(alphas > 1e-5)
    support_vector_alphas = alphas[index[0]]
    support_vector_labels = trainlabels[index[0]]
    support_vectors = traindata[index[0],:]
    
    
    #==================bias==============================================
    b1 = []
    for i in range(0,len(index[0]) ):
        if(support_vector_alphas[i]< C- 1e-5):
            b1.append( support_vector_labels[i]  - ND_hyperplane(support_vectors[i],support_vectors,support_vector_labels,support_vector_alphas,kernel,p) )
    b = np.mean(b1)
    
    
#==============================================================================
    
    class model_struct:
        pass
    
    model = model_struct()
    model.b = b
    model.sv = support_vectors
    model.sv_alphas = support_vector_alphas
    model.sv_labels = support_vector_labels
    model.kernel = kernel
    model.p = p
    
    return model



def ova(gram,data,labels,C,kernel,p = None):

#INPUT: labels - labels for all the OVA subclassifer  
    
    numclasses =  int( np.size(labels,1) )
    
    class ova_struct:
        pass
    ovamodel = ova_struct()
    
    for i in range(0,numclasses):
        print(i)
        x = "ovamodel.model" +str(i)
        exec("%s=SVM_learner_OVA(gram,data,labels[:,%d],10,rbf,p = 1 )" %(x,i) )
        
    return ovamodel


def SVM_classifier_vote(data, model):
    
    testdata = np.asarray(data)
#==============================================================================                           
    b = model.b
    numdata = int( np.size(testdata,0) )
    distance_from_hyperplane = np.empty((numdata,1))
    #calculate perpendicular distance of testvector to hyperplane
    for i in range(0,numdata):        
        distance_from_hyperplane[i] =  ND_hyperplane(testdata[i,:],model.sv,model.sv_labels,model.sv_alphas,model.kernel,model.p)   + b
    #sign function for the labels
    return distance_from_hyperplane

def OVA_classifier(data,ovamodel):
    testdata = np.asarray(data)
    numdata = int( np.size(testdata,0) )
    G = np.zeros((numdata,1))
    for i in range(0,10):
        print(i)
        x = "ovamodel.model"+str(i)
        exec("ouput = SVM_classifier_vote(data,%s)"%(x) )
        G = np.hstack((G,ouput))
        predictedlabels = np.argmax(G[:,1:], axis=1)
    return predictedlabels


def SVM_learner_OVO(data,labels,C,kernel,p = None ):
    
#==============================================================================
    traindata = np.asarray(data)
    trainlabels = np.asarray(labels)
    
    numdata =  int( np.size(traindata,0) )
    #=========Gram matrix======================
    X =  np.zeros((numdata,numdata))
    for i in range(0,numdata):
        for j in range(0,numdata):
            X[i,j] = kernel(traindata[i,:],traindata[j,:],p) 
    
    #==========Kernal matrix==================
    Y = np.outer(trainlabels,trainlabels)
    P = np.multiply(X,Y)
        
    #==========minus of p1 norm===================================
    q = np.ones( numdata )*(-1)
    
    #============equality constraits==============================
    A = trainlabels.reshape( (1,numdata) )
    b = np.zeros(1)
    
    #================ineqaulity constraits========================
    
    G = np.vstack( (np.identity(numdata)*(-1),np.identity(numdata) ) )
    h = np.hstack( (np.zeros(numdata),np.ones(numdata)*C) )
    
    #=================quadratic minimization======================
    
    try:
        alphas = solve_qp(P, q, G, h, A, b)
    except ValueError:
        P = P + (np.identity(numdata))*(1e-5)
        alphas = solve_qp(P, q, G, h, A, b)
    
    #all alphas not approximately equal to zero are support vectors
    
    index = np.where(alphas > 1e-5)
    support_vector_alphas = alphas[index[0]]
    support_vector_labels = trainlabels[index[0]]
    support_vectors = traindata[index[0],:]
    
    
    #==================bias==============================================
    b1 = []
    for i in range(0,len(index[0]) ):
        if(support_vector_alphas[i]< C- 1e-5):
            b1.append( support_vector_labels[i]  - ND_hyperplane(support_vectors[i],support_vectors,support_vector_labels,support_vector_alphas,kernel,p) )
    b = np.mean(b1)
    
    
#==============================================================================
    
    class model_struct:
        pass
    
    
    model = model_struct()
    model.b = b
    model.sv = support_vectors
    model.sv_alphas = support_vector_alphas
    model.sv_labels = support_vector_labels
    model.kernel = kernel
    model.p = p
    
    return model



def OVO(data,C,kernel,p = None):
    #data is list of numpy array for each class separated 
    class ovo_struct:
        pass
    ovomodel = ovo_struct()
    for i in range(0,10):
        print(i)
        for j in range(i+1,10):
            tempdata = np.vstack((np.asarray(data[i]),np.asarray(data[j])))
            templabels = np.vstack((np.ones(( len(data[i]),1)),-np.ones(( len(data[j]),1) ) )  )
            x = "ovomodel.model" +str(i)+str(j)
            exec("%s=SVM_learner_OVO(tempdata,templabels,10,rbf,p = 1 )" %(x) )
    
    return ovomodel


def SVM_classifier(data, model):
        
    testdata = np.asarray(data)
#==============================================================================                           
    b = model.b
    numdata = int( np.size(testdata,0) )
    distance_from_hyperplane = np.empty((numdata,1))
    #calculate perpendicular distance of testvector to hyperplane
    for i in range(0,numdata):        
        distance_from_hyperplane[i] =  ND_hyperplane(testdata[i,:],model.sv,model.sv_labels,model.sv_alphas,model.kernel,model.p)   + b
    #sign function for the labels
    predictedlabels = np.sign(distance_from_hyperplane)

    return predictedlabels

def OVO_classifier(data,ovomodel):
    testdata = np.asarray(data)
    numdata = int( np.size(testdata,0) )
    G = np.zeros((numdata,1))
    for i in range(0,10):
        print(i)
        for j in range(i+1,10):
            x = "ovomodel.model" +str(i)+str(j)
            exec("ouput = SVM_classifier(data,%s)"%(x) )
            G = np.hstack((G,ouput))
    return G[:,1:]


def vote(out):
    count =0
    ballet = np.zeros((10,1))
    for i in range(0,10):
        for j in range(i+1,10):
            if(out[count]==1):
                ballet[i] +=1
            else:
                ballet[j] +=1
            count+=1
    return np.argmax(ballet)

###################################Performace_Evaluation###################################

def classification_error(predictedlabels, testlabels):    
    err = float(np.sum(predictedlabels != testlabels))
    percentage_error = err*100/len(testlabels)    
    return percentage_error

def confusionmat(testlabels,predictedlabels):
    conmat = np.zeros((10,10))
    for i in range(0,len(testlabels)):
        conmat[ int(testlabels[i]),int(predictedlabels[i]) ] +=1
    return conmat
