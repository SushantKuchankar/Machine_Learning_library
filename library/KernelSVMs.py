'''
Author: Sushant Kuchankar

'''
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

    
def SVM_learner(data,C,kernel,p = None ):
    
#     INPUT : 
#     data   - m X n+1 matrix, where m is the number of training points and n+1 th column corresponds to vector of training labels for the training data
#     C      - SVM regularization parameter (positive real number)
#     
#     
#     OUTPUT :
#     returns the structure 'model' which has the following fields:
#     
#     b - SVM bias term
#     sv - the subset of training data, which are the support vectors
#     sv_alphas - m X 1 vector of support vector coefficients
#     sv_labels - corresponding labels of the support vectors
#     
#	  Install "qpsolvers" package
#     Github link: https://github.com/stephane-caron/qpsolvers
                 
#                 alphas = solve_qp(P, q, G, h, A, u)

#==============================================================================
    
    traindata = np.asarray(data[:,:2])
    trainlabels = np.asarray(data[:,2:])
    
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
    
    
def SVM_classifier(data, model):
    

#     INPUT
#     testdata - m X n matrix of the test data samples
#     # model    - SVM model structure returned by SVM_learner
#     
#     OUTPUT
#     predictedlabels - m x 1 vector of predicted labels
#     
#     Write code here to find predictedlabels


    
    testdata = np.asarray(data[:,:2])
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
