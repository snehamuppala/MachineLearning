#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:46:15 2018

@author: snehamuppala
"""

import pandas
from sklearn.cluster import KMeans 
import numpy as np
from sklearn.utils import shuffle
import math
import csv





#----here


maxAcc = 0.0
maxIter = 0
Lambda_closedForm = 0.01 #Regularisation rate
TrainingPercent = 80 #80% of original data
ValidationPercent = 10 #10% of original data
TestPercent = 10 #10% of original data
M = 10# Number of basis functions
Lambda_SGD           = 3#Regularisation rate
learningRate = 0.01
PHI = [] # design matrix
IsSynthetic = False
itr=1000
GSCFeatures = 'GSC-Features-Data/GSC-Features.csv'
GSC_same_pairs = 'GSC-Features-Data/same_pairs.csv'
GSC_diffn_pairs = 'GSC-Features-Data/diffn_pairs.csv'

def GenerateGSCData(GSCFeatures,GSC_same_pairs,GSC_diffn_pairs):
    GSCFeatures = pandas.read_csv(GSCFeatures)
    GSC_same_pairs = pandas.read_csv(GSC_same_pairs)
    GSC_diffn_pairs = pandas.read_csv(GSC_diffn_pairs)
   
    

    merged = pandas.merge(GSC_same_pairs, GSCFeatures, how='left',
                  left_on='img_id_A', right_on='img_id')

    merged = merged.drop('img_id', 1) # drop duplicate info

    merged1 = pandas.merge(GSC_same_pairs, GSCFeatures, how='left',
                  left_on='img_id_B', right_on='img_id')
    merged1 = merged1.drop('img_id_B', 1)
    merged1 = merged1.drop('img_id_A', 1)
    merged1 = merged1.drop('img_id', 1)

    merged1 = merged1.drop('target', 1)


    concat = pandas.concat([merged,merged1],join='inner',axis=1,ignore_index=False)
    concat = concat.sample(n = 1000)#select number of samples



    merged2 = pandas.merge(GSC_diffn_pairs, GSCFeatures, how='left',
                  left_on='img_id_A', right_on='img_id')

    merged2 = merged2.drop('img_id', 1) # drop duplicate info




    merged3 = pandas.merge(GSC_diffn_pairs, GSCFeatures, how='left',
                  left_on='img_id_B', right_on='img_id')
    merged3 = merged3.drop('img_id_B', 1)
    merged3 = merged3.drop('img_id_A', 1)
    merged3 = merged3.drop('img_id', 1)

    merged3 = merged3.drop('target', 1)

    concat1 = pandas.concat([merged2,merged3],join='inner',axis=1,ignore_index=False)

    concat1 = concat1.sample(n = 1000)#select number of samples

    result = pandas.concat([concat,concat1])
    result= shuffle(result)


    
    target=np.asarray(result.iloc[:,2:3])
   
    result1=np.asarray(result.iloc[:,3:515])

    
    result2=np.asarray(result.iloc[:,515:1027])



    

    difference=abs(result1-result2)

    

    difference = pandas.DataFrame(difference)

    target = pandas.DataFrame(target)

    concatenation=np.asarray(result.iloc[:,3:1027])
    concatenation = pandas.DataFrame(concatenation)


    difference.to_csv('GSC-Features-Data/GSC_Dataset_Diff.csv',index = False)
    target.to_csv('GSC-Features-Data/GSC_Dataset_target.csv',index = False)
    concatenation.to_csv('GSC-Features-Data/GSC_Dataset_Concat.csv',index = False)

    


HumanObservedFeaturesData = 'HumanObserved-Features-Data/HumanObserved-Features-Data.csv'
HOFD_same_pairs = 'HumanObserved-Features-Data/same_pairs.csv'
HOFD_diffn_pairs = 'HumanObserved-Features-Data/diffn_pairs.csv'


def GenerateHumanDataset(HumanObservedFeaturesData,same_pairs,diffn_pairs):
    HumanObservedFeaturesData = pandas.read_csv(HumanObservedFeaturesData)
    same_pairs = pandas.read_csv(same_pairs)
    diffn_pairs = pandas.read_csv(diffn_pairs)
    merged = pandas.merge(same_pairs, HumanObservedFeaturesData, how='left',
                  left_on='img_id_A', right_on='img_id')

    merged = merged.drop('img_id', 1) # drop duplicate info
    merged = merged.drop('Unnamed: 0', 1)




    merged1 = pandas.merge(same_pairs, HumanObservedFeaturesData, how='left',
                  left_on='img_id_B', right_on='img_id')
    merged1 = merged1.drop('img_id_B', 1)
    merged1 = merged1.drop('img_id_A', 1)
    merged1 = merged1.drop('img_id', 1)
    merged1 = merged1.drop('Unnamed: 0', 1)
    merged1 = merged1.drop('target', 1)
    


    concat = pandas.concat([merged,merged1],join='inner',axis=1,ignore_index=False)

    merged2 = pandas.merge(diffn_pairs, HumanObservedFeaturesData, how='left',
                  left_on='img_id_A', right_on='img_id')

    merged2 = merged2.drop('img_id', 1) 
    merged2 = merged2.drop('Unnamed: 0', 1)



    merged3 = pandas.merge(diffn_pairs, HumanObservedFeaturesData, how='left',
                  left_on='img_id_B', right_on='img_id')
    merged3 = merged3.drop('img_id_B', 1)
    merged3 = merged3.drop('img_id_A', 1)
    merged3 = merged3.drop('img_id', 1)
    merged3 = merged3.drop('Unnamed: 0', 1)
    merged3 = merged3.drop('target', 1)

   

    concat1 = pandas.concat([merged2,merged3],join='inner',axis=1,ignore_index=False)
    

    concat1 = concat1.sample(n = 800)#select number of samples
  
    
    result = pandas.concat([concat,concat1])
    





    result = shuffle(result)
    target=np.asarray(result.iloc[:,2:3])
   
    result1=np.asarray(result.iloc[:,3:12])

   
    result2=np.asarray(result.iloc[:,12:21])



   

    difference=abs(result1-result2)

   

    difference = pandas.DataFrame(difference)

    target = pandas.DataFrame(target)

    concatenation=np.asarray(result.iloc[:,3:1027])
    concatenation = pandas.DataFrame(concatenation)

   
    
    difference.to_csv('HumanObserved-Features-Data/HumanObservedDataset_Diff.csv',index = False)
    target.to_csv('HumanObserved-Features-Data/HumanObservedDataset_target.csv',index = False)
    concatenation.to_csv('HumanObserved-Features-Data/HumanObservedDataset_Concat.csv',index = False)


#GenerateGSCData(GSCFeatures,GSC_same_pairs,GSC_diffn_pairs)
#GenerateHumanDataset(HumanObservedFeaturesData,HOFD_same_pairs,HOFD_diffn_pairs)









#The number of basis functions determines the complexity of the model. We find the data points which are close together and fit a linear regression model over the clusters separately. The number of clusters are determined iteratively. 

# In[3]:

# This funtion extracts the target varibles 
def GetTargetVector(filePath):
    
    t = []
    
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
#print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

    
#We take transpose of data matrix.
    dataMatrix = np.transpose(dataMatrix)     
#print ("Data Matrix Generated..")
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #Extracting data for traning  with 80% of data
    
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
  #print(str(TrainingPercent) + "% Training Target Generated..")
    
    return t
#converting the data into matrix form.
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
  
    return d2
# this function extracts 10 % data for validating
def GenerateValData(rawData, ValPercent, TrainingCount): #validating for 10% of 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t
#The variance of each column of the training data set i.e across each dimension of the data point shall denote the spread/radius of our gaussian basis function.  
   # Σ = [variance1      0         0      .......      0
         #    0      variance2     0      .......      0 
               #         0      variance3 .......
                       #                  .......  variance46]
   #where varianceJ = variance of jth column of dataset
   #Hence Σ is a 46 x 46 diagonal matrix of variances.
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):# we generate big sigma for only same rows and columns. 
    #we neglect the varience of different columns as we dont need them.
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(150,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


#􏰂1                      ⊤−1 􏰃
#φj(x)=exp −0.5(x−μj) Σj (x−μj) (2)
#this funtion calculates the basis function for the linear regression model which is Gaussian radial basis functions

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L


#this function gives us the values of phi
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    
    
    return phi_x
#Using these values of µj and Σ, we can now fit the RBF kernels on our data. We have all the necessary values for computing the design matrix(ϕ).
#ϕj (x) = exp (−0.5* transpose(x − µj)* inverse(Σj) *(x − µj))
  
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI


#w∗ = inv((λI + transpose(Φ) * Φ))* transpose(Φ)*y   
    #where I is identity matrix;
    #Φ is the design matrix of training data representation in M dimensions using RBF Kernels

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W
  

#Based on the validation, we apply the model on testing data for better result.
def GetValTest(VAL_PHI,W):

    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# ## Fetch and Prepare Dataset

# In[4]:


RawTarget = GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv') #contains target values
RawData   = GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Concat.csv',IsSynthetic) 





# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
print("shape:target",TrainingTarget.shape)

TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print("shape :training",TrainingData.shape)



# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))

print("val data",ValData.shape)
# ## Prepare Test Data

# In[7]:

#10% of data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))


# In[8]:


ErmsArr = []
AccuracyArr = []
#To the determine the number of clusters we use K-means algorithm which forms K clusters of given points. Here, we cluster the data into M clusters equal to number of basis functions. This will return M centres of the clusters which is mu. Mu are the data points from training set which is vector of 41-dimension.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))

Mu = kmeans.cluster_centers_
#The variance of each column of the training data set, across each dimension of the data point shall denote the spread of our basis function. Big sigma is  diagonal matrix of variance.
#generating Big sigma  which is our co varience matrix which has values in diagonal.

#We can fit the basis function and compute the design the matrix on the data using the values Mu and big sigma. 

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
#design matrix for training 
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

# weights are assigned for closed form sloution
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(Lambda_closedForm)) 

TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
#calculating and assigning the design matrix for test and validating
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[9]:




# ## Finding Erms on training, validation and test set 

# In[10]:





# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Processing----------------')
print ('----------------------------------------------------')


# In[13]:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#weights initialised ramdomly.
#weights = np.random.uniform(-1.0,1.0,size=(1,M))[0]

W_Now        = np.dot(220, W)#linearity of differentiation


L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []



for i in range(0,itr):
 #Number of iterations for SGD is 1000. The weights are updated for each data point using the formula:
#w(τ+1) = w(τ) + ∆w(τ)
#where ∆w(τ) = −η(τ)* ∇E is called the weight updates. 

    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(Lambda_SGD,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
#∇E is the derivative of our cost function, and it determines the change in cost at a particular iteration, given the weights we used to compute the cost. Since our cost function comprises of two linearly differentiable terms, the cost function and the regularisation term, we can write:
#∇E = ∇Ed + λ*∇Ew
#where ∇Ed = −(yi − transpose(w(τ))* ϕ(xi))* ϕ(xi)
#and ∇Ew = w(τ)
# y(x, w) = transpose(w*) * ϕ(x)
   
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))


    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[14]:

print ('UBITname      = snehamup')
print ('Person Number = 50288710')
print ('----------------------------------------------------')
print ("------------------CEDAR Data------------------------")
print ('----------------------------------------------------')
print ('----------LinearRegression-Gradient Descent Solution----------')
print ('1.LinearRegression-Human Observed Dataset with feature concatentation')
print ("M = 10 \nLambda  = 3\nLearning rate=0.01,\nIterations=1000")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),10)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),10)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),10)))








RawTarget = GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv') #contains target values
RawData   = GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Diff.csv',IsSynthetic) #contains  columns- features and  rows of data


# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
#print("shape:target",TrainingTarget.shape)

TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
#print("shape :training",TrainingData.shape)



# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))#taget for 
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))#

#print("val data",ValData.shape)#
# ## Prepare Test Data

# In[7]:

#10% of data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))



# In[8]:


ErmsArr = []
AccuracyArr = []
#To the determine the number of clusters we use K-means algorithm which forms K clusters of given points. Here, we cluster the data into M clusters equal to number of basis functions. This will return M centres of the clusters which is mu. Mu are the data points from training set which is vector of 41-dimension.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))

Mu = kmeans.cluster_centers_
#The variance of each column of the training data set, across each dimension of the data point shall denote the spread of our basis function. Big sigma is 41X41 diagonal matrix of variance.
#generating Big sigma  which is our co varience matrix which has values in diagonal.

#We can fit the basis function and compute the design the matrix on the data using the values Mu and big sigma. 

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
#design matrix for training 
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

# weights are assigned for closed form sloution
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(Lambda_closedForm)) 

TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
#calculating and assigning the design matrix for test and validating
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[9]:




# ## Finding Erms on training, validation and test set 

# In[10]:





# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Processing----------------')
print ('----------------------------------------------------')


# In[13]:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#weights initialised ramdomly.
#weights = np.random.uniform(-1.0,1.0,size=(1,M))[0]

W_Now        = np.dot(220, W)#linearity of differentiation
#Lambda_SGD           = 3#Regularisation rate
#learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []



for i in range(0,itr):
 #Number of iterations for SGD is 400. The weights are updated for each data point using the formula:
#w(τ+1) = w(τ) + ∆w(τ)
#where ∆w(τ) = −η(τ)* ∇E is called the weight updates. 

    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(Lambda_SGD,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
#∇E is the derivative of our cost function, and it determines the change in cost at a particular iteration, given the weights we used to compute the cost. Since our cost function comprises of two linearly differentiable terms, the cost function and the regularisation term, we can write:
#∇E = ∇Ed + λ*∇Ew
#where ∇Ed = −(yi − transpose(w(τ))* ϕ(xi))* ϕ(xi)
#and ∇Ew = w(τ)
# y(x, w) = transpose(w*) * ϕ(x)
   
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))


    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[14]:


print ('2. LinearRegression-Human Observed Dataset with feature Difference')
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),10)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),10)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),10)))




RawTarget = GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv') #contains target values
RawData   = GenerateRawData('GSC-Features-Data/GSC_Dataset_Concat.csv',IsSynthetic) #contains 41 columns- features and 62623 rows of data


# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
#print("shape:target",TrainingTarget.shape)#55699

TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
#print("shape :training",TrainingData.shape)#41x55699



# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))#taget for 6962
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))#41x6962

#print("val data",ValData.shape)#41x6962
# ## Prepare Test Data

# In[7]:

#10% of data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))



# In[8]:


ErmsArr = []
AccuracyArr = []
#To the determine the number of clusters we use K-means algorithm which forms K clusters of given points. Here, we cluster the data into M clusters equal to number of basis functions. This will return M centres of the clusters which is mu. Mu are the data points from training set which is vector of 41-dimension.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))

Mu = kmeans.cluster_centers_
#The variance of each column of the training data set, across each dimension of the data point shall denote the spread of our basis function. Big sigma is 41X41 diagonal matrix of variance.
#generating Big sigma  which is our co varience matrix which has values in diagonal.

#We can fit the basis function and compute the design the matrix on the data using the values Mu and big sigma. 

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
#design matrix for training 
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

# weights are assigned for closed form sloution
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(Lambda_closedForm)) 

TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
#calculating and assigning the design matrix for test and validating
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[9]:




# In[10]:





# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Processing----------------')
print ('----------------------------------------------------')


# In[13]:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#weights initialised ramdomly.
#weights = np.random.uniform(-1.0,1.0,size=(1,M))[0]

W_Now        = np.dot(220, W)#linearity of differentiation
#Lambda_SGD           = 3#Regularisation rate
#learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []



for i in range(0,itr):
 #Number of iterations for SGD is 400. The weights are updated for each data point using the formula:
#w(τ+1) = w(τ) + ∆w(τ)
#where ∆w(τ) = −η(τ)* ∇E is called the weight updates. 

    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(Lambda_SGD,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
#∇E is the derivative of our cost function, and it determines the change in cost at a particular iteration, given the weights we used to compute the cost. Since our cost function comprises of two linearly differentiable terms, the cost function and the regularisation term, we can write:
#∇E = ∇Ed + λ*∇Ew
#where ∇Ed = −(yi − transpose(w(τ))* ϕ(xi))* ϕ(xi)
#and ∇Ew = w(τ)
# y(x, w) = transpose(w*) * ϕ(x)
   
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))


    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[14]:


print ('3. LinearRegression-GSC Dataset with feature concatentation')
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),10)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),10)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),10)))

RawTarget = GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv') #contains target values
RawData   = GenerateRawData('GSC-Features-Data/GSC_Dataset_Diff.csv',IsSynthetic) #contains 41 columns- features and 62623 rows of data


# ## Prepare Training Data

# In[5]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
#print("shape:target",TrainingTarget.shape)#55699

TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
#print("shape :training",TrainingData.shape)#41x55699



# ## Prepare Validation Data

# In[6]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))#taget for 6962
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))#41x6962

#print("val data",ValData.shape)#41x6962
# ## Prepare Test Data

# In[7]:

#10% of data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))



# In[8]:


ErmsArr = []
AccuracyArr = []
#To the determine the number of clusters we use K-means algorithm which forms K clusters of given points. Here, we cluster the data into M clusters equal to number of basis functions. This will return M centres of the clusters which is mu. Mu are the data points from training set which is vector of 41-dimension.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))

Mu = kmeans.cluster_centers_
#The variance of each column of the training data set, across each dimension of the data point shall denote the spread of our basis function. Big sigma is 41X41 diagonal matrix of variance.
#generating Big sigma  which is our co varience matrix which has values in diagonal.

#We can fit the basis function and compute the design the matrix on the data using the values Mu and big sigma. 

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
#design matrix for training 
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)

# weights are assigned for closed form sloution
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(Lambda_closedForm)) 

TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
#calculating and assigning the design matrix for test and validating
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[9]:




# ## Finding Erms on training, validation and test set 

# In[10]:





# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Processing----------------')
print ('----------------------------------------------------')


# In[13]:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#weights initialised ramdomly.
#weights = np.random.uniform(-1.0,1.0,size=(1,M))[0]

W_Now        = np.dot(220, W)#linearity of differentiation
#Lambda_SGD           = 3#Regularisation rate
#learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []



for i in range(0,itr):
 #Number of iterations for SGD is 400. The weights are updated for each data point using the formula:
#w(τ+1) = w(τ) + ∆w(τ)
#where ∆w(τ) = −η(τ)* ∇E is called the weight updates. 

    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(Lambda_SGD,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
#∇E is the derivative of our cost function, and it determines the change in cost at a particular iteration, given the weights we used to compute the cost. Since our cost function comprises of two linearly differentiable terms, the cost function and the regularisation term, we can write:
#∇E = ∇Ed + λ*∇Ew
#where ∇Ed = −(yi − transpose(w(τ))* ϕ(xi))* ϕ(xi)
#and ∇Ew = w(τ)
# y(x, w) = transpose(w*) * ϕ(x)
   
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))


    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[14]:


print ('4.Linear Regression-GSC Dataset with feature subtraction')

print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),10)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),10)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),10)))





from matplotlib import pyplot as plt

#readings taken during hyperparameters tunning
print ('1.LinearRegression-Human Observed Dataset with feature concatentation')
BF=[5,10,15]
training=[0.514932374,0.495757611,0.4954817769]
val=[0.5215131037,0.4981246284,0.4979810919]





plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.6, .6, "λ=3,η=0.01")
plt.ylim(0.2, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()

lr=[0.01,0.05,0.09]

#training=[0.495757611,0.4975578181,0.5247626393]
val=[0.4981246284,0.4979970493,0.5169295492]

plt.figure(figsize=[5, 5])
plt.plot(lr, val, marker='.')
#plt.plot(lr, training, marker='.')
plt.figtext(.3, .3, "λ=3,BF=10")
plt.ylim(0.2, 0.6)
plt.xlabel('Learning rate')
plt.ylabel('Root mean square Error')
plt.show()

print ('2. LinearRegression-Human Observed Dataset with feature Difference')
BF=[5,10,15]
training=[0.5036766548,0.4974597985,0.4973721643]
val=[0.5076592795,0.4942176202,0.4933907477]





plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.6, .6, "λ=3,η=0.01")
plt.ylim(0.2, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()

lr=[0.01,0.05,0.09]

#training=[0.4974597985,0.4978407797,0.5402101939]
val=[0.4942176202,0.4935719783,0.5290253308]





plt.figure(figsize=[5, 5])
plt.plot(lr, val, marker='.')
#plt.plot(lr, training, marker='.')
plt.figtext(.3, .3, "λ=3,BF=10")
plt.ylim(0.2, 0.6)
plt.xlabel('Learning rate')
plt.ylabel('Root mean square Error')
plt.show()

print ('3. LinearRegression-GSC Dataset with feature concatentation')

BF=[5,10,15]
training=[0.516236252,0.4982504067,0.4965839395]
val=[0.5084968329,0.496756666,0.4965709321]





plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.3, .3, "λ=3,η=0.01")
plt.ylim(0.2, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()

lr=[0.01,0.05,0.09]

#training=[0.4982504067,0.5010252699,0.5076807066]
val=[0.496756666,0.4973360245,0.5090027799]





plt.figure(figsize=[5, 5])
plt.plot(lr, val, marker='.')
#plt.plot(lr, training, marker='.')
plt.figtext(.3, .3, "λ=3,BF=10")
plt.ylim(0.2, 0.6)
plt.xlabel('Learning rate')
plt.ylabel('Root mean square Error')
plt.show()

print ('4.Linear Regression-GSC Dataset with feature subtraction')

BF=[5,10,15]
training=[0.4918891574,0.4566038783,0.4496515217]
val=[0.4812424068,0.4482519345, 0.4448588491]





plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.3, .3, "λ=3,η=0.01")
plt.ylim(0.2, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()






lr=[0.01,0.05,0.09]

#training=[0.4982504067,0.4531679284,0.4854011097]
val=[0.4482519345,0.4485356595,0.4911797325]

plt.figure(figsize=[5, 5])
plt.plot(lr, val, marker='.')
#plt.plot(lr, training, marker='.')
plt.figtext(.3, .3, "λ=3,BF=10")
plt.ylim(0.2, 0.6)
plt.xlabel('Learning rate')
plt.ylabel('Root mean square Error')
plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score






def GetTargetVector(filePath):
    
    t = []
    
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
#print("Raw Training Generated..")
    return t
#This funtion extracts data from uerylevelnorm_X.csv 
def GenerateRawData(filePath):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

    
#We take transpose of data matrix.
    #dataMatrix = np.transpose(dataMatrix)     
#print ("Data Matrix Generated..")
    return dataMatrix  
            
#contains 41 columns- features and 62623 rows of data




np.random.seed(12)
#num_observations = 5000

#x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
#x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

#simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
#simulated_labels = np.hstack((np.zeros(num_observations),
                              #np.ones(num_observations)))



def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result
def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs

def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

#inp_df   = np.asarray(GenerateRawData('HumanObservedDataset_Concat.csv') )
#out_df = np.asarray(GetTargetVector('HumanObservedDataset_target.csv')) #contains target values


def Logistic_Regression(inp_df,out_df):
    
    scaler = StandardScaler()
    inp_df = scaler.fit_transform(inp_df,out_df)
#
    X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.20, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
    X_tr_arr = X_train
    X_ts_arr = X_test
    y_tr_arr = y_train
    y_ts_arr = y_test
    X_vl_arr = X_val
    y_vl_arr = y_val
    

    #print('Input Shape', (X_tr_arr.shape))
    #print('Output Shape', X_test.shape)
#Get number of features
    n_features = X_tr_arr.shape[1]
    #print('Number of Features', n_features)
    w, b = weightInitialization(n_features)
#Gradient Descent
    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=300)
#Final prediction
    w = coeff["w"]
    b = coeff["b"]
    #print('Optimized weights', w)
    #print('Optimized intercept',b)
#
    final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
    final_val_pred = sigmoid_activation(np.dot(w,X_vl_arr.T)+b)
#
    m_tr =  X_tr_arr.shape[0]
    m_ts =  X_ts_arr.shape[0]
    m_vl =  X_vl_arr.shape[0]
    y_tr_pred = predict(final_train_pred, m_tr)
    print('Training Accuracy',(accuracy_score(y_tr_pred.T, y_tr_arr))*100)
    y_vl_pred = predict(final_val_pred, m_vl)
    print('Validation Accuracy',(accuracy_score(y_vl_pred.T, y_vl_arr))*100)
    y_ts_pred = predict(final_test_pred, m_ts)
    print('Test Accuracy',(accuracy_score(y_ts_pred.T, y_ts_arr))*100)


    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title('Cost reduction')
    plt.show()



HOD_Concat   = np.asarray(GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Concat.csv') )
HOD_Concat_target = np.asarray(GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv')) #contains target values
print ('----------Logistic Regression----------')
print ("learning_rate=0.0001\n  no_iterations=300000")
print("sigmoid_activation")
print ('1.Logistic Regression-Human Observed Dataset with feature concatentation')

Logistic_Regression(HOD_Concat,HOD_Concat_target)

HOD_Diff   = np.asarray(GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Diff.csv') )
HOD_Diff_target  = np.asarray(GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv')) #contains target values

print ('2. Logistic Regression-Human Observed Dataset with feature Difference')
Logistic_Regression(HOD_Concat,HOD_Diff_target)

GSC_Concat   = np.asarray(GenerateRawData('GSC-Features-Data/GSC_Dataset_Concat.csv') )
GSC_Concat_target = np.asarray(GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv')) #contains target values
print ('3. Logistic Regression-GSC Dataset with feature concatentation')
Logistic_Regression(GSC_Concat,GSC_Concat_target)

GSC_Diff  = np.asarray(GenerateRawData('GSC-Features-Data/GSC_Dataset_Diff.csv') )
GSC_Diff_target = np.asarray(GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv')) #contains target values

print ('4.Logistic Regression-GSC Dataset with feature subtraction')
Logistic_Regression(GSC_Diff,GSC_Diff_target)




import numpy as np  
import pandas as pd  
import tensorflow as tf  
import urllib.request as request  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn.model_selection import train_test_split


def GetTargetVector(filePath):
    
    t = []
    
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
#print("Raw Training Generated..")
    return t
#This funtion extracts data from uerylevelnorm_X.csv 
def GenerateRawData(filePath):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

    
#We take transpose of data matrix.
    #dataMatrix = np.transpose(dataMatrix)     
#print ("Data Matrix Generated..")
    return dataMatrix  
            
#contains 41 columns- features and 62623 rows of data




np.random.seed(12)
#num_observations = 5000

#x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
#x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

#simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
#simulated_labels = np.hstack((np.zeros(num_observations),
                              #np.ones(num_observations)))

#simulated_labels = np.asarray(GetTargetVector('output_t.csv')) #contains target values
#simulated_separableish_features   = np.asarray(GenerateRawData('output.csv') )
print("Activation function= sigmoid ")
print("Optimization method= GradientDescentOptimizer")                             
HumanObservedDataset_target = np.asarray(GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv')) #contains target values
HumanObservedDataset_Concat   = np.asarray(GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Concat.csv') )

Xtrain, Xtest, ytrain, ytest = train_test_split(HumanObservedDataset_Concat, HumanObservedDataset_target, test_size=0.2, random_state=0)
ytrain = pd.get_dummies(ytrain)  
ytest = pd.get_dummies(ytest)


 
def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(Xtrain.shape[0], Xtrain.shape[1]), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(ytrain.shape[0], ytrain.shape[1]), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand( Xtrain.shape[1], hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain.shape[1]), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain, y: ytrain}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2
num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000
print("Neural Network-Human Observed Dataset with feature concatentation")
plt.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: -%d-" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
X = tf.placeholder(shape=(Xtest.shape[0], Xtest.shape[1]), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(ytest.shape[0], ytest.shape[1]), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest.values)]
    accuracy = 100 * sum(correct) / len(correct)
    
    print('Neural Network -%d-, accuracy: %.2f%%' % (hidden_nodes, accuracy))


HumanObservedDataset_target1 = np.asarray(GetTargetVector('HumanObserved-Features-Data/HumanObservedDataset_target.csv')) #contains target values
HumanObservedDataset_Diff   = np.asarray(GenerateRawData('HumanObserved-Features-Data/HumanObservedDataset_Diff.csv') )

Xtrain1, Xtest1, ytrain1, ytest1 = train_test_split(HumanObservedDataset_Diff, HumanObservedDataset_target1, test_size=0.2, random_state=0)
ytrain1 = pd.get_dummies(ytrain1)  
ytest1 = pd.get_dummies(ytest1)


 
def create_train_model1(hidden_nodes, num_iters):

    # Reset the graph
    #tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(Xtrain1.shape[0], Xtrain1.shape[1]), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(ytrain1.shape[0], ytrain1.shape[1]), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(Xtrain1.shape[1], hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain1.shape[1]), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain1, y: ytrain1})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain1, y: ytrain1}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000
print("Neural Network-Human Observed Dataset with feature subtraction")
plt1.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model1(hidden_nodes, num_iters)
    plt1.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: -%d-" % hidden_nodes)

plt1.xlabel('Iteration', fontsize=12)  
plt1.ylabel('Loss', fontsize=12)  
plt1.legend(fontsize=12)  
X = tf.placeholder(shape=(Xtest1.shape[0], Xtest1.shape[1]), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(ytest1.shape[0], ytest1.shape[1]), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest1, y: ytest1})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest1.values)]
    accuracy = 100 * sum(correct) / len(correct)
    
    print('Neural Network -%d-, accuracy: %.2f%%' % (hidden_nodes, accuracy))
    
    
    
GSC_Dataset_target = np.asarray(GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv')) #contains target values
GSC_Dataset_Concat  = np.asarray(GenerateRawData('GSC-Features-Data/GSC_Dataset_Concat.csv') )

Xtrain2, Xtest2, ytrain2, ytest2= train_test_split(GSC_Dataset_Concat, GSC_Dataset_target, test_size=0.2, random_state=0)
ytrain2 = pd.get_dummies(ytrain2)  
ytest2 = pd.get_dummies(ytest2)


 


def create_train_model2(hidden_nodes, num_iters):

    # Reset the graph
    #tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(Xtrain2.shape[0], Xtrain2.shape[1]), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(ytrain2.shape[0], ytrain2.shape[1]), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(Xtrain2.shape[1], hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain2.shape[1]), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain2, y: ytrain2})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain2, y: ytrain2}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000
print("Neural Network-GSC Dataset with feature concatentation")
plt1.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model2(hidden_nodes, num_iters)
    plt1.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

plt1.xlabel('Iteration', fontsize=12)  
plt1.ylabel('Loss', fontsize=12)  
plt1.legend(fontsize=12)  
X = tf.placeholder(shape=(Xtest2.shape[0], Xtest2.shape[1]), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(ytest2.shape[0], ytest2.shape[1]), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest2, y: ytest2})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest2.values)]
    accuracy = 100 * sum(correct) / len(correct)
   
    print('Neural Network -%d-, accuracy: %.2f%%' % (hidden_nodes, accuracy))
    
       
    
    




GSC_Dataset_target1 = np.asarray(GetTargetVector('GSC-Features-Data/GSC_Dataset_target.csv')) #contains target values
GSC_Dataset_Diff   = np.asarray(GenerateRawData('GSC-Features-Data/GSC_Dataset_Diff.csv') )

Xtrain3, Xtest3, ytrain3, ytest3= train_test_split(GSC_Dataset_Diff, GSC_Dataset_target1, test_size=0.2, random_state=0)
ytrain3 = pd.get_dummies(ytrain3)  
ytest3 = pd.get_dummies(ytest3)




def create_train_model3(hidden_nodes, num_iters):

    # Reset the graph
    #tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(Xtrain3.shape[0], Xtrain3.shape[1]), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(ytrain3.shape[0], ytrain3.shape[1]), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(Xtrain3.shape[1], hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain3.shape[1]), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain3, y: ytrain3})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain3, y: ytrain3}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000
print("Neural Network-GSC Dataset with feature subtraction")
plt1.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model3(hidden_nodes, num_iters)
    plt1.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: -%d-" % hidden_nodes)

plt1.xlabel('Iteration', fontsize=12)  
plt1.ylabel('Loss', fontsize=12)  
plt1.legend(fontsize=12)  
X = tf.placeholder(shape=(Xtest3.shape[0], Xtest3.shape[1]), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(ytest3.shape[0], ytest3.shape[1]), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest3, y: ytest3})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest3.values)]
    accuracy = 100 * sum(correct) / len(correct)
   
    print('Neural Network  -%d-, accuracy: %.2f%%' % (hidden_nodes, accuracy))




