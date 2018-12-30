
# coding: utf-8

# In[1]:
from sklearn.cluster import KMeans 
#Scikit-learn is a free software machine learning library for the Python
# it has various machine learning algorithms-eg: Kmeans.
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt


# In[2]:

list_lambda=[0.001,0.01,0.1,0.5]
for Lambda_closedForm in list_lambda:
	maxAcc = 0.0
	maxIter = 0
	#Lambda_closedForm = 0.01 #Regularisation rate
	TrainingPercent = 80 #80% of original data
	ValidationPercent = 10 #10% of original data
	TestPercent = 10 #10% of original data
	M = 10# Number of basis functions
	PHI = [] # design matrix
	IsSynthetic = False
	#The number of basis functions determines the complexity of the model. We find the data points which are close together and fit a linear regression model over the clusters separately. The number of clusters are determined iteratively. 

	# In[3]:

	# This funtion extracts the target varibles from Querylevelnorm_t.csv returns as t-69623
	def GetTargetVector(filePath):
		t = []
		with open(filePath, 'rU') as f:
			reader = csv.reader(f)
			for row in reader:  
				t.append(int(row[0]))
	#print("Raw Training Generated..")
		return t
	#This funtion extracts data from uerylevelnorm_X.csv 
	def GenerateRawData(filePath, IsSynthetic):    
		dataMatrix = [] 
		with open(filePath, 'rU') as fi:
			reader = csv.reader(fi)
			for row in reader:
				dataRow = []
				for column in row:
					dataRow.append(float(column))
				dataMatrix.append(dataRow)   

		if IsSynthetic == False :
	#deleting the columns 5,6,7,8,9 because of 0 varience, with 0 with cannot find inverse matrix. So 46 columns to 41
			dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
	#We take transpose of data matrix.
		dataMatrix = np.transpose(dataMatrix)     
	#print ("Data Matrix Generated..")
		return dataMatrix

	def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #Extracting data for traning  with 80% of data- 55698
		
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
	def GenerateValData(rawData, ValPercent, TrainingCount): #validating for 10% of data-6962
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
	  
	def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
		DataT = np.transpose(Data)
		TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01)) #why        
		PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
		BigSigInv = np.linalg.inv(BigSigma)
		for  C in range(0,len(MuMatrix)):
			for R in range(0,int(TrainingLen)):
				PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
		#print ("PHI Generated..")
	   
		return PHI
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


	RawTarget = GetTargetVector('Querylevelnorm_t.csv') #contains target values
	RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic) #contains 41 columns- features and 62623 rows of data


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

	#print("test data",ValData.shape)#41x6962
	#print("Amount of Training data:",TrainingData.shape[1])
	#print("Amount of Testing data:",len(TestDataAct))#6962
	#print("Amount of Validation data:",len(ValDataAct))
	# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

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


	#print("Mu shape:",Mu.shape)
	#@print("big sigma:",BigSigma.shape)
	#print("phi",TRAINING_PHI.shape)
	#print("weigth:",W.shape)
	#print(VAL_PHI.shape)
	#print(TEST_PHI.shape)

	# ## Finding Erms on training, validation and test set 

	# In[10]:


	TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
	VAL_TEST_OUT = GetValTest(VAL_PHI,W)
	TEST_OUT     = GetValTest(TEST_PHI,W)
	#values are getting assigned to training, validation and testing.
	TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
	ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
	TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


	# In[11]:


	print ('UBITname      = snehamup')
	print ('Person Number = 50288710')
	print ('----------------------------------------------------')
	print ("------------------LeToR Data------------------------")
	print ('----------------------------------------------------')
	print ("-------Closed Form with Radial Basis Function-------")
	print ('----------------------------------------------------')
	print ("M = 10 \nLambda = ",Lambda_closedForm)
	print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
	print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
	print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[12]:


print ('----------------------------------------------------')
print ('--------------Processing----------------')
print ('----------------------------------------------------')


# In[13]:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#weights initialised ramdomly.

Lambda_SGD =[0.5,1,2,3,4] #Regularisation rate



for Lambda_SGD in Lambda_SGD:
	learningRate = 0.1
	weights = np.random.uniform(-1.0,1.0,size=(1,M))[0]
	W_Now        = np.dot(220, weights)#linearity of differentiation
	L_Erms_Val   = []
	L_Erms_TR    = []
	L_Erms_Test  = []
	W_Mat        = []
	for i in range(0,400):
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

	print ('UBITname      = snehamup')
	print ('Person Number = 50288710')
	print ('----------------------------------------------------')
	print ("------------------LeToR Data------------------------")
	print ('----------------------------------------------------')
	print ('----------Gradient Descent Solution--------------------')
	print ("M = 10 \n\neta=0.01")
	print ("lambda",Lambda_SGD)
	print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),10)))
	print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),10)))
	print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),10)))




print ('---------------------Graphs--> Closed form Solution-------------------------------')



"""
Features Commonality plot
"""



BF=[5,10,15,25,30]


val=[0.5410875,0.5384290,0.537345,0.537910,0.537700]
training=[0.552832,0.5494546,0.546866,0.544095,0.542377]

plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.6, .6, "λ=0.001")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()



val1=[0.541087,0.538421,0.537367,0.537706,0.537234]
train1=[0.55283,0.54945,0.546889,0.5443064,0.542753]

plt.figure(figsize=[5, 5])
plt.plot(BF, val1, marker='.')
plt.plot(BF, train1, marker='.')
plt.figtext(.6, .6, "λ=0.01")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()


val2=[0.54108,0.53894,0.538535,0.53828,0.537814]
train2=[0.552833,0.549709,0.5481073,0.546977,0.545845]

plt.figure(figsize=[5, 5])
plt.plot(BF, val2, marker='.')
plt.plot(BF, train2, marker='.')
plt.figtext(.6, .6, "λ=0.5")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()



val3=[0.541079,0.539256,0.538804,0.53855,0.538120]
train3=[0.552834,0.549896,0.548456,0.547591,0.546684]

plt.figure(figsize=[5, 5])
plt.plot(BF, val3, marker='.')
plt.plot(BF, train3, marker='.')
plt.figtext(.6, .6, "λ=1")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()



print ('---------------------Graphs--> Stochastic gradient descent solution -------------------------------')




BF=[5,10,25,30]


val=[3.49319,3.86242,4.1487,4.37489]
training=[3.46893,3.99449,4.21814,4.43802]

plt.figure(figsize=[5, 5])
plt.plot(BF, val, marker='.')
plt.plot(BF, training, marker='.')
plt.figtext(.6, .6, "λ=0.5, η=0.01")
plt.ylim(3, 5)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()



val1=[0.64513,0.64766,0.69027,0.69538]
train1=[0.65526,0.66591,0.69316,0.69375]


plt.figure(figsize=[5, 5])
plt.plot(BF, val1, marker='.')
plt.plot(BF, train1, marker='.')
plt.figtext(.6, .6, "λ=1, η=0.01")
plt.ylim(0.6, 0.8)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()


val2=[0.5414,0.53847,0.53842,0.53782]
train2=[0.55324,0.54963,0.54441,0.54293]








plt.figure(figsize=[5, 5])
plt.plot(BF, val2, marker='.')
plt.plot(BF, train2, marker='.')
plt.figtext(.6, .6, "λ=2, η=0.01")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()



val3=[0.54593,0.53842,0.53789,0.53735]
train3=[0.55766,0.5496,0.54429,0.54284]





plt.figure(figsize=[5, 5])
plt.plot(BF, val3, marker='.')
plt.plot(BF, train3, marker='.')
plt.figtext(.6, .6, "λ=3, η=0.01")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()







val3=[0.55305,0.54892,0.54605,0.54714]
train3=[0.56407,0.55977,0.55563,0.55301]



plt.figure(figsize=[5, 5])
plt.plot(BF, val3, marker='.')
plt.plot(BF, train3, marker='.')
plt.figtext(.6, .6, "λ=4, η=0.01")
plt.ylim(0.5, 0.6)
plt.xlabel('Number of Basis Functions')
plt.ylabel('Root mean square Error')
plt.show()






