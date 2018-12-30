
# coding: utf-8

# In[1]:


import numpy as np #NumPy is a package in Python used for Scientific Computing. NumPy package is used to perform different operations. 
import tensorflow as tf #TensorFlow is an open-source software symbolic math library, used for machine learning applications
from tqdm import tqdm_notebook #tqdm provides a native Jupyter widget (compatible with IPython v1-v4 and Jupyter), fully working nested bars and color hints 
import pandas as pd #pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. 
from keras.utils import np_utils #Keras is an open source neural network library written in Python, which is capable of running on top of TensorFlow. It’s the go-to library for building neural networks with ease.
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Logic Based FizzBuzz Function [Software 1.0]

# In[2]:


def fizzbuzz(n): # Funtion definition for Logic based FizzBuzz implementation, which returns FizzBuzz when a number divisible by both 3 and 5, Fizz when divisible by 3, Buzz when divisible by 5 and others
    
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format.
 

# In[3]:

#Creates training and testing data in csv format For training start=101 and end=1001 and for testing start=1 and end=101
def createInputCSV(start,end,filename): 
    
    # Why list in Python? - Python list takes items which need not be of the same type. The list is a most versatile datatype available in python which can be written as list of comma -seperated items.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data? - Training data is used for learning, to fit the parameters of eg: classifier. Training data is labeled data that we need to train our models.
    for i in range(start,end): # Generating the input and output data i.e input data consist of numbers eg:101....1000 and output consist of respective labels(Fizz, buzz,FizzBuzz)
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe? -Dataframe is Pandas library, and they are defined as a two-dimensional labeled data structures with columns of potentially different types.
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv- Dataframe is called for dataset to write into csv file.
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[4]:


def processData(dataset):
    
    # Why do we have to process? -The data which is chosen may not be in a format that is suitable for us to work with. Here we are processing our decimal and strings to binary.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data) # we are processing the data to convert it to binary which is suitable to work with.
    processedLabel = encodeLabel(labels) # we are converting labels to binary form
    
    return processedData, processedLabel


# In[5]:

def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10? - To convert each input into a vector of activation points. This is done by converting into binary. For each data instance we vector which is of binary form
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData) #returns a vector eg: [1 0 0 0 0 1 0 0 1 1 ]


# In[6]:


def encodeLabel(labels): # Mapping the input to respective labels- Fizz Buzz FizzBuzz others.
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])
#np_utils.to_categorical converts a class vector (integers) to binary class matrix.
# matrix like [0, 0, 0,1], [1, 0, 0, 0],[0, 0, 1, 0],[1, 0, 0, 0] for FizzBuzz, Fizz, Buzz and others
    return np_utils.to_categorical(np.array(processedLabel),4) 

       
    


# In[7]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# In[8]:


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset- Dataset is processed in the format that is suitable to workm with i.e binary
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)


# ## Tensorflow Model Definition

# In[9]:


# Defining Placeholder- A placeholder is simply a variable that we will assign data to at a later date.
#Which is used to feed data into the graph through these placeholders later.
inputTensor  = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])


# In[10]:

NUM_HIDDEN_NEURONS_LAYER_1 = 100  #Increasing the hidden layers will reduce the error and can increase accuracy. Small number of hidden layers may lead to underfitting.

LEARNING_RATE = 0.05 #Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. 
                     #The learning rate defines how quickly a network updates its parameters. 

# Initializing the weights to Normal Distribution
#tf.random_normal is a tensor of the specified shape filled with random normal values.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01)) #weights are randomly initialized

# Initializing the input to hidden layer weights- 
input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1]) #The input_hidden_weights is simply weighted and summed.
# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])
#weights are randomly initialized for shape (100,4) and standard deviation 0.01
#hidden nodes  only receive input from the other nodes, such as input or preceding hidden nodes and they only output to other nodes, either as output or other, following hidden nodes. 


# Computing values at the hidden layer
#Matrix multiplication is carried for inputTensor and input_hidden_weights and activation function is applied which is relu here.
#Activation function calculates the “weighted sum” of its input, adds a bias and then decides whether it should be kept or ignored.  
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

# Computing values at the output layer
#output_layer is obtained by performing matrix multiplications of hidden_layer and hidden_output_weights
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
# Error functions represents the inaccuracy of predictions in this implementation.
#here the softmax_cross_entropy_with_logits is applied to outputTensor 
#computes the cross entropy of the result after applying the softmax function
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
#Optimization methods are used to find the values of parameters of a function that minimizes a cost function. We are using GradientDescentOptimizer
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
#Returns the index with the largest value across axes of a tensor and type of output_layer
prediction = tf.argmax(output_layer, 1)


# # Training the Model

# In[11]:

#Number of epochs is the number of times the whole training data is shown to the network while training.  
NUM_OF_EPOCHS = 5000
#number of sub samples given to the network 
BATCH_SIZE = 128

training_accuracy = []
#A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines)
#for that and holds the actual values of intermediate results and variables.
with tf.Session() as sess:
    
    # Set Global Variables ?-All variables are initialized at once using the tf.global_variables_initializer(). 
    #This op must be ran after the model being fully constructed
    tf.global_variables_initializer().run()
    
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        #np.random.permutation randomly permute a sequence, or return a permuted range.
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData  = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
           # sess.run returns Tensors from requested fetches and outputs 
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch-trainning accuracy is calculated for each epoch
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
    # Testing--returns the tensors requested 
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})


# In[12]:

#Dataframe is defined as a two-dimensional labeled data structures with columns of potentially different types.
    
df = pd.DataFrame()
df['acc'] = training_accuracy
df.plot(grid=True) #for plotting the graph for training_accuracy for given number of epochs


# In[13]:

#decoding the binary form of encoded data to decimal form which is suitable to for calculations further
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# # Testing the Model [Software 2.0]

# In[16]:


wrong   = 0
right   = 0

predictedTestLabelList = []

#zip will map the similar index of multiple containers so that they can be used just using as single entity.
for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:  #np.argmax returns the maximum value along a given axis.
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100)) #calculating the accuracy 

# Please input your UBID and personNumber 
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()
#input ubid and person number in csv file
testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "snehamup")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50288710")

predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList
#predicted data is  written into csv file
opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

