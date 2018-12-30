#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:29:13 2018

@author: snehamuppala
"""



from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import scipy.sparse

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Extract feature values and labels from the data
mnist_train_labels = np.array(mnist.train.labels)
mnist_train_images =  np.array(mnist.train.images)
mnist_valid_images =  np.array(mnist.validation.images)
mnist_valid_labels =  np.array(mnist.validation.labels)
mnist_test_labels =  np.array(mnist.test.labels)
mnist_test_images =  np.array(mnist.test.images)
example = mnist_train_images[100]
mnist_train_images.shape
#plt.imshow(np.reshape(example,[28,28]))

# ...
# Here we determine the probabilities and predictions for each class when given a set of input data
# ...

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

# ...
# Here we perform the softmax transformation: This allows us to get probabilities for each class score that sum to 100%.
# ...
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
#     print("Returning softmax : ", sm)
    return sm



def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples    
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad


x = mnist_train_images
y = mnist_train_labels
w = np.zeros([x.shape[1], 10])
lam = 1
iterations = 100
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x,y,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    #print("Loss for iteration : ", i, "is : ", loss)

plt.plot(losses)
plt.show()
classWeightsToVisualize = 8

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    c = 0;
    for i in range(len(someY)):
        test = someY[i, prede[i]]
        if  test == 1:
            c+=1
    accuracy = c/(float(len(someY)))        
    return accuracy
print ('Training Accuracy: ', getAccuracy(x,y))
testX = mnist_test_images
testY = mnist_test_labels
print ('Test Accuracy: ', getAccuracy(testX,testY))



def get_my_usps_data(): 
    import zipfile
    import os
    from PIL import Image
    import PIL.ImageOps  
    import numpy as np
    import tensorflow  as tf
    import matplotlib.pyplot as plt

    filename="usps_dataset_handwritten.zip"

    #Defining height,width for resizing the images to 28x28 like MNIST digits
    height=28
    width=28

    #Defining path for extracting dataset zip file
    extract_path = "usps_data"

    #Defining image,label list
    images = []
    img_list = []
    labels = []

    #Extracting given dataset file    
    with zipfile.ZipFile(filename, 'r') as zip:
        zip.extractall(extract_path)

    #Extracting labels,images array needed for training    
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)

        if "Numerals" in path:
            image_files = [fname for fname in files if fname.find(".png") >= 0]
            for file in image_files:
                labels.append(int(path[-1]))
                images.append(os.path.join(*path, file)) 

    #Resizing images like MNIST dataset   
    for idx, imgs in enumerate(images):
        img = Image.open(imgs).convert('L') 
        img = img.resize((height, width), Image.ANTIALIAS)
        img_data = list(img.getdata())
        img_list.append(img_data)

    #Storing image and labels in arrays to be used for training   
    USPS_img_array = np.array(img_list)
    USPS_img_array = np.subtract(255, USPS_img_array)
    USPS_label_array = np.array(labels)
    #print(USPS_label_array.shape)
    nb_classes = 10
    targets = np.array(USPS_label_array).reshape(-1)
    aa = np.eye(nb_classes)[targets]
    USPS_label_array = np.array(aa, dtype=np.int32)
    #print(USPS_label_array)


    USPS_img_array = np.float_(np.array(USPS_img_array))
    for z in range(len(USPS_img_array)):
        USPS_img_array[z] /= 255.0 

   
    
    
    return USPS_img_array, USPS_label_array

USPS_img_array, USPS_label_array = get_my_usps_data()








import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes labels

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax predicted values

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        #if (epoch+1) % display_step == 0:
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ('UBITname      = snehamup')
    print ('Person Number = 50288710')
    print("1. Logistic Regression")
    print("Logistic Regression Training accuracy of MNIST :", accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))
    print("Logistic Regression VAlidation accuracy of MNIST:", accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels}))
    print("Logistic Regression TESTING accuracy of MNIST:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Logistic Regression TESTING accuracy of USPS:", accuracy.eval({x: np.float_(USPS_img_array) , y: np.float_(USPS_label_array)}))




data = MNIST('./MNIST_Dataset_Loader/dataset/')

#print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

#print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)  
    
p1,y_pred=getProbsAndPreds(test_img)
#print(y_pred)
#print(mnist.test.labels)
conf_mat = confusion_matrix(test_labels,y_pred)



print('\nLogistic Regression:Confusion Matrix for Test Data MNIST: \n',conf_mat)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat)
plt.title('Confusion Matrix for MNIST Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()



from PIL import Image
import os
import numpy as np


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


USPS_img_array1=np.array(USPSMat)
USPS_label_array1=np.array(USPSTar)

p1,y_pred_usps=getProbsAndPreds(USPS_img_array1)
conf_mat = confusion_matrix(USPS_label_array1,y_pred_usps)



print('\nLogistic Regression :Confusion Matrix for Test Data USPS: \n',conf_mat)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat)
plt.title('Confusion Matrix for USPS Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()
   







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:11:36 2018

@author: snehamuppala
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#%matplotlib inline  
print ("PACKAGES LOADED")
mnist = input_data.read_data_sets('data/', one_hot=True)

# NETWORK TOPOLOGIES
n_hidden_1 = 256 
n_hidden_2 = 128 
n_input    = 784 
n_classes  = 10  

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# NETWORK PARAMETERS
stddev = 0.1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#print ("NETWORK READY")
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])


# PREDICTION
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
cost = tf.nn.softmax_cross_entropy_with_logits( logits=pred, labels=y)
# optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost) 
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))


# INITIALIZER
init = tf.initialize_all_variables()
#print ("FUNCTIONS READY")

# PARAMETERS
training_epochs = 20
batch_size      = 100
display_step    = 4
# LAUNCH THE GRAPH

sess = tf.Session()
sess.run(init)

# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY
    if (epoch+1) % display_step == 0:
        #print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        #print ("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds1 = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds1)
        #print ("TEST ACCURACY: %.3f" % (test_acc))
        feeds={x:USPS_img_array, y:USPS_label_array}
        
        
        usps_acc = sess.run(accr, feed_dict=feeds)
        #print ("USPS ACCURACY: %.3f" % (usps_acc))

print("2. Multilayer Perceptron Neural Network ")
print("Training_epochs: 20")
print("Batch_size: 100")
print("learning_rate: 0.001")
print ("Neural Network TRAIN ACCURACY MNIST: %.3f" % (train_acc))  
print ("Neural Network TEST ACCURACY MNIST: %.3f" % (test_acc))  
print ("Neural Network TEST ACCURACY USPS: %.3f" % (usps_acc))  

# Random Forest Classifier

#import sys
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST



import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



#old_stdout = sys.stdout
#log_file = open("summary.log","w") 
#sys.stdout = log_file




data = MNIST('./MNIST_Dataset_Loader/dataset/')

#print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

#print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)










clf = RandomForestClassifier(n_estimators=10, n_jobs=10)
clf.fit(X_train,y_train)

with open('MNIST_RFC.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_RFC.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X_test,y_test)

y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


conf_mat = confusion_matrix(y_test,y_pred)

print('\n3.Random Forest Classifier with n_estimators = 100, n_jobs = 10')

print('\nMNIST RFC Trained Classifier Accuracy : ',confidence)
#print('\nPredicted Values: ',y_pred)
print('\nMNIST RFC Accuracy of Classifier on validation Image Data MNIST:',accuracy)

print('\nRandom Forest ClassifierConfusion Matrix: \n',conf_mat)
# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title('Confusion Matrix for MNIST Testing Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




test_labels_pred = clf.predict(test_img)



acc = accuracy_score(test_labels,test_labels_pred)




conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

print('\n 3.Random Forest Classifier with n_estimators = 10, n_jobs = 10')
print('\nRFC Trained Classifier Accuracy MNIST: ',confidence)
print('\nRFC Accuracy of Classifier on Testing Image Data MNIST:',acc)


print('\nConfusion Matrix for Test Data USPS: \n',conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data USPS')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()



from PIL import Image
import os
import numpy as np


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


USPS_img_array=np.array(USPSMat)
USPS_label_array=np.array(USPSTar)



test_labels_pred_usps = clf.predict(USPS_img_array)
#test_labels_pred_usps1 = clf.predict(USPSMat)
acc_usps_1=(test_labels_pred_usps==USPSTar).mean()

acc_usps = accuracy_score(USPS_label_array,test_labels_pred_usps)



print("Accuracy of USPS Testing set",acc_usps)

print('\nAccuracy of Classifier on USPS Test Images: ',acc_usps)
print('\n Creating Confusion Matrix for USPS Test Data...')
conf_mat_test_usps = confusion_matrix(USPS_label_array,test_labels_pred_usps)

print('\nConfusion Matrix for Test Data : \n',conf_mat_test_usps)
plt.matshow(conf_mat_test_usps)
plt.title('Confusion Matrix for Test Data USPS')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()


import sys
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')




from PIL import Image
import os
import numpy as np

USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


USPS_img_array=USPSMat
USPS_label_array=USPSTar




# Load MNIST Data
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)


# Pickle the Classifier for Future Use

print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=1, kernel='poly')
clf.fit(X_train,y_train)

with open('MNIST_SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle','rb')
clf = pickle.load(pickle_in)


acc = clf.score(X_test,y_test)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test,y_pred)
print('\n4. SVM Classifier with gamma = 1; Kernel = polynomial')
print('\nMNIST SVM Trained Classifier Accuracy: ',acc)

print('\nMNIST SVM Accuracy of Classifier on Validation Images: ',accuracy)

test_labels_pred_usps = clf.predict(USPS_img_array)
acc_usps = accuracy_score(USPS_label_array,test_labels_pred_usps)

test_labels_pred = clf.predict(test_img)


acc = accuracy_score(test_labels,test_labels_pred)

print('\nMNIST SVM Accuracy of Classifier on Test Images: ',acc)
print('\nUSPS SVM Accuracy of Classifier on USPS Test Images: ',acc_usps)




print('\n nSVM MNIST Creating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)
print('\nSVM MNIST Confusion Matrix: \n',conf_mat)

# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title(' SVM Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print('\nMNIST Confusion Matrix for Test Data: \n',conf_mat_test)
# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('SVM MNIST Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

test_labels_pred_usps = clf.predict(USPS_img_array)
acc_usps = accuracy_score(USPS_label_array,test_labels_pred_usps)


print('\n USPS Creating Confusion Matrix for Test Data...')
conf_mat_test1 = confusion_matrix(USPS_label_array,test_labels_pred_usps)

print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('\nAccuracy of Classifier on USPS Test Images: ',acc_usps)
print('\nAccuracy of Classifier on Test Images: ',acc)
print('\nUSPS Confusion Matrix for Test Data: \n',conf_mat_test1)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test1)
plt.title('USPS Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()


import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test1=x_test
y_test1=y_test
num_classes=10
image_vector_size=28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

image_size = 784 
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False,validation_split=.2)
loss,accuracy = model.evaluate(x_test, y_test, verbose=False)
print(accuracy)
img_list=[]
y_predcm = model.predict(x_test)
for i in range(10000):
    #print(np.argmax(y_predcm[i]))
    img_list.append(np.argmax(y_predcm[i]))

img_list = np.array(img_list, dtype=np.uint8)
#print(img_list)

conf_mat_test = confusion_matrix(y_test1,img_list)

print('\nConfusion Matrix for Test MNIST Data: \n',conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test MNIST Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()



from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection, svm, preprocessing
from scipy.stats import mode

from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.linear_model import LogisticRegression
from MNIST_Dataset_Loader.mnist_loader import MNIST
import numpy as np
from sklearn import model_selection



data = MNIST('./MNIST_Dataset_Loader/dataset/')

#print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

#print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
x = train_img

#Labels
y = train_labels
import random





x_train=train_img
x_test=test_img
y_train=train_labels
y_test=test_labels


print("RF")
model1 = RandomForestClassifier(n_estimators=5,n_jobs=10)
print("svm")
model2 = svm.SVC(gamma=1, kernel='poly',probability = True)
#model3= LogisticRegression()

model1.fit(x_train,y_train)

model2.fit(x_train,y_train)
#model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)

pred2=model2.predict_proba(x_test)
#pred3=model3.predict_proba(x_test)


def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

y=[]

pred3=np.concatenate((img_list, y_pred,pred1,pred2), axis=None)
pred3 = np.array(pred3, dtype=np.uint8)
for i in range (0,len(x_test)):
    y.append(mode1(pred3[i]))
    
y = np.array(y, dtype=np.uint8)  

accuracy = accuracy_score(y_test1, y)
print("Majority Voting Accurancy",accuracy)




