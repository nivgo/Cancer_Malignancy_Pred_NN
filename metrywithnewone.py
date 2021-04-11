
# # Predicting Cancer Malignancy with a 2 layer neural network coded from scratch in Python.

# **This notebook holds the Python code connected to this 3 part article:**
# 
# **<a href="https://towardsdatascience.com/the-keys-of-deep-learning-in-100-lines-of-code-907398c76504" target="_blank">Part 1</a> | <a href="https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2" target="_blank">Part 2</a> | <a href="https://medium.com/predicting-breast-cancer-tumors-with-your-own-neural-network-76271a05e941" target="_blank">Part 3</a>**<br>
# 
# **With this code and the associated articles, you are going to:**
# - Create a neural network from scratch in Python. Train it using the gradient descent algorithm.
# - Apply that basic network to The Wisconsin Cancer Data-set. Predict if a tumor is benign or malignant, based on 9 different features.
# - Explore deeply how back-propagation and gradient descent work.
# - Review the basics and explore advanced concepts. 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2

np.set_printoptions(threshold=np.inf)

def plotCf(a,b,t):
    cf =confusion_matrix(a,b)
    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title(t)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(a))) # length of classes
    class_labels = ['0','1']
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
    plt.show();


# **The Dlnet 2 layer neural network class**
# 
# A 2 layer neural network class with gradient descent in less than 100 lines of code

# In[3]:


def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def Relu(Z):
    return np.maximum(0,Z)

def dRelu2(dZ, Z):    
    dZ[Z <= 0] = 0    
    return dZ

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

class dlnet:
    def __init__(self, x, y):
        self.debug = 0;
        self.X = x
        self.Y = y
        self.Yh=np.zeros((1,self.Y.shape[1])) 
        self.L=2
        self.dims = [1024, 40, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.acc = []
        self.lr = 0.003
        self.sam = self.Y.shape[1]
        self.threshold = 0.5

        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return 

    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = Relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2

        self.Yh=A2
        loss=self.nloss(A2)
        return self.Yh, loss

    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        
        return


    def pred(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, loss= self.forward()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0

        print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
        
        return comp
    
    def gd(self,X, Y, iter = 10000):
        np.random.seed(1)
        epochforplot = []
        self.nInit()
        x = X
        y = Y
    
        for i in range(0, iter):
            Yh, loss = self.forward()
            self.backward()
        
            if i % 200 == 0:
                print("Cost after iteration %i: %f" %(i, loss))
                self.loss.append(loss)
                epochforplot.append(i)
                comp = np.zeros((1, x.shape[1]))
                pred, loss = self.forward()

                for k in range(0, pred.shape[1]):
                    if pred[0, k] > self.threshold:
                        comp[0, k] = 1
                    else:
                        comp[0, k] = 0

                self.acc.append(float(np.sum((comp == y) / x.shape[1])))
                print("Accuracy: " + str(np.sum((comp == y) / x.shape[1])))




        fig, axs = plt.subplots(1,2)
        axs[0].plot(epochforplot, np.squeeze(self.loss))
        axs[1].plot(epochforplot, np.round(np.squeeze(self.acc), 2))


        axs[0].set_title("Loss vs Epochs")
        axs[1].set_title("Accuracy vs Epochs")
        axs[0].set(xlabel='epoch', ylabel='loss')
        axs[1].set(xlabel='epoch', ylabel='accuracy')
        plt.show()
        plt.close(fig)
        return 

###########Import Pictures and Arranging In Victor################
directory = 'training'
entries = os.listdir(directory)

training_label_vector = np.zeros(shape = (len(entries), 1), dtype = int)
training_data_mat = np.zeros(((1024,len(entries))))


place = 0
for entry in entries:
    img = cv2.imread(directory + '/' + str(entry))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vector = img.ravel()
    training_data_mat[:,place] = img_vector
    if entry[0] == 'n':
        training_label_vector[place] = 0
    if entry[0] == 'p':
        training_label_vector[place] = 1
    place += 1

###########Import Pictures and Arranging In Victor################
directory = 'validation'
entries = os.listdir(directory)

validation_label_vector = np.zeros(shape = (len(entries), 1), dtype = int)
validation_data_mat = np.zeros(((1024,len(entries))))


place = 0
for entry in entries:
    img = cv2.imread(directory + '/' + str(entry))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vector = img.ravel()
    validation_data_mat[:,place] = img_vector
    if entry[0] == 'n':
        validation_label_vector[place] = 0
    if entry[0] == 'p':
        validation_label_vector[place] = 1
    place += 1

training_data_mat = training_data_mat.T/255.
validation_data_mat = validation_data_mat.T/255.
''''''''''''''''''''''''

#split data into train and test set
training_data_mat = training_data_mat.T
validation_data_mat = validation_data_mat.T
training_label_vector = training_label_vector.T
validation_label_vector = validation_label_vector.T

print( training_data_mat.shape, training_label_vector.shape, validation_data_mat.shape, validation_label_vector.shape)

nn = dlnet(training_data_mat, training_label_vector)
nn.lr=0.05
nn.dims = [1024, 20, 1]

nn.gd(training_data_mat, training_label_vector, iter = 2000)



pred_train = nn.pred(training_data_mat, training_label_vector)
pred_validation = nn.pred(validation_data_mat, validation_label_vector)




nn.threshold=0.5

nn.X,nn.Y= training_data_mat, training_label_vector
target=np.around(np.squeeze(training_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(training_data_mat, training_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Training Set')

nn.X,nn.Y= validation_data_mat, validation_label_vector
target=np.around(np.squeeze(validation_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(validation_data_mat, validation_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Validation Set')




nn.threshold=0.7

nn.X,nn.Y= training_data_mat, training_label_vector
target=np.around(np.squeeze(training_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(training_data_mat, training_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Training Set')

nn.X,nn.Y= validation_data_mat, validation_label_vector
target=np.around(np.squeeze(validation_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(validation_data_mat, validation_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Validation Set')




nn.threshold=0.9

nn.X,nn.Y= training_data_mat, training_label_vector
target=np.around(np.squeeze(training_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(training_data_mat, training_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Training Set')

nn.X,nn.Y= validation_data_mat, validation_label_vector
target=np.around(np.squeeze(validation_label_vector), decimals=0).astype(np.int)
predicted=np.around(np.squeeze(nn.pred(validation_data_mat, validation_label_vector)), decimals=0).astype(np.int)
plotCf(target,predicted,'Cf Validation Set')




nn.X,nn.Y= validation_data_mat, validation_label_vector
yvalh, loss = nn.forward()
print("\ny", np.around(validation_label_vector[:, 0:50, ], decimals=0).astype(np.int))
print("\nyh",np.around(yvalh[:,0:50,], decimals=0).astype(np.int),"\n")         

