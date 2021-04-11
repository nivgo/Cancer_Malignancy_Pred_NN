import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
from sklearn.metrics import confusion_matrix
import itertools

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


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Relu(Z):
    return np.maximum(0, Z)

def dRelu2(dZ, Z):
    dZ[Z <= 0] = 0
    return dZ

def dRelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ


class dlnet:
    def __init__(self, x, y, x_val, y_val):
        self.debug = 0;
        self.X = x
        self.Y = y
        self.Xval = x_val
        self.Yval = y_val
        self.Yh = np.zeros((1, self.Y.shape[1]))
        self.Yhval = np.zeros((1, self.Y.shape[1]))
        self.L = 2
        self.dims = [1024, 40, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.loss_val = []
        self.acc = []
        self.acc_val = []
        self.lr = 0.003
        self.sam = self.Y.shape[1]
        self.samval = self.Yval.shape[1]
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
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2

        self.Yh = A2
        loss = self.nloss(A2)
        return self.Yh, loss

    def forward_val(self):
        Z1 = self.param['W1'].dot(self.Xval) + self.param['b1']
        A1 = Relu(Z1)

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)

        self.Yhval = A2
        loss_val = self.nlossval(A2)
        return self.Yhval, loss_val

    def nloss(self, Yh):
        loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
        return loss
    def nlossval(self, Yh):
        loss = (1. / self.samval) * (-np.dot(self.Yval, np.log(Yh).T) - np.dot(1 - self.Yval, np.log(1 - Yh).T))
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh) - np.divide(1 - self.Y, 1 - self.Yh))

        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
        dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
        dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
        dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))

        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
        dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
        dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
        dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

        return

    def pred(self, x, y):
        self.X = x
        self.Y = y
        comp = np.zeros((1, x.shape[1]))
        pred, loss = self.forward()

        for i in range(0, pred.shape[1]):
            if pred[0, i] > self.threshold:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        print("Acc: " + str(np.sum((comp == y) / x.shape[1])))

        return comp

    def gd(self, X, Y, X_val, Y_val, epochs=10000):
        np.random.seed(1)
        epochforplot = []
        self.nInit()
        x = X
        y = Y
        xval = X_val
        yval = Y_val

        for i in range(0, epochs):
            Yh, loss = self.forward()
            self.backward()

            if i % 200 == 0:
                print("Training Loss after %i percent of epochs: %f " % ((i/epochs)*100, loss))
                self.loss.append(loss)
                epochforplot.append(i)
                comp = np.zeros((1, x.shape[1]))
                compval = np.zeros((1, xval.shape[1]))
                pred, loss = self.forward()
                pred_val, loss_val = self.forward_val()
                self.loss_val.append(loss_val)

                for k in range(0, pred.shape[1]):
                    if pred[0, k] > self.threshold:
                        comp[0, k] = 1
                    else:
                        comp[0, k] = 0
                for k in range(0, pred_val.shape[1]):

                    if pred_val[0, k] > self.threshold:
                        compval[0, k] = 1
                    else:
                        compval[0, k] = 0


                self.acc.append(float(np.sum((comp == y) / x.shape[1])))
                self.acc_val.append(float(np.sum((compval == yval) / xval.shape[1])))
                print("Accuracy training: " + str(np.sum((comp == y) / x.shape[1])))
                print("Accuracy validation: " + str(np.sum((compval == yval) / xval.shape[1])))

        # plt.plot(np.squeeze(self.loss))
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(epochforplot, np.squeeze(self.loss),label="Training")
        axs[0].plot(epochforplot, np.squeeze(self.loss_val), label="Validation")
        leg1 = axs[0].legend(loc="upper right")
        axs[1].plot(epochforplot, np.round(np.squeeze(self.acc), 2),label="Training")
        axs[1].plot(epochforplot, np.round(np.squeeze(self.acc_val), 2),label="Validation")
        leg2 = axs[1].legend(loc="lower right")

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

training_label_vector = np.zeros(shape=(len(entries), 1), dtype=int)
training_data_mat = np.zeros(((1024, len(entries))))

place = 0
for entry in entries:
    img = cv2.imread(directory + '/' + str(entry))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vector = img.ravel()
    training_data_mat[:, place] = img_vector
    if entry[0] == 'n':
        training_label_vector[place] = 0
    if entry[0] == 'p':
        training_label_vector[place] = 1
    place += 1

directory = 'validation'
entries = os.listdir(directory)

validation_label_vector = np.zeros(shape=(len(entries), 1), dtype=int)
validation_data_mat = np.zeros(((1024, len(entries))))

place = 0
for entry in entries:
    img = cv2.imread(directory + '/' + str(entry))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vector = img.ravel()
    validation_data_mat[:, place] = img_vector
    if entry[0] == 'n':
        validation_label_vector[place] = 0
    if entry[0] == 'p':
        validation_label_vector[place] = 1
    place += 1

training_data_mat = training_data_mat.T / 255.
validation_data_mat = validation_data_mat.T / 255.
''''''''''''''''''''''''

# split data into train and test set
training_data_mat = training_data_mat.T
validation_data_mat = validation_data_mat.T
training_label_vector = training_label_vector.T
validation_label_vector = validation_label_vector.T



nn = dlnet(training_data_mat, training_label_vector, validation_data_mat,validation_label_vector)
nn.lr = 0.0005
nn.dims = [1024, 22, 1]

nn.gd(training_data_mat, training_label_vector, validation_data_mat, validation_label_vector, epochs=17000)
print("Training Finished")
pred_train = nn.pred(training_data_mat, training_label_vector)

trained_dict = {'weights': (nn.param['W1'].tolist(), nn.param['W2'].tolist()),
                'biases': (nn.param['b1'].tolist(), nn.param['b2'].tolist()),
                'nn_hdim': nn.dims[1], 'activation_1': 'ReLU', 'activation_2': 'sigmoid', 'IDs': ( 312270028)}
pred_validation = nn.pred(validation_data_mat, validation_label_vector)



#Checking different thresholds

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


























with open('trained_dict.txt', 'w') as outfile:
    json.dump(trained_dict, outfile)
