import numpy as np
import random
import matplotlib.pyplot as plt
from sympy import numbered_symbols #importing painter
#library with prepared datasets
#Database with Irises
from sklearn import datasets

#Neural Network Setup
INPUT_DIMENSIONS = 4 
OUTPUT_DIMENSIONS = 3
FIRST_LAYERS = 10 #Amount of Inner Layers 
BATCH_SIZE = 5 #Amount of values in Batch 

#Learning Setup 
LEARNING_RATE = 0.0002
EPOCHS = 400
loss_arr = []

#Setting up the Dataset
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...],iris.target[i]) for i in range(len(iris.target))]

w1 = np.random.randn(INPUT_DIMENSIONS, FIRST_LAYERS)
b1 = np.random.randn(1, FIRST_LAYERS)
w2 = np.random.randn(FIRST_LAYERS,OUTPUT_DIMENSIONS)
b2 = np.random.randn(1, OUTPUT_DIMENSIONS)

#making initial weights a little bit stranger 
w1 = (w1 - 0.5) * 2 * np.sqrt(1/INPUT_DIMENSIONS)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIMENSIONS)
w2 = (w2 - 0.5) * 2 * np.sqrt(1/FIRST_LAYERS)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/FIRST_LAYERS)

#Activator operation
def relu(input_):
    return np.maximum(input_,0)

#A derivative taken from Relu
def relu_deriv(input_):
    return (input_ >=0).astype(float)

def softmax(input_):
    output_ = np.exp(input_)
    return output_/np.sum(output_)

#A version of Softmax function, but to array of values
def softmax_batch(input_arr):
    output_ = np.exp(input_arr)
    return output_ / np.sum(output_, axis=1, keepdims=True)

#Prodecure to get an error rate
def sparse_cross_entropy(z, input_):
    return -np.log(z[0, input_])

def sparse_cross_entropy_batch(z, input_):
    return -np.log(np.array([z[j, input_[j]] for j in range(len(input_))]))

#One hot encoding
def to_full(input_, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0,input_] = 1
    return y_full

#one hot enconding, but with batch of right answers
def to_full_batch(input_y, num_classes):
    y_full = np.zeros((len(input_y),num_classes))
    for j, yj in enumerate(input_y):
        y_full[j,yj] = 1
    return y_full

#Prediction (to compare with current training result)
def predict(input_):
    t1 =  input_ @ w1 + b1
    h1 = relu(t1)
    t2 =  h1 @ w2 + b2
    h2 = relu(t2)
    output_ = softmax(t2)
    return output_

#Calculate Accuracy of prediction
def calc_accuracy():
    correct = 0 
    for x,y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred==y:
            correct += 1
    acc = correct / len(dataset)
    return acc

#A loss graph painter
def painter(input_1):
    plt.plot(input_1)
    plt.show()

#Training Circle
for ep in range(EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset)//BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]) #zip makes a = (0,1,2); b = (3,4,5) to c = ((0,3),(1,4),(2,5))
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        #Forward 
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        #Backward 
        y_full = to_full_batch(y, OUTPUT_DIMENSIONS)
        dE_dt2 = z -y_full
        dE_dw2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims = True)
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dw1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims = True)

        #Update  
        w1 -= LEARNING_RATE * dE_dw1
        b1 -= LEARNING_RATE * dE_db1
        w2 -= LEARNING_RATE * dE_dw2
        b2 -= LEARNING_RATE * dE_db2

        #Loss Rate
        loss_arr.append(E)
        
accuracy = calc_accuracy()
print("Accuracy: ", accuracy)
painter(loss_arr)