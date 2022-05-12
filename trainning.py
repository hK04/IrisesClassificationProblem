import numpy as np
#importing painter
import matplotlib.pyplot as plt
#library with prepared datasets
#Database with Irises
from sklearn import datasets

#Neural Network Setup
INPUT_DIMENSIONS = 4 
OUTPUT_DIMENSIONS = 3
FIRST_LAYERS = 5

#Learning Setup 
LEARNING_RATE = 0.01
EPOCHS = 100
Counter = 0
loss_arr = []
id__of_loss_arr = []

#Setting up the Dataset
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...],iris.target[i]) for i in range(len(iris.target))]

w1 = np.random.randn(INPUT_DIMENSIONS, FIRST_LAYERS)
b1 = np.random.randn(1, FIRST_LAYERS)
w2 = np.random.randn(FIRST_LAYERS,OUTPUT_DIMENSIONS)
b2 = np.random.randn(1, OUTPUT_DIMENSIONS)

#Activator operation
def relu(input_):
    return np.maximum(input_,0)

#A derivative taken from Relu
def relu_deriv(input_):
    return (input_ >=0).astype(float)

def softmax(input_):
    output_ = np.exp(input_)
    return output_/np.sum(output_)

#Prodecure to get an error rate
def sparse_cross_entropy(z, input_):
    return -np.log(z[0, input_])

#One hot encoding
def to_full(input_, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0,input_] = 1
    return y_full

#Prediction (to compare with current training result)
def predict(input_):
    t1 =  input_ @ w1 + b1
    h1 = relu(t1)
    t2 =  h1 @ w2 + b2
    h2 = relu(2)
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

#Training Circle
for ep in range(EPOCHS):
    for i in range(len(dataset)):
        x, y = dataset[i]
        #Forward 
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)

        #Backward 
        y_full = to_full(y, OUTPUT_DIMENSIONS)
        dE_dt2 = z -y_full
        dE_dw2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dw1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        #Update  
        w1 -= LEARNING_RATE * dE_dw1
        b1 -= LEARNING_RATE * dE_db1
        w2 -= LEARNING_RATE * dE_dw2
        b2 -= LEARNING_RATE * dE_db2

        #Loss Rate

        loss_arr.append(E)
        Counter += 1
        id__of_loss_arr.append(Counter)

def painter(input_1):
    plt.plot(input_1)
    plt.show()

accuracy = calc_accuracy()
print("Accuracy: ", accuracy)
painter(loss_arr)