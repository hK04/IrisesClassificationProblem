from pydoc import classname
import numpy as np 

INPUT_DIMENSIONS = 4 
OUTPUT_DIMENSIONS = 3
FIRST_LAYERS = 10

CLASSNAMES = ['Setosa', 'Versicolor', 'Virginica']

x = np.random.rand(INPUT_DIMENSIONS)
W1 = np.random.rand(INPUT_DIMENSIONS, FIRST_LAYERS)
b1 = np.random.rand(FIRST_LAYERS)
W2 = np.random.rand(FIRST_LAYERS, OUTPUT_DIMENSIONS)
b2 = np.random.rand(OUTPUT_DIMENSIONS)

#activator operation
def relu(input_):
    return np.maximum(input_,0)

def softmax(input_):
    output_ = np.exp(input_)
    return output_/np.sum(input_)
#main function
def predict(input_):
    t1 =  input_ @ W1 + b1
    h1 = relu(t1)
    t2 =  h1 @ W2 + b2
    h2 = relu(2)
    output_ = softmax(t2)
    return output_

results = predict(x)
pred_class = np.argmax(results)

print("Predicted class: ", CLASSNAMES[pred_class] )