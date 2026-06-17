import numpy as np

##Vectorization

import time
a = np.random.randn(1_000_000)
b = np.random.randn(1_000_000)

#Vectorized 
start = time.time()
result = a*b #fast af!!
print(f"Vector:{time.time()-start:.3f}s")

#Basic Ops.
a = np.array([1.0,2.0,3.0,4.0])
b = np.array([10.0,20.0,30.0, 40.0])
print(a+b)
print(a * b)
print(a**2)
print(a/b)

#Aggregations
data = np.array([1.0,2.0,3.0,4.0,5.0])
total = np.sum(data) #scalar sum
cumulative = np.cumsum(data) #cumulative summation
relu = np.maximum(data,0) #ReLU vectorized
dot = np.dot(a,b) #Dot Product , can also do sum(a*b)

#DL funcs
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum()

def mse_loss(predictions,targets):
    return np.mean((predictions-targets)**2)
def cross_entropy(probs, target_class):
    return -np.log(probs[target_class] + 1e-8) #1e-8 to prevent log(0)

#Exercises
#1
def normalize(x):
    return (x-x.min())/(x.max()-x.min())
s = np.array([2.0,5.0,8.0,14.0,20.0])
print(normalize(s))

#2
def sigmoid(x):
    return 1/(1+np.exp(-x))
s = np.array([-3.0,-1.0,0.0,1.0,3.0])
print(sigmoid(s))

#3
def softmax(x):
    e_x = np.exp(x-x.max())
    return e_x/(e_x.sum())
s = np.array([2.0,1.0,0.1])
print(softmax(s))

#4
a = np.random.randn(1_000_000)
b = np.random.randn(1_000_000)

start = time.time()
result = []
for i in range(1_000_000):
    result.append(a[i]*b[i])
print(f"Loop:{time.time()-start:.4f}") #Loop

start = time.time()
result = a*b
print(f"Vector:{time.time()-start:.4f}") #Vectorized

#5
def mse_loss(predictions, targets):
    return np.mean((predictions-targets)**2)
predictions = np.array([0.9, 0.2, 0.8, 0.4, 0.6])
targets     = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
print(mse_loss(predictions,targets))

#6
x = np.array([1.0,2.0,3.0,4.0])
print(np.exp(x)/np.sum(np.exp(x)))