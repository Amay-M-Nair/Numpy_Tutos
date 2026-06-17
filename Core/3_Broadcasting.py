import numpy as np

##Broadcasting

m = np.array([[1,2,3],[4,5,6]]) #shape(2,3)
print(m+10) #Scalar Broadcasting

v = np.array([10,20,30]) #shape(3,)
print(m+v) #Vector Broadcasting

col = np.array([[1],[2],[3]]) #shape = (1,3)
row = np.array([[10,20,30,40]]) #shape = (4,1)
print(col+row) #shape(3,4)
"""
NumPy compares shapes **from the right**, dimension by dimension. Two dimensions are compatible if:
- they are **equal**, or
- one of them is **1**
If shapes don't have the same number of dimensions, NumPy **pads with 1s on the left** until they match.
here result shape=(2,3) and (3,) = (2,3) and (3,1) = (2,3)
for Bias Addition & Normalization
"""
#Normalization
X = np.random.randn(32, 128)
mean = X.mean(axis=0)   # shape (128,) — mean of each feature across batch
std  = X.std(axis=0)    # shape (128,)

X_norm = (X - mean) / std   # both broadcast across batch
                             # shape: (32, 128)

#Reshape Trick
a = np.array([1,2,3]) #shape (3,)
b = np.array([10,20,30]) #shape (3,)
#not compatible
print(a[:,None]+b) #Added fake(None) dimension to get element wise add(outer product), shape=(3,1)+(3,) -> (3,3)
#[:, None] and [None, :] constantly to control which axes broadcast

#Exercises

v = np.array([1,2,3,4])
m = np.ones((4,4)) 
print(m+v) #1
print(m + v[:,None]) #2
#3
X = np.array([[1.0, 2.0, 4.0],
              [2.0, 4.0, 8.0],
              [3.0, 6.0, 12.0]])
x = X.mean(axis=0)
s = X.std(axis = 0)
normX = (X-x)/s
print(normX)

#4
a = np.ones((5,1)) #Shape(5,1)
b = np.ones((1,3)) #Shape(1,3)
print((a+b).shape) #Shape(5,3)

#5
X = np.array([[2.0, 4.0, 6.0, 8.0],
              [1.0, 3.0, 5.0, 7.0],
              [3.0, 5.0, 7.0, 9.0]])  # shape (3, 4)
w = np.array([0.1, 0.2, 0.3, 0.4])   # shape (4,)
print(X*w) #shape(3,4) by padding of w 4 times
