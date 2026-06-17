import numpy as np

##Basics

a = np.array([1,2,3,4,5])

print(a)
print(type(a)) #<class 'numpy.ndarray>
print(a.dtype) #int64
print(a.shape) #(5,)- size and dimensions in tuple
print(a.ndim) #1 - no. of dimensions
print(a.size) #5 - elements in tuple

#1D - vector
v = np.array([1,2,4])
print(v.shape)
#2D - matrix
m = np.array([[1,2,3],[2,3,4]])
print(m.shape)
#3D - matrix batch
t = np.array([[[1,2],[2,4]],[[5,6],[3,6]]])
print(t.shape)
#Shapes are imp. in DL

#Specific type force
c = np.array([1,2,4],dtype = np.float32)
print(c.dtype)

#Shortcut Creations
np.zeros((3,4)) #3x4 - zero matrix
np.ones((2,3)) #2x3 - ones matrix
np.eye(3) #3x3 identity matrix
np.arange(0,10,2) #vetor start from 0 and end in 10 with steps of 2
np.linspace(0,1,5) #5 evenly spaced points from 0 to 1
np.random.randn(3,3) #3x3 matrix standard normal dist. (0,1)
np.random.rand(3,3) #3x3 uniform (0,1)
np.full((2,3),7) #2x3 matrix filled with 7

#Exercises
#1
z = np.zeros((4,3))
print(z.shape," ",z.dtype," ",z.size)

#2
u = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(u[1][2])

#3
s = np.linspace(0,1,8)
print(s[3])

#4
r = np.array(np.random.randn(3,3),dtype = np.float32)
print(r.shape," ",r.dtype)

#5
l = np.arange(1,10,3)
print(l)
