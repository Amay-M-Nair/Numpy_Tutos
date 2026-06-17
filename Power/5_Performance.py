import numpy as np

#Performance

#View vs Copy

a = np.array([1, 2, 3, 4, 5, 6])

b = a[2:5] #view shares memory
b[0] = 999
print(a)

c = a[2:5].copy() #independent memory
c[0] = 0
print(a) 

a = np.arange(12).reshape(3,4)

b = a[1:3] #slice -> view
c = a[[0,2]] #fancy idx -> copy
d = a.reshape(4,3) #view

print(b.base is a) #True -> view
print(c.base is a) #False -> copy
print(d.base is a)

"""
Views (no copy, fast):

Basic slicing: a[1:3], a[:, 2], a[::2]
reshape (when memory is contiguous)
transpose, swapaxes
squeeze, expand_dims

Always copies (new memory, slower):

Fancy indexing: a[[0,2,4]]
Boolean indexing: a[a > 0]
np.concatenate, np.stack
Operations that produce new values: a + b, a * 2

"""

#Memory Layout

"""
# [[0 1 2]
#  [3 4 5]]

# C order (row-major, default): rows stored contiguously
# Memory: [0, 1, 2, 3, 4, 5]

# Fortran order (column-major): columns stored contiguously  
# Memory: [0, 3, 1, 4, 2, 5]

"""
a = np.random.randn(1000,1000)
row_sums = np.sum(a,axis=1) #fast iteration, contiguous in C order

print(a.flags['C_CONTIGUOUS']) #true -> rows are contiguous
print(a.T.flags['C_CONTIGUOUS']) #false -> transpose breaks contiguity

#Contiguous

a = np.random.randn(4, 6)
b = a.T   #not C-contiguous

print(b.flags['C_CONTIGUOUS'])   # False

b_contig = np.ascontiguousarray(b) #fix
print(b_contig.flags['C_CONTIGUOUS'])   # True

#Profiling
import time

def time_it(fn, n=100):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter()-start)/n*1000 #ms

a = np.random.randn(1000,1000)

print(time_it(lambda:a[500,:].sum())) #fast row ops
print(time_it(lambda:a[:,500].sum())) #slow column ops

#In-place ops

a = np.random.randn(1000,1000)

b=a*2 #new array
a*=2 #inplace modification, saves memory

#Exercise

#1
weights = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])
sub = weights[0:2,:]
sub[0,0]=999
print(weights)

sub[0,0]= 1.0
sub = weights[:2,:].copy()
sub[0, 0]=999
print(weights)

#2
a = np.arange(24).reshape(4, 6)

b = a[1:3, :]
c = a[[0, 2], :]
d = a.reshape(6, 4)
e = a.T
f = a[a > 10]

print(b.base is a.base)
print(c.base is a.base)
print(d.base is a.base)
print(e.base is a.base)
print(f.base is a.base)

#3
import time
import numpy as np

def time_it(fn, n=100):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000  # ms

a = np.random.randn(2000, 2000)

print(time_it(lambda: a[1000, :].sum()))    
print(time_it(lambda: a[:, 1000].sum()))    

#4
x = np.random.randn(1000, 512)
x-= np.mean(x,axis=-1,keepdims=True)
x/=np.std(x,axis=-1,keepdims=True)

print(np.mean(x,axis=-1))
print(np.std(x,axis=-1))

#5
def zero_negatives(arr):
    result = arr.copy()
    result[result < 0] = 0
    return result

x = np.random.randn(4, 4)
out = zero_negatives(x)
print(x.shape)
print(out.shape)