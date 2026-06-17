import numpy as np

##Slicing & Indexing

#1D
a = np.array([1,2,3,4,5])
print(a[0])
print(a[-1])
print(a[1:4])
print(a[::2])
print(a[::-1])

#2D
m = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(m[0,0])
print(m[1,2])
print(m[-1,-1])

print(m[0,:])
print(m[:,1])
print(m[1:,:2])

row_0_2d = m[0:1,:] #shape(1,3) -> matrix
row_0_1d = m[0,:]  #shape(3,) -> vector

#Boolean Masking
a = np.array([3,1,4,1,5,9,2,6])
mask = a>3
print(mask) #prints bool values for mask
print(a[mask]) #prints true elements
print(a[a>3])

#Where?
a = np.array([3,-1,4,-1,5,-9])
result = np.where(a>0,a,0) #works like ternary: condn, true,false
print(result)

#Fancy
m = np.array([[1,2],[2,4],[5,6],[3,6]])
print(m[[0,2,3]]) #rows passed as array

#Exercises
m = np.array([[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12],
              [13, 14, 15, 16]])

#1
row_3_1d = m[2,:]
print(row_3_1d.shape)

#2
column_2_1d = m[:,1]
print(column_2_1d.shape)

#3
sub_bottom_2x2 = m[2:,2:]
print(sub_bottom_2x2)

#4
print(m[m>10])

#5
a = np.where(m<8,0,m)
print(a)

#6
vocab = np.array([[0.1, 0.2],   # token 0
                  [0.3, 0.4],   # token 1
                  [0.5, 0.6],   # token 2
                  [0.7, 0.8]])  # token 3
print(vocab[[2,0,3,1]])
