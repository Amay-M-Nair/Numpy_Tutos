import numpy as np

##Stacking & Splitting

#Concatenate
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

print(np.concatenate([a,b],axis=0)) #Vertical stacking(4,2)
print(np.concatenate([a,b],axis=1)) #Horizontal stacking(2,4)

#stack
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.stack([a,b],axis=0)) #front axis stacking
print(np.stack([a,b],axis=-1)) #back axis stacking

#hstack & vstack
print(np.vstack([a,b])) #Horizontal stacking(2,4)
print(np.hstack([a,b])) #Vertical stacking(4,2)

#split
x = np.arange(12).reshape(4,3)
parts = np.split(x,2,axis=0) #split to 2 equal parts
print(parts[0]) #(2,3)
print(parts[1]) #(2,3)

parts = np.split(x,[1,3],axis=0) #split to 3 parts 0:1, 1:3, 3:
parts = np.split(x,[1,2,3],axis=0) #split to 4 parts 0:1, 1:2, 2:3, 3:

#DL patterns

sample1 = np.random.randn(3,224,224)
sample2 = np.random.randn(3,224,224)
sample3 = np.random.randn(3,224,224)
batch = np.stack([sample1,sample2,sample3],axis=0) #Batch Stacking at Front
print(batch.shape)

txt = np.random.randn(32,512) #text embeddings
img = np.random.randn(32,512) #image embeddings
combined = np.concatenate([txt,img],axis=1) #Feature Stacking to combine both features
print(combined.shape)

batch = np.random.randn(32,128)
gpu_batches = np.split(batch,4,axis=0) #4 batch of 8 samples each for GPU process.
print(gpu_batches[0].shape) #(8,128)

x = np.random.randn(32,128,192)
Q,K,V = np.split(x,3,axis=-1) #Splitting to Q,K,V attributes in Attention Mech.
print(Q.shape) #(32,128,64)

#Exercise

#1
a = np.array([[1, 2, 3], [4, 5, 6]])     # shape (2, 3)
b = np.array([[7, 8, 9], [10, 11, 12]])  # shape (2, 3)
a_hs= np.concatenate([a,b],axis=0)
a_vs= np.concatenate([a,b],axis=1)
print(a_hs.shape)
print(a_vs.shape)

#2
x = np.array([1,2,3])
y = np.array([4,5,6])
z = np.array([7,8,9])
print(np.stack([x,y,z]).shape) #(3,3) 
print(np.stack([x,y,z],axis=1).shape) #(3,3) but transposed of the prev stacking

#3
sample1 = np.random.randn(3,8,8)
sample2 = np.random.randn(3,8,8)
sample3 = np.random.randn(3,8,8)
sample4 = np.random.randn(3,8,8)
batch_samples = np.stack([sample1,sample2,sample3,sample4],axis=0)
print(batch_samples.shape)

#4
g = np.arange(24).reshape(6, 4)
g_split = np.split(g,3,axis=0)
for i in g_split:
    print(i.shape,"\n",i)

#5
x = np.random.randn(2, 10, 192)  # (batch=2, seq_len=10, 3*d_model)
Q,K,V = np.split(x,3,axis=-1)
print(Q.shape,"\n",K.shape,"\n",V.shape)

#6
a = np.ones((3, 4))
b = np.ones((3, 4))

print(np.concatenate([a, b], axis=0).shape) #(6,4)
print(np.concatenate([a, b], axis=1).shape) #(3,8)
print(np.stack([a, b], axis=0).shape) #(2,3,4)
print(np.stack([a, b], axis=1).shape) #(3,2,4) stacking at index 1