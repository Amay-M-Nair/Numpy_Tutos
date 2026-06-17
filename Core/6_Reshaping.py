import numpy as np

##Reshaping

#reshape (old_size==new_size)
a = np.arange(12)
print(a.shape)  #(12,)
b = a.reshape(3,4)
print(b)   #(3,4)
c=a.reshape(2,2,3)
print(c.shape) #(2,2,3)

# -1 shortcut
a = np.arange(24)

print(a.reshape(4,-1).shape) #(4,6) -> 24/4 = 6
print(a.reshape(-1,8).shape) #(3,8) -> 24/8 = 3
print(a.reshape(2,3,-1).shape) #(2,3,4) -> 24/(2*3) = 4

#flatten vs ravel (Collapse to 1D)
m = np.array([[1,2,4],[4,5,6]])
f = m.flatten() #always returns copy (for safety)
r = m.ravel() #returns view of copy(if possible), modifying a ravel may alter original(for speed)

#squeeze & expand_dims
a = np.array([[[1,2,3]]]) #shape (1,1,3)

print(np.squeeze(a).shape) #(3,) -> removes ALL size-1 dims
print(np.squeeze(a,axis=0).shape) #(1,3) -> removes only axis 0

a= np.array([1,2,3]) #shape (3,)

print(np.expand_dims(a, axis=0).shape) #(1,3) - add dim at front
print(np.expand_dims(a, axis=1).shape) #(3,1) - add dim at end, can use None to expand too(see Broadcasting)

#Reshape patterns
x = np.random.randn(32,64,7,7) #(batch,channels,H,W)
x = x.reshape(32,-1) #(32,3136) spatial dims flattened for linear layers with batches

img = np.random.randn(3,224,224)  #(C,H,W) single image
img = np.expand_dims(img,axis=0) #fake batch to add dimension for single sample

x = np.random.randn(32,128,512) 
x = x.reshape(32,128,8,64) #split embedding dim into heads 8 heads each with d_k = 64

#Exercises
#1
s = np.arange(36)
s_6_6 = s.reshape(6,6)
s_3_3_4 = s.reshape(3,3,4)
s_2_var = s.reshape(2,-1)
print(s_6_6,)
print(s_3_3_4)
print(s_2_var)

#2
r = np.random.randn(4,3)
print(r.flatten().shape)
print(r.reshape(-1).shape)

#3
img = np.random.randn(3,64,64)
img_new = np.expand_dims(img,axis=0)
print(img_new.shape)
img_new = img[None,:]
print(img_new.shape)

#4
out = np.random.randn(1,1,8)
out2 = np.squeeze(out)
out2_e = np.expand_dims(out2,axis=0)
print(out2.shape)
print(out2_e.shape)

#5
x = np.random.randn(8, 16, 4, 4)  # (batch=8, channels=16, H=4, W=4)
x_f = x.reshape(8,-1)
print(x_f.shape)
print(x[None].shape) #can add fake dim.
