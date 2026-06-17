import numpy as np

##Random & Reproducibility

#Random Module
rng = np.random.default_rng(0) #seed
"""
Same seed = same sequence of random numbers = reproducible results.
"""
u = rng.random((3,4)) #uniform [0,1)
print(u.min(),u.max())
n = rng.standard_normal((3,4)) #Standard normal N(0,1) and shape(3,4)
print(n.mean(),n.std()) #approx 0 and 1

n2 = rng.normal(loc=5.0, scale=2.0, size=(3,4)) #normal with custom mean and std
print(n2.mean(),n2.std()) #approx 5 and 2
print(n2.shape)

i = rng.integers(0,10,size=(5,)) #Integers [0,10)
print(i) #size = 5

arr = np.array([10,20,30,40,50])
print(rng.choice(arr,size=3,replace=False)) #Random choice array, w/o Replacement

#Weight Initialization

W = np.zeros((4,4)) # BAD: all zeros
"""Every neuron computes the same thing → symmetry never breaks → network doesn't learn"""

W = np.random.randn(4, 4) * 10 ## BAD: too large
"""Activations explode through layers → gradients explode → NaN loss"""

W = np.random.randn(4, 4) * 0.0001 # BAD: too small
"""Activations vanish through layers → gradients vanish → nothing learns"""

#Xavier-Glorot Initialization (Designed for tanh and sigmoid activations)

def xavier_uniform(fan_in,fan_out):    
    limit = np.sqrt(6.0/ (fan_in + fan_out))
    return np.random.uniform(-limit,limit,size=(fan_in,fan_out))

def xavier_normal(fan_in,fan_out):
    std = np.sqrt(2.0/(fan_in+fan_out))
    return np.random.randn(fan_in,fan_out)*std

W = xavier_normal(512,256)
print(W.std()) #if input has variance 1, setting Var(W) = 2/(fan_in+fan_out) keeps output variance approx 1

#Kaiming-He Initialization (Designed for ReLU activations)

def kaiming_normal(fan_in,fan_out):
    std = np.sqrt(2.0/fan_in)
    return np.random.randn(fan_in,fan_out) * std
    #The factor of 2 compensates for ReLU killing half the activations

W = kaiming_normal(512,256)
print(W.std())

#Verifying Initialization (Stable Variance Check)

np.random.seed(42)
n_layers = 10
x = np.random.randn(256,512)

print(f'Input std:{x.std():.4f}')

for i in range(n_layers):
    W = np.random.randn(512,512) #std approx 1
    x = np.tanh(x@W) #dead neurons no major change in std
    print(f"Layer {i+1} std: {x.std():.4f}") #BAD init - std explodes or vanishes

for i in range(n_layers):
    W = xavier_normal(512,512)
    x = np.tanh(x@W) #tanh squishes so variance shrinks, but still there is learning
    print(f"Layer {i+1} std: {x.std():.4f}")
    #still the perfect should be the std remains stable around 1

#Dropout 

np.random.seed(0)

def dropout(x,p=0.5, training=True):
    if not training:
        return x #no dropout at inference
    
    mask = np.random.rand(*x.shape) > p #Create binary mask: 1 with probability(1-p), 0 with probability p
    print(mask)
    return x * mask/(1-p)  #Scale by 1/(1-p) so expected value is unchanged - inverted dropout

x = np.ones((4,6))
print(dropout(x,p=0.5)) #roughly half the values are 0, half are 2.0(1/0.5)
"""The scaling ensures that at inference (when you just return x), 
   the expected magnitude is the same as during training.
   But some are killed during each forward pass depends on p"""


#Exercise 

#1
rng1 = np.random.default_rng(99)
a = rng1.standard_normal((3,4))
print(a)

rng2 = np.random.default_rng(99)
b = rng2.standard_normal((3,4))
print(b)

print(np.allclose(a,b))

#2
n_layers = 6
x = np.random.randn(256,512)
print("Xavier:")
for i in range(n_layers):
    W = xavier_normal(512,512)
    x = np.tanh(x@W)
    print(f"Layer {i+1}: std: {x.std():.4f}")

print("Kaiming:")
x = np.random.randn(256,512)
for i in range(n_layers):
    W = kaiming_normal(512,512)
    x = np.tanh(x@W)
    print(f"Layer {i+1}: std: {x.std():.4f}")

#3
x = np.ones((1000,))

b = dropout(x,0.3)
print(b[b==0].size)
print(f"{np.abs((1/0.7)-np.mean(b[b>0])):.8f}")

#4
def Relu(x):
    return np.maximum(0,x)

x = np.random.randn(128,256)
layer_size = 256
n_layers = 10

xL = xS = xK = x

for i in range(n_layers):
    WL = np.random.randn(256,256)*1.0 #large std (Exploding)
    WS = np.random.randn(256,256)*0.01 #small std (Vanishing)
    K = kaiming_normal(256,256) #kaiming

    xL = Relu(xL@WL)
    xS = Relu(xS@WS)
    xK = Relu(xK@K)

    print(f"Large layer {i+1}: {xL.std():.4f}")
    print("==========================")
    print(f"Small layer {i+1}: {xS.std():.4f}")
    print("==========================")
    print(f"Kaiming layer {i+1}: {xK.std():.4f}")
    print("==========================")

#5
probs = np.array([0.1,0.3,0.4,0.2]) #prob for each segment i

def random_choice(probs, samples):
    boundary = np.cumsum(probs) #cumulative sum of prob for each segment
    r = np.random.rand(samples) #prob of sample r
    return np.searchsorted(boundary,r) #for returning which index will each r fit

samples = random_choice(probs,10000)

for i in range(len(probs)):
    emp = np.mean(samples==i)
    print(f"idx {i}: expected {probs[i]}, got {emp}") #Inverse Transform Sampling