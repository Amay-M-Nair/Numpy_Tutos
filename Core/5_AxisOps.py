import numpy as np

##Axis Operandi

#Axis 
m = np.array([[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12]]) #shape -> (3,4)
print(np.sum(m)) #Sums everything
print(np.sum(m,axis=0)) #Rowvise summation , (4,)
print(np.sum(m,axis=1)) #Colvise summation , (3,)
'''The mental model: **axis=0 collapses rows, axis=1 collapses columns.** The axis you specify is the one that disappears.'''

#Common Axis Ops
print(np.sum(m,axis=0))
print(np.mean(m,axis=0))
print(np.max(m,axis=0))
print(np.min(m,axis=1))
print(np.argmax(m,axis=1))
print(np.std(m,axis=1))

#keepdims 
m = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
mean = np.mean(m,axis=1,keepdims=True) #shape (2,1)
print(m-mean) #No incorrect Broadcasting

#Batch Operations
X = np.random.randn(2,3,4) #Batch of 2, Sample of 3 tokens, 4 feature each
print(np.mean(X,axis=0).shape) #(3,4) - average across batch
print(np.mean(X,axis=1).shape) #(2,4) - average across sequence
print(np.mean(X,axis=2).shape) #(2,3) - average across features

print(np.sum(X,axis=(1,2)).shape) #(2,) - summing everything except batches(multiple axes operation)

X = np.random.randn(32,128,2)
mean = np.mean(X,axis=0,keepdims=True)
std = np.std(X,axis=0, keepdims= True)
X_norm = (X-mean)/(std+1e-8) #Batch Normalization, each sequence now has features with mean=0,std = 1 in a batch

logits = np.random.randn(32,10) #32 batches with 10 classes each, each batch doesnt have any relation to other
logits-=np.max(logits,axis=1,keepdims=True) #Subtract max per sample (32,10)
probs = np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims=True) #Softmax over Batch, Probs for each sample (32,10)
sum_probs = np.sum(probs,axis=1)
print(sum_probs) #Each features in diff sample sums to 1

#Exercises
#1
X = np.array([[ 2.0,  4.0,  6.0,  8.0],
              [ 1.0,  3.0,  5.0,  7.0],
              [ 3.0,  6.0,  9.0, 12.0],
              [ 0.0,  2.0,  4.0,  6.0]])
mean = np.mean(X,axis=0)
print(mean)

#2
sum_row = np.sum(X,axis=1)
print(sum_row)
print(np.argmax(sum_row))

#3
mean = np.mean(X,axis=1,keepdims=True)
print(mean)
print(X-mean)

#4
mean = np.mean(X,axis=0, keepdims=True)
std = np.std(X,axis=0,keepdims=True)
print((X-mean)/std)

#5
logits = np.array([[ 2.0,  1.0,  0.1],
                   [ 0.5,  2.5,  0.3],
                   [ 1.0,  1.0,  3.0],
                   [ 0.2,  0.2,  0.2]])
logits-=np.max(logits,axis=1,keepdims=True)
soft_probs = np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims=True)
print(soft_probs)
sum_probs = np.sum(soft_probs,axis=1)
print(sum_probs)
predicted = np.argmax(soft_probs,axis=1,keepdims=True)
print(predicted)
