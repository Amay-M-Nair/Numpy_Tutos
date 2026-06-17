import numpy as np

## Advanced Masking & Indexing

#Gather
scores = np.array([[0.1, 0.5, 0.3, 0.1],
                   [0.2, 0.1, 0.6, 0.1],
                   [0.4, 0.3, 0.2, 0.1],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.3, 0.3, 0.3, 0.1]])  #shape (5,4)

labels = np.array([1,2,0,3,2])
correct_scores = scores[np.arange(5),labels] #first is row parameter, second is column parameter
print(correct_scores)

#Padding
tokens = np.array([[4, 7, 2, 9, 0, 0],
                   [3, 1, 0, 0, 0, 0],
                   [8, 5, 6, 2, 7, 1]])

pad_mask = (tokens != 0)
print(pad_mask) #gives boolean array with pad mask check

scores = np.random.randn(3,6,6)
key_mask = pad_mask[:,None,:] #shape (3,1,6) to broadcast over every pos
scores = np.where(key_mask,scores,-1e9) #to convert 0 to -1e9 to prevent attending those pos

#Casual Mask
"""
For autoregressive models (GPT-style), token at position i can only attend to positions 0...i. 
This prevents the model from "seeing the future"
"""
seq_len = 6

casual_mask = np.tril(np.ones((seq_len,seq_len))).astype(bool) #upper triangle(exc diag) = future pos(masked)
print(casual_mask.astype(int))

scores = np.random.randn(6,6)
masked_scores = np.where(casual_mask,scores, -1e9) #upper triangle approx 0, softmax gives 0 atttention


seq_len = 6
batch = 3

scores = np.random.randn(3,6,6)
casual_mask = casual_mask = np.tril(np.ones((seq_len,seq_len))).astype(bool)
pad_mask = (scores!=0)
combined = casual_mask & pad_mask #combining both padding and casual masks

#Scatter
n_classes = 5
labels = np.array([2,0,4,1])

one_hot = np.zeros((4,n_classes)) #empty one-hot matrix
one_hot[np.arange(4),labels] = 1.0 #Scattering 1s into pos based on row,column paras
print(one_hot)

#np.where with Conditions
x = np.array([-3,-1,0,2,4,6,8])

clipped = np.where(x<0,0,np.where(x>5,5,x)) #clipping to be in range [0,5]
print(clipped)

leaky = np.where(x>0, x , 0.01*x) #leaky relu
print(leaky)

#Argmax, Argmin, Argsort

scores = np.array([[0.1, 0.7, 0.2],
                   [0.5, 0.3, 0.2],
                   [0.1, 0.2, 0.7]])

preds = np.argmax(scores,axis=1)
print(preds) #predicted class per sample

sorted_idx = np.argsort(scores,axis=1)
print(sorted_idx) #sorted indices by score(ascending)

top2 = np.argsort(scores,axis=1)[:,::-1][:,:2] #argsort + descending + slice to 2 = top2 
print(top2) #top-k selection

#Exercises

#1
probs = np.array([[0.1, 0.6, 0.2, 0.1],
                  [0.3, 0.1, 0.5, 0.1],
                  [0.2, 0.2, 0.2, 0.4],
                  [0.7, 0.1, 0.1, 0.1]])

labels = np.array([1, 2, 3, 0])

correct_probs = probs[np.arange(4),labels]
cross_entropy = np.mean(-np.log(correct_probs))
print(cross_entropy)

#2
tokens = np.array([[5, 3, 8, 0, 0],
                   [2, 7, 1, 9, 4],
                   [6, 0, 0, 0, 0]])

pad_mask = (tokens!=0)
seq_len = np.sum(np.where(pad_mask,1,0))
print(seq_len)

#3
seq_len = 5
x = np.random.randn(5,5)

casual_mask = np.tril(np.ones((seq_len,seq_len))).astype(bool)
x_corr = np.where(casual_mask,x,-1e9)

x_corr-= np.max(x_corr,axis=1,keepdims=True)
probs = np.exp(x_corr)/np.sum(np.exp(x_corr),axis=1,keepdims=True)

print(probs)

#4
labels = np.array([3, 0, 2, 1, 4])
n_classes = 5

one_hot = np.zeros((n_classes,labels.size))
one_hot[np.arange(n_classes),labels]=1.0
print(one_hot)

one = np.ones((5,1))
oner = np.sum(one_hot,axis=1)
print(np.allclose(one,oner))

#5
logits = np.random.randn(4, 6)

idx = np.argsort(logits)[:,::-1][:,:3]
print(idx)

#6
tokens = np.array([[3, 7, 2, 5, 0],
                   [1, 4, 0, 0, 0],
                   [8, 2, 6, 1, 9]])  # (batch=3, seq_len=5)

scores = np.random.randn(3, 5, 5)    # attention scores (batch, query, key)
seq_len = 5

pad_mask = (tokens!=0)
pad_mask = pad_mask[:,None,:]

casual_mask = np.tril(np.ones((seq_len,seq_len))).astype(bool)
casual_mask = casual_mask[None,:,:]

mask = pad_mask & casual_mask

corrected_scores = np.where(mask,scores,-1e9)
corrected_scores-=np.max(corrected_scores,axis=-1,keepdims=True)
probs = np.exp(corrected_scores)/np.sum(np.exp(corrected_scores),axis=-1,keepdims=True)

print(probs)