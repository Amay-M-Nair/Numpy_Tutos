import numpy as np

##Einsum

"""
Rule 1: Each letter represents one dimension. Same letter = same dimension.
Rule 2: Letters that appear on the left but not in the output -> are summed over (contracted).
Rule 3: Letters in the output -> are kept.
"""

#Dot Product
a = np.array([1.0,2.0,3.0])
b = np.array([4.0,5.0,6.0])

result = np.einsum('i,i->',a,b) #sum of a[i]*b[i]
print(result)

#Element-wise multiply - keep dims
result = np.einsum('i,i->i',a,b) 
print(result) #result[i] = a[i]*b[i]

#Outer Product
result = np.einsum('i,j->ij',a,b)
print(result) #result[i][j] = a[i]*b[j]

#Matrix Multiply
A = np.random.randn(3,4)
B = np.random.randn(4,5)

result = np.einsum('ij,jk->ik',A,B) #result[i,k] = A[i,j]*B[j,k]
print(result.shape)

#Matrix Transpose
A_t = np.einsum('ij->ji',A) #A_t[j,i] = A[i,j]
print(result.shape)

#Trace
A = np.random.randn(4,4)
result = np.einsum('ii->',A) #Sum of a[i,i]

#Sum along axis
np.einsum('ij->i',A) #row sum
np.einsum('ij->j',A) #column sum

#Batched Operations

A = np.random.randn(8, 3, 4)   # batch of 8, each (3,4)
B = np.random.randn(8, 4, 5)   # batch of 8, each (4,5)
result = np.einsum('bij,bjk->bik',A,B) #result[b,i,k] = A[b,i,j] * B[b,j,k]
print(result.shape) #Batch Matrix Multiplications

A = np.random.randn(8, 64)   # 8 vectors
B = np.random.randn(8, 64)   # 8 vectors
result = np.einsum('bi,bj->b',A,B) #result[b] = A[b,i]*B[b,j]
print(result.shape) #Batch dot products


#Attention Score Computations(DL)
batch, heads, seq, d_k = 2,8,10,64

Q = np.random.randn(batch, heads, seq, d_k)
K = np.random.randn(batch, heads, seq, d_k)
V = np.random.randn(batch, heads, seq, d_k)

scores = np.einsum('bhqd,bhkd->bhqk',Q,K)/np.sqrt(d_k) #scores = Q @ K^T /sqrt(d_k), scores[b,h,q,k] = (Q[b,h,q,d] * K[b,h,k,d])/np.sqrt(d_k)
print(scores.shape)#attention scores, ie:step1

scores = scores - scores.max(axis=-1, keepdims=True)
attn_weights = np.exp(scores)/np.sum(np.exp(scores),axis=-1,keepdims=True) #Softmax last axis on scores, ie:step2

out = np.einsum('bhsq,bhqd->bhsd',attn_weights,V) #attn_weights @ V, out[b,h,s,d] = attn_weights[b,h,s,q] * V[b,h,q,d]
print(out.shape) #Weighted sum of values, ie: step3

#Exercise

#1
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
dp = np.einsum('i,i->',a,b)
ewm = np.einsum('i,i->i',a,b)
op = np.einsum('i,j->ij',a,b)
print(dp,'\n',ewm,'\n',op)

A = np.random.randn(3,4)
B = np.random.randn(4,2)
print(np.einsum('ij,jk->ik',A,B))

#2
X = np.random.randn(6, 4)
CX1 = X.T @ X
CX2 = np.einsum('bi,bj->ij',X,X)
print(np.allclose(CX1,CX2))

#3
A = np.random.randn(5, 8)   # 5 vectors of dim 8
B = np.random.randn(5, 8)   # 5 vectors of dim 8
bdp = np.einsum('bi,bi->b',A,B)
print(bdp,'\n',bdp.shape)

#4
Q = np.random.randn(6, 32)   # 6 tokens, d_k=32
K = np.random.randn(6, 32)
V = np.random.randn(6, 32)

attn_scores = np.einsum('ik,jk->ij',Q,K)
weighted_V = np.einsum('ij,jk->ik',attn_scores,V)
print(weighted_V.shape)

#5
V_batch = np.array([[1,2,4],[4,7,6],[5,3,8],[3,7,1]])
adotb = np.einsum('ij,kj->ik',V_batch,V_batch)
a_sq = np.einsum('ij,ij->i',V_batch,V_batch)
b_sq = np.einsum('ij,ij->i',V_batch,V_batch)

a_sq = a_sq[:,None] #To broadcast
b_sq = b_sq[None,:] #To broadcast
L2dist = a_sq + b_sq - 2*adotb
print(L2dist)

#6
W = np.random.randn(5, 4)   # weight matrix
x = np.random.randn(4)      # single input vector
wX = W @ x
Wx = np.einsum('ij,j->i',W,x)
print(np.allclose(wX, Wx))

Bx = np.random.randn(3,4) #Batch of 3 vectors
BxW = Bx @ W.T
bxw = np.einsum('ij,kj->ik',Bx,W)
print(np.allclose(BxW,bxw))

#7
batch, heads, seq, d_k = 3, 4, 8, 16

Q = np.random.randn(batch, heads, seq, d_k)
K = np.random.randn(batch, heads, seq, d_k)
attn_scores = np.einsum('bhij,bhkj->bhik',Q,K)/np.sqrt(d_k)
attn_scores2 = (Q @ K.transpose(0,1,3,2))/np.sqrt(d_k)
print(np.allclose(attn_scores,attn_scores2))
print(attn_scores.shape)

mask = np.triu(np.ones((seq,seq)),k=1).astype(bool) #Upper Triangular Matrix with j>i -> True
mask = mask.reshape(1,1,seq,seq)#To broadcast properly
print(mask)
attn_scores = np.where(mask,-1e9,attn_scores)
print(attn_scores[0,0,0,1]) #Autoregression

attn_scores -= np.max(attn_scores,axis=-1,keepdims=True) #max in each query,axis=-1
soft_scores = np.exp(attn_scores)/np.sum(np.exp(attn_scores),axis=-1,keepdims=True) #per query, axis=-1
print(soft_scores.shape) #Softmax outputs,  Autoregressed Attention Scores Computation

#8
seq_q, seq_k, d = 5, 7, 16
W = np.random.randn(d, d)    # bilinear weight matrix
Q = np.random.randn(seq_q, d)
K = np.random.randn(seq_k, d)

scores = np.einsum('ij,jk,lk->il',Q,W,K)
print(scores.shape)
x23 = scores[2][3]
X23 = Q[2] @ W @ K[3]
print(np.isclose(x23,X23))

batch = 4
Q = np.random.randn(batch, seq_q, d)
K = np.random.randn(batch, seq_k, d)
scores = np.einsum('bij,jk,blk->bil',Q,W,K)
print(scores.shape) # Billinear Attention Mechanism