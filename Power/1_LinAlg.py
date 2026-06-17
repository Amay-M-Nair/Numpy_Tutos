import numpy as np

##Linear Algebra

#dot vs matmul
a = np.array([1,2,3]) #(3,)
b = np.array([4,5,6]) #(3,)

print(np.dot(a,b)) #dot product for vectors

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(np.dot(A,B))
print(np.matmul(A,B))
print(A@B)  #All these give matrix multiplication for matrices(2d Arrays)
"""
    They diverge for 3D+. **Rule: always use `@` or `np.matmul` for matrix multiply.
    Use `np.dot` only for 1D dot products.Each output element is a dot product between a row of A and a column of B.
"""

#Batched Matrix Multiply

A = np.random.randn(4,3,5)  #(4,3,5) -> batch of 4 matrices
B = np.random.randn(4,5,2)  #(4,5,2) -> batch of 4 matrices
C = A@B                     #n_batches = const.
print(C.shape)              #(4,3,2) -> 4 independent matrix multiplies

#Transpose

A = np.array([[1,2,3],[4,5,6]])
print(A.T.shape) #.T reverses all axes
print(A.T)

x = np.random.randn(2,3,4) 
print(np.transpose(x,(0,2,1)).shape) #(2,4,3)
print(x.swapaxes(1,2).shape) #(2,4,3) these used for selective axis mapping

K = np.random.randn(2,8,10,64)
K_T = K.transpose(0,1,3,2) 
print(K_T.shape)

#Norms

v = np.array([3.0,4.0]) #Vector
print(np.linalg.norm(v)) #5.0, L-2 norm
print(np.linalg.norm(v,ord=1)) #7.0, L-1 norm
print(np.linalg.norm(v, ord=np.inf)) #4.0, L-infinity norm

A = np.random.randn(3,4) #Matrix
print(A)
print(np.linalg.norm(A)) #Frobenius Norm, sqrt of sum of squares, scalar
print(np.linalg.norm(A,axis=1)) #L-2 norm of each row, shape (3,)
print(np.linalg.norm(A,axis=0,ord=1)) #L-1 norm of each column, shape (4,)
"""
    Gradient clipping computes the global gradient norm across all parameters and scales if it exceeds a threshold.
    Weight decay penalises the Frobenius norm of weight matrices.
"""
#Inverses & Solving Systems

A = np.array([[2.0,1.0],[1.0,3.0]])
A_inv = np.linalg.inv(A) #Inverse, expensive and numerically unstable
print(A @ A_inv) #should be Identity matrix

b = np.array([5.0,10.0]) #Solving Ax = b - more stable than computing inverse
x = np.linalg.solve(A,b)
print(x)   #Solution vector
print(A@x) #should be b

#Eigenvalues & Eigenvectors

A = np.array([[4.0,2.0],[1.0,3.0]])
eigenvalues, eigenvectors = np.linalg.eig(A) #contains eigenvalues and eigenvector attributes
print(eigenvalues) #[5.0 2.0] - the "stretching factors"
print(eigenvectors) 

#SVD

A = np.random.randn(4,3)
U, S, Vt = np.linalg.svd(A, full_matrices=False) #Economy SVD, better for DL
print(U.shape) #(4,3), no extra column
print(S.shape) #(3,) -singular values, always positive
print(Vt.shape) #(3,3)

A_reconstructed = U @ np.diag(S) @ Vt  #To reconstruct A, np.diag(S) -> created diagonal matrix with S
print(np.allclose(A,A_reconstructed))  #True
"""
LoRA approximates a weight update ΔW as a product of two small matrices A @ B which is exactly a rank-r SVD approximation.
"""

#Exercises

#1
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [10., 11., 12.]])  # shape (4, 3)

W = np.random.randn(3, 5)       # shape (3, 5)
b = np.random.randn(5)          # shape (5,)
out = X @ W + b
print(out) 
print(out.shape)

#2
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
B = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])     # (2, 3)
AxB = np.dot(A,B)
AXB = A @ B
print(np.allclose(AxB, AXB)) 

#3
Q = np.random.randn(6, 64)   # 6 tokens, d_k=64
K = np.random.randn(6, 64)   # 6 tokens, d_k=64

normalizer = np.sqrt(64)
K_Transpose = K.T
scores = (Q @ K.T)/normalizer
print(scores.shape)

#4
X = np.array([[3.0, 4.0],
              [5.0, 12.0],
              [8.0, 15.0]])
hypo = np.linalg.norm(X,axis=1)
print(hypo)

#5
gradients = np.array([0.5, -2.0, 1.5, -0.8, 3.0, -1.2])
max_norm = 1.0
global_norm = np.linalg.norm(gradients)
if global_norm > max_norm:
    clipped_gradients = (gradients*max_norm)/global_norm
    print(clipped_gradients)
    print(np.linalg.norm(clipped_gradients))
else:
    print(gradients)
    print(global_norm)

#6
W = np.random.randn(8,4) 
U, S, Vt = np.linalg.svd(W,full_matrices=False)
W_re = U @ np.diag(S) @ Vt
print(np.allclose(W,W_re))
S_rank2 = S[:2]
U_rank2 = U[:,:2]
Vt_rank2 = Vt[:2,:]
W_re_rank2 = U[:,:2] @ np.diag(S_rank2) @ Vt[:2,:]
print(W_re_rank2)