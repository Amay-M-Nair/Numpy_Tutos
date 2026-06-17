import numpy as np

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


##Final Exercise
scores = np.array([
    [72, 85, 90, 60, 78],   # student 0
    [88, 92, 76, 95, 84],   # student 1
    [55, 60, 70, 65, 58],   # student 2
    [91, 88, 94, 87, 90],   # student 3
    [40, 55, 60, 50, 45],   # student 4
    [78, 82, 80, 75, 85],   # student 5
])

#1
avg_stud_score = np.mean(scores,axis=1)
avg_sub_score = np.mean(scores,axis = 0)
print(avg_stud_score)
print(avg_sub_score)
best_stud = np.argmax(avg_stud_score)
print(best_stud)

#2
score_max = np.max(scores,axis=0,keepdims=True)
score_min = np.min(scores,axis=0,keepdims=True)
norm_x = (scores - score_min)/(score_max-score_min)
print(np.round(norm_x,2))

#3
graded = np.where(scores>=65,1,0)
print(graded)
mask = 1-graded
fails = np.sum(mask)
print(fails)

#4
graded = np.where(scores>=65,1,0)
toppers = np.where(np.sum(graded,axis=1)==5,1,0)
n_toppers = np.sum(toppers)
print(n_toppers)

#5
new_scores = scores.reshape(3,10)
means = np.mean(new_scores,axis=1)
print(means.shape)

#6
scores_sub_mean = np.mean(scores, axis=0, keepdims=True)
scores_sub_std = np.std(scores,axis=0,keepdims=True)
stand_scores = (scores-scores_sub_mean)/scores_sub_std
print(np.mean(stand_scores,axis=0))
print(np.std(stand_scores,axis=0))

#7
bonus = np.full((6,5),5)
bonus_scores = scores+bonus
stack_scores = np.vstack([scores,bonus_scores])
print(stack_scores.shape)

#8
boost_scores = stack_scores[6:,:]
max_sub_idx = np.argmax(boost_scores,axis=1)
print(max_sub_idx)
print(max_sub_idx.shape)