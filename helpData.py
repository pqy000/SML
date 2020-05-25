import pickle
import numpy as np
import random
from scipy.sparse import csr_matrix

with open('./Data/experimental_adj.pkl', 'rb') as f:
    adj = pickle.load(f)
print(adj.format)

with open('./Data/experimental_features.pkl', 'rb') as f:
    features = pickle.load(f)
print(features.shape)

with open('./Data/experimental_train.pkl', 'rb') as f:
    train = pickle.load(f)
#print(train)
print(min(train))

targets = [i + 543486 for i in range(50500)]
random.shuffle(targets)
k = 500
new_features = np.zeros((k, 100))
for i in range(k):
    for j in range(100):
        num = np.random.random()
        if num >= 0.5:
            new_features[i][j] = 99.9
        else:
            new_features[i][j] = 99.9
new_row = []
new_col = []
for i in range(k):
    node_targets = targets[i*100:(i+1)*100]
    for target in node_targets:
        if target < 593486:
            new_row.append(i)
            new_col.append(target)
        else:
            new_row.append(i)
            new_col.append(target)
            new_row.append(target - 593486)
            new_col.append(i + 593486)
new_data = np.asarray([1 for i in range(len(new_row))])
new_row = np.asarray(new_row)
new_col = np.asarray(new_col)
new_adj = csr_matrix((new_data, (new_row, new_col)), shape=(k, 593486+k), dtype=np.int)
print(new_adj.format)
np.save('feature.npy', new_features)
f = open('adj.pkl', 'wb')
pickle.dump(new_adj, f)
f.close()