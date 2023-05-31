import numpy as np
import pickle

def loadData(fname='data/ip_distributions_seven_mins'):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)

data = loadData()
# data = data[:10]
keys = set()
for x in data:
    keys = keys.union(set(x.keys()))

sortedKeys = np.array(sorted(list(keys)))

n = len(data)
d = len(sortedKeys)
print(n, d)
mat = np.zeros((n, d))

for i in range(n):
    for key in data[i]:
        if key >= d or sortedKeys[key] != key:
            k = np.searchsorted(sortedKeys, key)
        else:
            k = key
        mat[i, k] = data[i][key]

np.save('data/ip_sevenmin.npy', mat)