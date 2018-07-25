import cPickle  as pickle
import numpy as np

x1 = pickle.load(open("training_set_positives.p", "rb"))
x2 = pickle.load(open("training_set_negatives.p", "rb"))
x1 = np.array(x1.values())
x2 = np.array(x2.values())
x= np.concatenate((x1, x2))

# ================= First feature
def feature1 (x):
    ratios_x = []
    for i in range(x.shape[0]):
        ratios_x.append(np.count_nonzero(x[i])/float(x[i].size))
    feature1_value = np.array(ratios_x)
    return feature1_value

#ratios_x2 = []
#for i in range(x1.shape[0]):
 #   ratios_x2.append(np.count_nonzero(x2[i])/float(x2[i].size))
#ratios_x2 = np.array(ratios_x2)

#feature1 = np.concatenate((ratios_x,ratios_x2))

# ================== Second Feature
def feature2(x):
    def count_adjacent_true(arr):
        assert len(arr.shape) == 1
        assert arr.dtype == np.bool
        if arr.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        if sum(arr) == 0:
            return 0
        sw = np.insert(arr[1:] ^ arr[:-1], [0, arr.shape[0]-1], values=True)
        swi = np.arange(sw.shape[0])[sw]
        offset = 0 if arr[0] else 1
        lengths = swi[offset+1::2] - swi[offset:-1:2]
        return max(lengths)

    row = np.array( x,dtype=bool)
   # row = np.array([row for row in samples])
    #print row.shape
    #flat_rows=[]
    #for item in row:
     #   flat_rows.append(item.flatten())
    feature2_value = []
    for item in row:
        count = 0
        for j in item:
            count += count_adjacent_true(j)
        feature2_value.append(count)

    feature2_value= np.array(feature2_value)
    return feature2_value

f1 = feature1(x)
f2 = feature2(x)

features = np.column_stack((f1,f2))
pos = np.ones(150)
neg = - pos
output = np.concatenate((pos,neg))