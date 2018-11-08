# data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

# machines
from dnn import DNN

# helper function to calculate mismatches
def mismatches(x,y,title):
    count = 0
    for i,j in zip(x,y):
        if i != j:
            count += 1
    print(title, ' --> total:', len(x), 'mismatches:', count, 'ratio:', count/len(x),
        'model accuracy:', (len(x)-count)/len(x))

# read in dataset
df = pd.read_csv('dataset.txt')
# df = df.sample(frac=1)

# seperate feature vectors and labels
X = df.iloc[:,:-1].values
print(X[0])
X= scale(X)
print(X[0])
y = df.iloc[:, [4]].values

# y_number for adaline, hot_y is one-hot encoding of y for logisticregression_net
y_num = [1 if i=='setosa' else 2 if i=='versicolor' else 3 for i in y]
hot_y = np.delete(np.eye(4)[y_num], 0, axis=1)

print('\n\nMachines with 2 Features for Graphing\n')



d = DNN(shape=[4,1,3], eta =1, n_epoch=100)
d.fit(X, hot_y)
p = d.predict(X)
p = [list(i).index(max(list(i))) + 1 for i in p]
mismatches(p,y_num,'ass')

# lrn1 = lrn.LogisticRegressionNet(eta=.0001, n_iter=5000, C=0.0009)
# lrn1.fit(X, hot_y)
# predictions = lrn1.predict(X)
# mismatches(predictions, y_num, 'Logistic 2D')
# pdr.graph(X, y_num, lrn1, 'Logistic Net in 2D')



print('\n\nEnd.\n')
