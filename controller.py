# data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
import timeit
import matplotlib.pyplot as plt
from plot_decision_regions import graph

# machines
from fnn import FNN

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
df = df.sample(frac=1)

# seperate feature vectors and labels
X = df.iloc[:,:-1].values
X = scale(X)
y = df.iloc[:, [4]].values

# y_number for adaline, hot_y is one-hot encoding of y for logisticregression_net
y_num = [1 if i=='setosa' else 2 if i=='versicolor' else 3 for i in y]
hot_y = np.delete(np.eye(4)[y_num], 0, axis=1)

# run my neural net
start = timeit.default_timer()
d = FNN(shape=[4,1,1,3], eta =.1, n_epoch=100)
d.fit(X, hot_y)
p = d.predict(X)
p = [list(i).index(max(list(i))) + 1 for i in p]
stop = timeit.default_timer()
# plt.plot(d.Error)
# plt.show()
print('\n\nTime: ', stop-start)
mismatches(p,y_num,'My network:')

# run sklearns neural net
start = timeit.default_timer()
d = MLPClassifier(hidden_layer_sizes=[1,1], learning_rate_init=.1, max_iter=100)
d.fit(X, hot_y)
p = d.predict(X)
p = [list(i).index(max(list(i))) + 1 for i in p]
stop = timeit.default_timer()
print('\n\nTime: ', stop-start)
mismatches(p, y_num, 'Sklearns network:')

print('\n\nEnd.\n')
