import numpy as np
from pprint import pprint
from copy import copy
class FNN ():

    def __init__(self, shape, eta=1, n_epoch=1000):

        self.eta = eta
        self.n_epoch = n_epoch
        self.shape = shape
        self.num_layers = len(shape)
        self.initialize(shape)
        self.Error = []

    def fit(self, X, y):
        for _ in range(self.n_epoch):
            error = []
            for x,y_ in zip(X,y):
                self.feed_foward(x)
                e = self.back_propagate(y_)
                error.append(e)
            self.Error.append(sum(e))

    def predict(self, X):
        return np.array([copy(self.feed_foward(x)) for x in X])

    def feed_foward(self, X):
        self.z[0][:-1] = X
        for i in range(1, self.num_layers):
            input = np.dot(self.z[i-1], self.w[i])
            self.z[i][:-1] = self.activation(input)
            self.act_prime[i] = np.append(self.activation_prime(input),[1])
        return self.z[-1][:-1]

    def back_propagate(self, y):

        # helper functions to format data and calculate gradient (delta weights) and error
        gradient = lambda z, e: -self.eta * np.dot(np.atleast_2d(z).T, np.atleast_2d(e))
        error = lambda e, w, p: np.dot(np.atleast_2d(e), np.atleast_2d(w).T) * p

        # calculate error in output layer and update weights
        self.E[-1] = (self.z[-1][:-1] - y) * self.act_prime[-1][:-1]
        self.w[-1] += gradient(self.z[-2], self.E[-1])

        #TODO: fix last line in loop --> ugly and not needed! 
        # propagate error back through net and update weights
        for i in range(2, self.num_layers):
            self.E[-i] = error(self.E[-i+1], self.w[-i+1], self.act_prime[-i])
            self.w[-i] += gradient(self.z[-i-1], np.atleast_2d(self.E[-i][0][:-1]))
            self.E[-i] = self.E[-i][0][:-1]

        return (self.z[-1][:-1]-y)**2


    def activation(self, X):
        return 1 / (1+np.exp(-X))

    def activation_prime(self, X):
        return self.activation(X)*(1-self.activation(X))

    def initialize(self, shape):

        self.w = [[]] # weights
        self.E = [] # errors
        self.z = [] # node outputs
        self.act_prime = [] # derivative of activation function

        # cycle through all layers but the output layer
        for i in range(len(shape)-1):
            weights = np.random.uniform(-2,2,(shape[i]+1, shape[i+1]))
            self.w.append(weights)
            self.z.append(np.ones((shape[i]+1,)))
            self.act_prime.append(np.zeros((shape[i]+1,)))
            self.E.append(np.zeros(shape[i],))

        # add an output list for the output layer
        self.z.append(np.ones((shape[-1]+1,)))
        self.act_prime.append(np.zeros((shape[-1]+1,)))
        self.E.append(np.zeros((shape[-1])))
