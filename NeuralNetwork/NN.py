import numpy as np
from numpy import exp,log 

class NN():
    def __init__(self,layer_size,activation_layers):
        pass
    #Activation Functions
    def relu(self,a):
        return np.max(a,0)
    def identity(self,a):
        return a
    def sigmoid(self,a):
        return 1.0/(1.0+exp(a))    
    #End of Activation functions
    
    def fit(self,X,y):
        pass
    def forward_prop(self,X):
        pass
    def back_prop(self,X,y):
        pass


