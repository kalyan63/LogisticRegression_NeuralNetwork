import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class LogisticRegression():
    def __init__(self,intercept=True):
        self.__coef=None
        self.intercept=intercept
        self.__attribute=None

    def __sigmoid(self,z):
        return (1/(1+math.exp(-z)))

    # def fill_att(self,Y):
    #     y=list()
    #     for i in Y:
    #         if(len(self.__attribute)==0):
    #             self.__attribute[i]=0
    #             y.append(0)
    #         elif(i in self.__attribute):
    #             y.append(self.__attribute[i])
    #         else:
    #             self.__attribute[i]=max(self.__attribute.values())+1
    #             y.append(self.__attribute[i]) 
    #     self.__attribute=dict([(key,value) for (value,key) in self.__attribute.items()])        
    #     return np.array(y)

    def fit_non_regularized(self,X,y,lr=0.01,lr_type='constant'):
        assert(X.shape[0]==y.shape[0])
        self.__attribute={}
        X=np.array(X)
        X_intercept=np.ones(X.shape[0])
        # self.__coef=np.ones(X.shape[1]+1)
        self.__coef=np.zeros(X.shape[1]+1)
        if(not self.intercept):
            X_intercept=np.zeros(X.shape[0])
        X=np.vstack((X_intercept,X.T)).T
        # y=self.fill_att(y)
        ##training

    def fit_autograd(self,X,y,lt=0.01,lr_type='constant',regularise=''):
        pass

    def predict(self,X):
        X=np.array(X)
        y_hat=np.matmul(X,self.__coef[1:])
        y_hat=y_hat+self.__coef[0]
        sig=np.vectorize(self.__sigmoid)
        y_hat=sig(y_hat)
        y_hat=np.where(y_hat<0.5,0,1)
        return y_hat

    def print_surface(self,X,y):
        pass