from math import dist
import autograd.numpy as np
from autograd.numpy.linalg import T
from matplotlib.pyplot import axis
import pandas as pd
from autograd.numpy import exp,log
from autograd import elementwise_grad as grad
import matplotlib.pyplot as plt

def lossSoftmax(w,X,y):
    big=exp(np.matmul(X,w.T))
    sum1=np.sum(exp(np.matmul(X,w.T)),axis=1)
    sum1=np.tile(sum1,(len(big[0]),1)).T
    return np.sum(y*log(big/sum1))

class Softmax():
    def __init__(self,intercept=True):
        self.intercept=intercept
        self.__coef=None
        self.__att=None
        self.__attPredict=None
    # Helper Functions
    def __softmaxFunc(self,X):
        Zi=np.array([exp(np.matmul(X,self.__coef[i][1:])+self.__coef[i][0]) for i in range(len(self.__coef))])
        return (Zi/np.sum(Zi))

    def index_to_names(self,a):
        return self.__attPredict[a]

    def __mapAttributes(self,y_train):
        self.__att=dict()
        y_train=np.array(y_train)
        new_y=list()
        for i in y_train:
            if(len(self.__att)==0):
                self.__att[i]=0
            elif(not i in self.__att):
                self.__att[i]=max(self.__att.values())+1
            new_y.append(self.__att[i])
        self.__attPredict=dict([(value,key) for (key,value) in self.__att.items()])    
        return np.array(new_y)            

    def __onehot(self,y):
        y_hot=np.zeros((len(y),len(self.__coef))) 
        for i in range(len(y)):
            y_hot[i][y[i]]=1
        return y_hot       
    #End Of helper Functions

    def fit_vectorized(self,X,y,batch_size=1,n_iter=100,lr=0.01,lr_type='constant'):
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()    
        assert(X.shape[0]==y.shape[0])
        X=np.array(X)
        X_intercept=np.ones(X.shape[0])
        if(not self.intercept):
            X_intercept=np.zeros(X.shape[0])
        X_arr=np.vstack((X_intercept,X.T)).T
        y_arr=self.__mapAttributes(y)
        self.__coef=np.ones((len(self.__att),X.shape[1]+1))   
        # self.__coef=np.zeros((len(self.__att),X.shape[1]+1))   
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                soft_values=self.predict(X_arr_b[:,1:],ret="Values")
                for i in range(len(self.__coef)):
                    self.__coef[i]=self.__coef[i]+lr*np.sum(X_arr_b*np.tile(np.where(y_arr_b==i,1,0)-soft_values[:,i],(len(self.__coef[0]),1)).T,axis=0)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])   
        elif(lr_type=='inverse'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                soft_values=self.predict(X_arr_b[:,1:],ret="Values")
                for i in range(len(self.__coef)):
                    self.__coef[i]=self.__coef[i]+(lr/iter)*np.sum(X_arr_b*np.tile(np.where(y_arr_b==i,1,0)-soft_values[:,i],(len(self.__coef[0]),1)).T,axis=0)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given")        
            quit()  

    def fit_autograd(self,X,y,batch_size=1,n_iter=100,lr=0.01,lr_type='constant'):
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()    
        assert(X.shape[0]==y.shape[0])
        X=np.array(X)
        X_intercept=np.ones(X.shape[0])
        if(not self.intercept):
            X_intercept=np.zeros(X.shape[0])
        X_arr=np.vstack((X_intercept,X.T)).T
        y_arr=self.__mapAttributes(y)
        self.__coef=np.ones((len(self.__att),X.shape[1]+1))   
        gd=grad(lossSoftmax)
        y_arr=self.__onehot(y_arr)
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                self.__coef=self.__coef+lr*gd(self.__coef,X_arr_b,y_arr_b)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])   
        elif(lr_type=='inverse'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                self.__coef=self.__coef+lr*gd(self.__coef,X_arr_b,y_arr_b)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given")        
            quit()

    def predict(self,X,ret='Attributes'):
        X_arr=np.array(X)
        smfc=np.vectorize(self.__softmaxFunc,signature='(n)->(m)')
        result=smfc(X)
        i2n=np.vectorize(self.index_to_names)
        if(ret=='Values'):
            return result
        else:    
            return i2n(np.argmax(result,axis=1))
    
    def confusion_matrix(self,X,y):
        conf=[[0]*len(self.__att) for i in range(len(self.__att))]
        y_hat=self.predict(X)
        y=np.array(y)
        for i in range(len(y_hat)):
            conf[y_hat[i]][y[i]]+=1
        print(conf)
        plt.imshow(conf)    
        plt.show()
