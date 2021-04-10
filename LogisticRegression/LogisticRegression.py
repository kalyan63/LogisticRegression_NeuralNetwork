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

    def fit_non_regularized(self,X,y,batch_size=1,n_iter=100,lr=0.01,lr_type='constant'):
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()    
        assert(X.shape[0]==y.shape[0])
        self.__attribute={}
        X=np.array(X)
        X_intercept=np.ones(X.shape[0])
        self.__coef=np.zeros(X.shape[1]+1)
        if(not self.intercept):
            X_intercept=np.zeros(X.shape[0])
        X_arr=np.vstack((X_intercept,X.T)).T
        y_arr=np.array(y)
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                self.__coef=self.__coef+lr*np.matmul(X_arr_b.T,(y_arr_b-self.predict(X_arr_b[:,1:],round=False)))
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
                self.__coef=self.__coef+(lr/iter)*np.matmul(X_arr_b.T,(y_arr_b-self.predict(X_arr_b[:,1:],round=False)))
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given")        
            quit()

        

    def fit_autograd(self,X,y,lt=0.01,lr_type='constant',regularise=''):
        pass
    
    #round is used to get rounded 0 or 1 else we can get exact sigmaoid value. 
    def predict(self,X,round=True):
        X_arr=np.array(X)
        y_hat=np.matmul(X_arr,self.__coef[1:])
        y_hat=y_hat+self.__coef[0]
        sig=np.vectorize(self.__sigmoid)
        y_hat=sig(y_hat)
        if(not round):
            return y_hat
        else:    
            y_hat=np.where(y_hat<0.5,0,1)
            return y_hat

    def plot_surface(self,X,y):
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        # y_hat=list(self.predict(X))
        y_hat=list(y)
        x_axis=list(X.iloc[:,0])
        y_axis=list(X.iloc[:,1])
        for i in range(len(x_axis)):
            if(y_hat[i]==1):
                plt.scatter(x_axis[i],y_axis[i],c='RED',cmap=plt.cm.RdYlBu)
            else:
                plt.scatter(x_axis[i],y_axis[i],c='BLUE',cmap=plt.cm.RdYlBu)
        plt.show()