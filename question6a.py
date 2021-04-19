from matrix import *
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from NeuralNetwork.NN import NN

#To get one hot encoding of digits data
def onehot(y):
    y_hot=np.zeros((len(y),10)) 
    for i in range(len(y)):
        y_hot[i][y[i]]=1
    return y_hot 

np.random.seed(24)
scalar=MinMaxScaler()
Digits=load_digits(as_frame=True)
Digit_data=Digits.data
Digit_data[list(Digit_data)]=scalar.fit_transform(Digit_data)
Digit_data["Truth"]=Digits.target
Digit_data=Digit_data.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(Digit_data.shape[0]))
X_train=Digit_data.iloc[:split_at,:-1]
y_train=Digit_data.iloc[:split_at,-1]
X_test=Digit_data.iloc[split_at:,:-1]
y_test=Digit_data.iloc[split_at:,-1]

#Testing for different iterations
for i in [10,50,100,1000,2000]:
    model=NN()
    model.add_layers(X_test.shape[1],[10],['s'])
    y_train_2=onehot(y_train)
    model.fit(X_train,y_train_2,n_iter=i)
    y_hat2=model.predict(X_test)
    y_hat=np.argmax(y_hat2,axis=1)
    print("Accuracy for iteratin= {}  is: {}".format(i,accuracy(y_hat,y_test)))