from sklearn.datasets import load_breast_cancer
from LogisticRegression.LogisticRegression import LogisticRegression
from matrix import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(24)
scalar=MinMaxScaler()
cancer=load_breast_cancer(as_frame=True)
data=cancer.data
data[list(data)]=scalar.fit_transform(data)
data["Truth"]=cancer.target
data=data.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(data.shape[0]))
X_train=data.iloc[:split_at,:-1]
y_train=data.iloc[:split_at,-1]
X_test=data.iloc[split_at:,:-1]
y_test=data.iloc[split_at:,-1]

# Vectorized
print("For Vectorized and Batch size=100")
for j in [100,1000,2000,3000,10000]:
    model=LogisticRegression()
    model.fit_Vectorized(X_train,y_train,batch_size=100,n_iter=j)
    y_hat=model.predict(X_test)
    print("\tAccuracy for iteration={} is: {} ".format(j,accuracy(y_hat,y_test)))

# Autograd
print("\n For Autograd and Batch size=300")
for j in [100,1000,5000,10000,20000]:
    model2=LogisticRegression()
    model2.fit_autograd(X_train,y_train,batch_size=300,n_iter=j,regularise='None')
    y_hat=model2.predict(X_test)
    print("\tAccuracy for iteration={} is: {}".format(j,accuracy(y_hat,y_test)))