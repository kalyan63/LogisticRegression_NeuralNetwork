from sklearn.datasets import load_breast_cancer
from LogisticRegression.LogisticRegression import LogisticRegression
from matrix import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

np.random.seed(24)
scalar=MinMaxScaler()
cancer=load_breast_cancer(as_frame=True)
data=cancer.data
data[list(data)]=scalar.fit_transform(data)
data["Truth"]=cancer.target
data=data.sample(frac=1).reset_index(drop=True)
X=np.array(data.iloc[:,:-1])
y=np.array(data.iloc[:,-1])

kf=KFold(n_splits=3)
best_accuracy=list()
for train, test in kf.split(X,y):
    model=LogisticRegression()
    model.fit_Vectorized(X[train],y[train],batch_size=100,n_iter=2000)
    y_hat=model.predict(X[test])
    best_accuracy.append(accuracy(y_hat,y[test]))

print(best_accuracy)
