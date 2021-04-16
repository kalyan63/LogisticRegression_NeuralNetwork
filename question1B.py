from sklearn.datasets import load_breast_cancer
from LogisticRegression.LogisticRegression import LogisticRegression
from matrix import *
import numpy as np
import pandas as pd
cancer=load_breast_cancer(as_frame=True)
data=cancer.data
data["Truth"]=cancer.target
data=data.iloc[:,[5,6,-1]]
data=data.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(data.shape[0]))
X_train=data.iloc[:split_at,:-1]
y_train=data.iloc[:split_at,-1]
X_test=data.iloc[split_at:,:-1]
y_test=data.iloc[split_at:,-1]

# Vectorized
model=LogisticRegression()
model.fit_Vectorized(X_train,y_train,batch_size=200,n_iter=10000)
y_hat=model.predict(X_test)
print("Accuracy of Vectorized is: ",accuracy(y_hat,y_test))
print("Coefficent is: ",model._LogisticRegression__coef)
model.plot_surface(X_test,y_test)

# Autograd
model2=LogisticRegression()
model2.fit_autograd(X_train,y_train,batch_size=300,n_iter=40000,regularise='None')
y_hat=model2.predict(X_test)
print("Accuracy of Autograd is: ",accuracy(y_hat,y_test))
print("Coefficent is: ",model2._LogisticRegression__coef)
model2.plot_surface(X_test,y_test)