from LogisticRegression.SoftMax import Softmax 
from matrix import *
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

np.random.seed(24)
scalar=MinMaxScaler()
Digits=load_digits(as_frame=True)
Digit_data=Digits.data
Digit_data[list(Digit_data)]=scalar.fit_transform(Digit_data)
Digit_data["Truth"]=Digits.target
Digit_data=Digit_data.sample(frac=1).reset_index(drop=True)
X=np.array(Digit_data.iloc[:,:-1])
y=np.array(Digit_data.iloc[:,-1])

SF=StratifiedKFold(n_splits=4)
best_accuracy=list()
for train, test in SF.split(X,y):
    model=Softmax()
    model.fit_vectorized(X[train],y[train],batch_size=100,n_iter=2000)
    y_hat=model.predict(X[test])
    best_accuracy.append(accuracy(y_hat,y[test]))

print(best_accuracy)
