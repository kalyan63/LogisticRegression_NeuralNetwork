from matplotlib import colors
from LogisticRegression.SoftMax import Softmax 
from matrix import *
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# For Question 1 answers
print("For Vectorized and batch size=5")
for j in [100,500,1000,5000,10000]:
    model=Softmax()
    model.fit_vectorized(X_train,y_train,batch_size=5,n_iter=j)
    y_hat=model.predict(X_test)
    print("\tAccuracy for iteration={} is: {}".format(j,accuracy(y_hat,y_test)))

#For Autograd 
print("For Autograd and batch size=5")
for j in [100,500,1000,5000,10000]:
    model=Softmax()
    model.fit_autograd(X_train,y_train,batch_size=5,n_iter=j)
    y_hat=model.predict(X_test)
    print("\tAccuracy for iteration={} is: {}".format(j,accuracy(y_hat,y_test)))

#Print Confusion matrix
model2=Softmax()
model2.fit_vectorized(X_train,y_train,batch_size=5,n_iter=10000)
model2.confusion_matrix(X_test,y_test)

#PCA fit
pca=PCA(n_components=2)
Reduced=pca.fit_transform(Digit_data.iloc[:,:-1])
x_ax = Reduced[:,0]
y_ax = Reduced[:,1]
digit = Digit_data.iloc[:,-1]
for digit_type in np.unique(digit):
    i = np.where(digit == digit_type)
    plt.scatter(x_ax[i], y_ax[i], label=digit_type)
plt.legend(prop={'size':10})
plt.show()
