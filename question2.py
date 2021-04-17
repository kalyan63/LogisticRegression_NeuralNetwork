from LogisticRegression.LogisticRegression import LogisticRegression
from matrix import *
import pandas as pd
import numpy as np

#Here we the set the value of seed so that the mixing is always same 
np.random.seed(42)
da=pd.read_csv('iris.csv')
label=np.array(da["species"])
label=np.where(label=="virginica",1,0)
iris=da.iloc[:,:-1]
iris["Truth"]=label
iris=iris.sample(frac=1).reset_index(drop=True)
split_at=int(0.7*(iris.shape[0]))
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]

print(len(X_train))
model=LogisticRegression()
model.fit_autograd(X_train,y_train,n_iter=1000,batch_size=10,regularise='l1',regularise_value=0.01)
y_hat=model.predict(X_test)
print("Accuracy: ",accuracy(y_hat,y_test))