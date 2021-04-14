from LogisticRegression.LogisticRegression import LogisticRegression
from matrix import *
import numpy as np
import pandas as pd 

da=pd.read_csv('iris.csv')
col1=da["sepal_width"]
col2=da["petal_width"]
label=np.array(da["species"])
label=np.where(label=="virginica",1,0)
iris=pd.merge(col1,col2,left_index=True,right_index=True)
iris["Truth"]=label
iris=iris.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(iris.shape[0]))
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]

model=LogisticRegression()
model.fit_autograd(X_train,y_train,n_iter=1000,batch_size=X_train.shape[0],regularise='l1',regularise_value=0.01)
print(model._LogisticRegression__coef)
y_hat=model.predict(X_test)
print("Accuracy: ",accuracy(y_hat,y_test))
model.plot_surface(X_test,y_test)