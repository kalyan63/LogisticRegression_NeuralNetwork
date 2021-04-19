from matrix import *
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from NeuralNetwork.NN import NN

np.random.seed(24)
scalar=MinMaxScaler()
Boston=load_boston()
Boston_data=pd.DataFrame(Boston.data)
Boston_data["Truth"]=Boston.target
Boston_data[list(Boston_data)]=scalar.fit_transform(Boston_data)
Boston_data=Boston_data.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(Boston_data.shape[0]))
X_train=np.array(Boston_data.iloc[:split_at,:-1])
y_train=np.array(Boston_data.iloc[:split_at,-1])
X_test=np.array(Boston_data.iloc[split_at:,:-1])
y_test=np.array(Boston_data.iloc[split_at:,-1])


#For different iterations
for i in [10,50,1000,2000]:
    Model=NN()
    Model.add_layers(X_train.shape[1],[10,5,1],['r','r','s'])
    Model.fit(X_train,y_train.reshape(-1,1),n_iter=i,loss='mse')
    y_hat=Model.predict(X_test)
    print("RMSE for iteration= {} is {}".format(i,rmse(y_hat,y_test.reshape(-1,1))))
    print("MAE  for iteration= {} is {}".format(i,mae(y_hat,y_test.reshape(-1,1))))