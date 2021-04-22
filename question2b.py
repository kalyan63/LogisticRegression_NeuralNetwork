from LogisticRegression.LogisticRegression import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# x_axis=list(range(1,101))
x_axis=[0.01*i for i in range(1,101)]
y_axis=list()
#For l1 regularization
rg_value=0.01
for j in range(1,101):
    if(j%20==0):
        print("Going at {} ".format(j))
    model=LogisticRegression()
    model.fit_autograd(X_train,y_train,batch_size=1,n_iter=2000,regularise='l1',regularise_value=j*rg_value)
    y_axis.append(np.abs(model._LogisticRegression__coef)) 
y_axis=np.array(y_axis)    
print("For L1 Regularization")        
for i in range(y_axis.shape[1]):
    plt.plot(x_axis,list(y_axis[:,i]),label=("Coef_"+str(i)))
plt.xlabel("Regularize value")
plt.ylabel("Magnitude of coefficent")    
plt.legend()
plt.savefig('q2.png')
plt.show()


    

