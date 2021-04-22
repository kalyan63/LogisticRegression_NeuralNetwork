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
iris_train=iris.iloc[:split_at,:]
iris_test=iris.iloc[split_at:,:]

# Nested Cross validation
# For l1 regularization
# fold=int(len(iris_train)*0.2)
# opt_rg={0:dict(),1:dict(),2:dict(),3:dict(),4:dict()}
# for i in range(5):
#     split_x1=i*fold
#     split_x2=split_x1+fold
#     X_train_1=iris_train.iloc[:split_x1,:-1].append(iris_train.iloc[split_x2:,:-1])
#     y_train_1=iris_train.iloc[:split_x1,-1].append(iris_train.iloc[split_x2:,-1])
#     X_validate=iris_train.iloc[split_x1:split_x2,:-1]
#     y_validate=iris_train.iloc[split_x1:split_x2,-1]
#     max_accuracy=0
#     rg_value=0.01
#     print("\nRound at {} ".format(i))
#     for j in range(1,101):
#         if(j%20==0):
#             print("Going at {} ".format(j),end=" ")
#         model=LogisticRegression()
#         model.fit_autograd(X_train_1,y_train_1,batch_size=10,n_iter=1000,regularise='l1',regularise_value=j*rg_value)
#         y_hat=model.predict(X_validate)
#         score=accuracy(y_hat,y_validate)
#         if(max_accuracy<score):
#             opt_rg[i]["RegularizeValue"]=j*rg_value
#             opt_rg[i]["Accuracy"]=score
#             max_accuracy=score  
# print("For L1 Regularization")                                
# for i in range(len(opt_rg)):
#     print("At fold ={}:\n\tAccuracy={} for Regularize Value={}".format(i,opt_rg[i]["Accuracy"],opt_rg[i]["RegularizeValue"]))


# #For l2 regularization
# fold=int(len(iris_train)*0.2)
# opt_rg={0:dict(),1:dict(),2:dict(),3:dict(),4:dict()}
# for i in range(5):
#     split_x1=i*fold
#     split_x2=split_x1+fold
#     X_train_1=iris_train.iloc[:split_x1,:-1].append(iris_train.iloc[split_x2:,:-1])
#     y_train_1=iris_train.iloc[:split_x1,-1].append(iris_train.iloc[split_x2:,-1])
#     X_validate=iris_train.iloc[split_x1:split_x2,:-1]
#     y_validate=iris_train.iloc[split_x1:split_x2,-1]
#     max_accuracy=0
#     rg_value=0.01
#     print("\nRound at {} ".format(i))
#     for j in range(1,101):
#         if(j%20==0):
#             print("Going at {} ".format(j),end=" ")
#         model=LogisticRegression()
#         model.fit_autograd(X_train_1,y_train_1,batch_size=10,n_iter=1000,regularise='l2',regularise_value=j*rg_value)
#         y_hat=model.predict(X_validate)
#         score=accuracy(y_hat,y_validate)
#         if(max_accuracy<score):
#             opt_rg[i]["RegularizeValue"]=j*rg_value
#             opt_rg[i]["Accuracy"]=score
#             max_accuracy=score  
# print("For L2 Regularization")                                
# for i in range(len(opt_rg)):
#     print("At fold ={}:\n\tAccuracy={} for Regularize Value={}".format(i,opt_rg[i]["Accuracy"],opt_rg[i]["RegularizeValue"]))


# #For impotance of feature using L1 Regularization
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]

model=LogisticRegression()
model.fit_autograd(X_train,y_train,n_iter=2000,batch_size=1,regularise='l1',regularise_value=0.05)
y_hat=model.predict(X_test)
print("With all Features: ")
print("Coefficent: {}".format(model._LogisticRegression__coef))
print("Accuracy:  ",accuracy(y_hat,y_test))

#Removing features to know the importance
Feature_list=[0,1,2,3]
for i in range(4):
    c=Feature_list.copy()
    c.remove(i)
    X_train=iris.iloc[:split_at,c]
    y_train=iris.iloc[:split_at,-1]
    X_test=iris.iloc[split_at:,c]
    y_test=iris.iloc[split_at:,-1]

    model=LogisticRegression()
    model.fit_autograd(X_train,y_train,n_iter=2000,batch_size=1,regularise='l1',regularise_value=0.05)
    y_hat=model.predict(X_test)
    print("\nRemoving feature number: {}".format(i+1))
    print("Coefficent: {}".format(model._LogisticRegression__coef))
    print("Accuracy:  ",accuracy(y_hat,y_test))