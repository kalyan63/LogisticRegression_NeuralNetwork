import numpy as np
def accuracy(y_hat, y):
    assert(y_hat.size == y.size)
    pridct=list(y_hat)
    Gtruth=list(y)
    correct_pridiction=0
    total=len(pridct)
    for i in range(len(pridct)):
        if(pridct[i]==Gtruth[i]):
            correct_pridiction+=1
    return correct_pridiction/total        

def precision(y_hat, y, cls):
    predict=list(y_hat)
    Gtruth=list(y)
    Trueclass=0
    allclass=0
    for i in range(len(predict)):
        if(predict[i]==cls):
            if(predict[i]==Gtruth[i]):
                Trueclass+=1
            allclass+=1
    return Trueclass/allclass            

def recall(y_hat, y, cls):
    predict=list(y_hat)
    Gtruth=list(y)
    Trueclass=0
    allclass=0
    for i in range(len(predict)):
        if(Gtruth[i]==cls):
            if(predict[i]==Gtruth[i]):
                Trueclass+=1
            allclass+=1
    return Trueclass/allclass
