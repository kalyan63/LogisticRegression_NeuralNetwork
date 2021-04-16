## Results for Question 1

1. **Result for Vectorized Logistic Regression on Breast Cancer Data:** 
    > For Vectorized and Batch size=100 (split=0.6)
        
        > Accuracy for iteration=100 is: 0.32894736842105265 
        > Accuracy for iteration=1000 is: 0.9429824561403509 
        > Accuracy for iteration=2000 is: 0.8771929824561403 
        > Accuracy for iteration=3000 is: 0.8070175438596491 
        > Accuracy for iteration=10000 is: 0.9517543859649122 

2. **Result for Autograd Logistic Regression on Breast Cancer Data:**
    >For Autograd and Batch size=300 and autograd (Just considered 6 features due to autograd limit)

        > Accuracy for iteration=100 is: 0.618421052631579
        > Accuracy for iteration=1000 is: 0.6228070175438597
        > Accuracy for iteration=5000 is: 0.75
        > Accuracy for iteration=20000 is: 0.8114035087719298
        > Accuracy for iteration=30000 is: 0.8157894736842105

3. **Plot for Decision boundry for breast cancer data**
    > Using vectorized: 
        > Accuracy (iter=10000,batch=200) is : **0.89**
        > !['Image of vec'](q1a1.png)

    > Using Autograd  :
        > Accuracy (iter=30000,batch=300) is :  **0.794**
        > !['Image'](q1b1.png)

4. **Plot for Decision boundry for iris data on Sepal and Petal width**
    
    > Accuarcy for split = 0.6 (Train set) is: **0.93** and **0.984**

    > Plot is: 

    > !['Image of Decision Boundry'](q1a.png) 

    > !['Image of Decision Boundry'](q1b.png)