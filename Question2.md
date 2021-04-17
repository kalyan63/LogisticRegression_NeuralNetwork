## Results of Question 2: 

1. **Result of L1 Regularization after Nested Cross Validation**
2. **Resukt of L2 Regularization after Nested Cross Validation**
3. **Undersatnding Feature Imporatnce**

    > For batch size=10 and iteration =1000 for L1 regularization the results are: 

        >   Coefficent: [ 0.36142426 -0.95850064 -0.46258305  0.95415171  1.29136658] 
            Accuracy:   0.9666666666666667

    > Now by removing feature with coefficent=1.291     

        >   Coefficent: [ 0.34566245 -0.89190234 -0.51768293  1.34452634]
            Accuracy:   0.9

    > Now by removing feature with coefficent=-0.46

        >   Coefficent: [ 0.31870442 -1.14394444  0.93094037  1.2900311 ]
            Accuracy:   0.9333333333333333   

    > **Here we can see the Feature importance by the value of coefficent. The less important feature would be close to Zero.**        