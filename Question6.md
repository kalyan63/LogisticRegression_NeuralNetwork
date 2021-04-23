## Results of Neural Network for Classification and regression

1. **Classification of digts data set of Sklearn**
    
    > Just used input layer and a output layer with sigmoid activation and loss='Cross Entropy'

    > Results : 

        > Accuracy for iteratin= 10  is: 0.6787204450625869
        > Accuracy for iteratin= 50  is: 0.9068150208623088
        > Accuracy for iteratin= 100  is: 0.9541029207232267
        > Accuracy for iteratin= 1000  is: 0.9624478442280946
        > Accuracy for iteratin= 2000  is: 0.9582753824756607
    
    > Same as above but with loss='mae'

    >Results: 

        > Accuracy for iteratin= 10  is: 0.09040333796940195
        > Accuracy for iteratin= 50  is: 0.07093184979137691
        > Accuracy for iteratin= 100  is: 0.20584144645340752
        > Accuracy for iteratin= 1000  is: 0.16968011126564672
        > Accuracy for iteratin= 2000  is: 0.3616133518776078

2. **Regression problem on Boston data set from skleran**

    > Hidden Layer = [1] and activation is relu and loss='MAE'

    > Result:- 

        > RMSE for iteration= 10 is 8.733211478880518
          MAE  for iteration= 10 is 5.830269924964247
        > RMSE for iteration= 50 is 8.030399881944936
          MAE  for iteration= 50 is 5.455363454935709
        > RMSE for iteration= 1000 is 6.8656421602757725
          MAE  for iteration= 1000 is 5.101458290547767
        > RMSE for iteration= 2000 is 5.955320729365012
          MAE  for iteration= 2000 is 4.987039022588076

    > Hidden Layer =[50,1] and activation is ['relu','relu'] and loss='MAE'

    > Result:-      

        > RMSE for iteration= 2 is 16.40981588098888
          MAE  for iteration= 2 is 13.29089035668089
        > RMSE for iteration= 50 is 6.785571496506081
          MAE  for iteration= 50 is 4.503286818416826
        > RMSE for iteration= 1000 is 3.6588620306523714
          MAE  for iteration= 1000 is 2.4505244931793446
        > RMSE for iteration= 2000 is 25.309300457641505
          MAE  for iteration= 2000 is 23.413793103448278

    > Hidden Layer =[10,5,1] and activation is sigmoid,relu,relu and loss='MAE' and lr=0.0005

    > Result:- 

        > RMSE for iteration= 1 is 18.033888671617913
          MAE  for iteration= 1 is 14.972749757263301
        > RMSE for iteration= 2 is 14.6481992843915
          MAE  for iteration= 2 is 11.30061154592307
        > RMSE for iteration= 50 is 8.23222116977498
          MAE  for iteration= 50 is 6.737612219239857
        > RMSE for iteration= 1000 is 4.358514083392962
          MAE  for iteration= 1000 is 3.2197795068723223
        > RMSE for iteration= 2000 is 4.742400998215945
          MAE  for iteration= 2000 is 2.982917454538722395     
