import autograd.numpy as np
from autograd.numpy import exp,log,abs
from autograd import elementwise_grad as grad1

np.random.seed(24)
#Activation Functions
def relu(a):
    return np.where(a>0,a,0)
def identity(a):
    return a
def sigmoid(a):
    return 1.0/(1.0+exp(a))    

activation_main={'r':relu,'i':identity,'s':sigmoid}
#End of Activation functions

#Loss funtions
def mae(W,A,layer,activation,y):
    for i in range(1,len(layer)):
        A['z_'+str(i)]=np.matmul(A['a_'+str(i-1)],W['w_'+str(i)].T)+np.tile(W['b_'+str(i)],(len(A['a_0']),1))
        A['a_'+str(i)]=activation_main[activation[i-1]](A['z_'+str(i)])
    return 1*np.sum(abs(y-A['a_'+str(len(layer)-1)]))
MAE=grad1(mae)

def cross_entropy(W,A,layer,activation,y):
    for i in range(1,len(layer)):
        A['z_'+str(i)]=np.matmul(A['a_'+str(i-1)],W['w_'+str(i)].T)+np.tile(W['b_'+str(i)],(len(A['a_0']),1))
        A['a_'+str(i)]=activation_main[activation[i-1]](A['z_'+str(i)])
    return -(np.sum(y*log(A['a_'+str(len(layer)-1)])+(1-y)*log(1-A['a_'+str(len(layer)-1)])))    
#End of Loss functions
CrossEntropy=grad1(cross_entropy)
class NN():
    def __init__(self):   
        self.WB={}
        self.activation=None
        self.AZ={}
        self.layers=None

    def add_layers(self,input_size,hiden_size,activation):
        '''input_size is len of input layer
        hiden_layer is the list of number of neurons in each layers and 
        activation is array with input \'i\' for Identity \'r\' for Relu \'s\' for sigmoid activation for each layer'''
        assert(len(hiden_size)==len(activation))
        assert(set(activation).issubset(set(['i','r','s'])))
        self.activation=activation
        self.layers=[input_size]
        self.layers.extend(hiden_size)
        for k in range(1,len(self.layers)):
            self.WB['w_'+str(k)]=np.array([[np.random.randn() for i in range(self.layers[k-1])] for j in range(self.layers[k])])
            self.WB['b_'+str(k)]=np.array([np.random.randn() for i in range(self.layers[k])])

    def fit(self,X,y,lr=0.01,n_iter=100,loss='Cross_entropy'):
        '''loss is Mean square error if loss=\'mse\' else any other input makes it Cross Entropy'''
        self.forward_prop(X)
        for i in range(n_iter):
            self.back_prop(y,lr,loss)

    def forward_prop(self,X):
        self.AZ['a_0']=np.array(X)
        for i in range(1,len(self.layers)):
            self.AZ['z_'+str(i)]=np.matmul(self.AZ['a_'+str(i-1)],self.WB['w_'+str(i)].T)+np.tile(self.WB['b_'+str(i)],(len(self.AZ['a_0']),1))
            self.AZ['a_'+str(i)]=activation_main[self.activation[i-1]](self.AZ['z_'+str(i)])
        return self.AZ['a_'+str(len(self.layers)-1)]

    def back_prop(self,y,lr,loss):
        Trained_weight=None
        if(loss=='mae'):
            Trained_weight=MAE(self.WB,self.AZ,self.layers,self.activation,y)
        else:
            Trained_weight=CrossEntropy(self.WB,self.AZ,self.layers,self.activation,y)   
        for i in range(1,len(self.layers)):
            self.WB['w_'+str(i)]-=lr*Trained_weight['w_'+str(i)]
            self.WB['b_'+str(i)]-=lr*Trained_weight['b_'+str(i)]

    def predict(self,X):
        return self.forward_prop(X)

