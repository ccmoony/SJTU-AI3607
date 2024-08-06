import jittor as jt
import numpy as np
from jittor import nn, Module
import matplotlib.pyplot as plt

class LinearRegression(Module):
    def __init__(self):
        self.layer1 = nn.Linear(2, 1)
    def execute (self,x) :
        y_pred = self.layer1(x)
        return y_pred
    
def get_data(data_size, plot=False):  # x[area,age] (k m^2,10year)
    x = np.zeros((data_size,2))
    x[:,0] = 0.05 + np.random.rand(data_size)
    x[:,1] = 3*np.random.rand(data_size)+ 0.1
    W = np.array([2,-1])
    y = np.dot(x,W) + 4 + 0.1*np.random.randn(data_size) # add noise
    y = np.reshape(y,(-1,1)) 
    if plot:
        plt.scatter(x[:,0], x[:,1], s=y*100, c=3*y, alpha=0.4, edgecolors="grey",linewidth=2)
        plt.title('Housing Price')
        plt.xlabel('Housing Area/k m^2')
        plt.ylabel('Age/10years')
        plt.show()
    return jt.float32(x), jt.float32(y)

def Train(Train_size,num_epoch):
    Train_x, Train_y = get_data(Train_size)
    model = LinearRegression()
    loss_function = nn.MSELoss()
    optimizer = nn.SGD(model.parameters(), lr=1e-1)
    loss_Train = []
    for i in range(num_epoch):
        pred_y = model(Train_x)
        loss = loss_function(pred_y , Train_y)
        optimizer.zero_grad()
        optimizer.step(loss)
        loss_Train.append(loss.numpy().sum())
        print(f"step {i}, loss = {loss.numpy().sum()}")
    print(model.parameters())
    x = [i for i in range(1,1+num_epoch)]
    plt.plot(x,loss_Train, color='b')
    plt.xlabel('epoch')
    plt.ylabel('MSEloss')
    plt.show()
    return model

if __name__ == "__main__":
    Train_size = 700
    Test_size = 300
    model = Train(Train_size,1000)
    Validation_x, Validation_y = get_data(Test_size)
    pred_y = model(Validation_x)
    x = [i for i in range(1,Test_size+1)]
    plt.plot(x,Validation_y.numpy(),label='Truth')
    plt.plot(x,pred_y.numpy(),label='Prediction')
    plt.xlabel('House number')
    plt.ylabel('Price/million')
    plt.legend()
    plt.show()