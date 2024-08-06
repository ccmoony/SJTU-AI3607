import pickle
import jittor as jt
import numpy as np
from jittor import Module, nn
from jittor.models.resnet import Bottleneck, ResNet
from jittor.models import resnet50

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(3,32,(5,5),padding=2),
                nn.ReLU(),
                nn.Conv2d(32,16,(3,3),padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,64,(5,5),padding=2),
                nn.ReLU(),
                nn.Conv2d(64,16,(3,3),padding=1),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16*8*8,10)
        )
    def execute(self,x):
        return self.layer(x)

class RNN(Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layer1 = nn.RNN(96,512,2)
        self.layer2 = nn.Linear(512,10)
        
    def execute(self,x):
        out = x.permute(2,0,1)
        out, _ = self.layer1(out)
        out = out[-1, :, :]
        out = self.layer2(out)  # 取序列中的最后一个输出
        return out
    
class Resnet(Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.layer1 = resnet50(pretrained=True)
        self.layer2 = nn.Linear(1000,10)
        
    def execute(self,x):
        out = self.layer1(x)
        return self.layer2(out)