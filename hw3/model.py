import pickle
import jittor as jt
import numpy as np
from jittor import Module, nn
import pygmtools as pygm
pygm.set_backend('jittor')

class DeepPermNet(Module):
    def __init__(self, patch_size= (16,16)):
        super(DeepPermNet, self).__init__()
        self.patch_size = patch_size
        self.patch_num = 32*32//(patch_size[0]*patch_size[1])
        self.layer1 = nn.Sequential(
                nn.Conv2d(3,32,(5,5),padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32,16,(3,3),padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,64,(5,5),padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*patch_size[0]*patch_size[1]//16, 512),
                nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.patch_num*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.patch_num**2)
        )
    def execute(self,x):
        batch_size = x.shape[0]
        x = x.view(-1, *x.shape[2:])
        x = self.layer1(x)
        x = x.view(batch_size,-1)
        x = self.layer2(x)
        x = x.view(batch_size,self.patch_num,self.patch_num)
        x = pygm.sinkhorn(x)
        return x
'''nn.Conv2d(3,32,(5,5),padding=2),
nn.ReLU(),
nn.Conv2d(32,16,(3,3),padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(16,64,(5,5),padding=2),
nn.ReLU(),
nn.Conv2d(64,16,(3,3),padding=1),
nn.MaxPool2d(2),
nn.ReLU(),
nn.Flatten(),
nn.Linear(16*patch_size[0]*patch_size[1]//16,512)'''
    





