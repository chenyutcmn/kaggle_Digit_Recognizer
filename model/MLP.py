import torch as t
from torch import nn
class MLP(t.nn.Module):
    def __init__(self):
        super(MLP , self).__init__()
        self.hind = nn.Sequential(
            nn.Linear(784 , 196),
            nn.ReLU(inplace=True),
            nn.Linear(196 , 49),
            nn.ReLU(inplace=True),
            nn.Linear(49 , 10)
        )
    
    def forward(self , x):
        out = self.hind(x)
        return out