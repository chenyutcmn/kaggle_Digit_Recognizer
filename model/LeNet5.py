import torch as t
from torch import nn


class LeNet5(t.nn.Module):
    def __init__(self):
        super(LeNet5 , self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels = 1 , out_channels = 6 , kernel_size = (5 , 5) , padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2 , 2) , stride=2),
            nn.Conv2d(6 , 16 , kernel_size = (5 , 5)),
            nn.MaxPool2d(kernel_size = (2 , 2) , stride=2),
            nn.Conv2d(16 , 120 , kernel_size = (5 , 5)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(120 , 84),
            nn.ReLU(),
            nn.Linear(84 , 10),
            nn.LogSigmoid()
        )

    def forward(self , img):
        output = self.convnet(img)
        output = output.view(img.size(0) , -1)
        output = self.fc(output)
        return output