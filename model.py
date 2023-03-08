import torch
from torch import nn


class yolov3(nn.Module):
    def __init__(self):
        super(yolov3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(3, 32, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv6 = nn.Conv2d(64, 64, 1)
        self.conv7 = nn.Conv2d()

    def forward(self, x0):
        x = self.conv1(x0)
        x = self.conv2(x)
        x1 = self.conv3(x0)
        x1= self.conv4(x1)
        x = x + x1
        x = self.conv5(x)

        return x
