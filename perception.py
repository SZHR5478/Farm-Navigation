import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from utils import weights_init


class CNN_simple(nn.Module):
    def __init__(self, obs_shape):
        super(CNN_simple, self).__init__()
        in_channels = obs_shape[0]
        if len(obs_shape) == 4:
            in_channels = obs_shape[1]
        self.conv1 = nn.Conv3d(in_channels, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv3d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(1, *obs_shape))
        out = self.forward(dummy_state)
        self.outdim = out.size(-1)
        self.apply(weights_init)
        self.train()

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(1, -1)
        return x

