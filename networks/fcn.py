import torch
import torch.nn as nn
from .common import *


def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add(nn.ReLU6())
    #
    model.add(nn.Linear(num_hidden, num_output_channels))
    #    model.add(nn.ReLU())
    model.add(nn.Softmax())
    #
    return model


"""
    全连接层
"""


class FcnNet(nn.Module):

    def __init__(self, input_channels=200, output_channels=1, num_hidden=1000):
        super(FcnNet, self).__init__()
        self.liner1 = nn.Linear(input_channels, num_hidden, bias=True)
        self.relu = nn.ReLU6()
        self.liner2 = nn.Linear(num_hidden, output_channels)
        self.softmax = nn.Softmax()

    def forward(self, input):
        l1 = self.relu(self.liner1(input))
        l2 = self.softmax(self.liner2(l1))
        return l2
