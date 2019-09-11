import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullyConnecteEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4_mean = nn.Linear(250, output_size)
        self.fc4_logsd = nn.Linear(250, output_size)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(250)

    def forward(self, x):
        x=x.view(-1,self.input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        #mean = torch.tanh(self.fc4_mean(x))
        mean = self.fc4_mean(x)
        logsd = self.fc4_logsd(x)
        return mean, logsd

class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_width, input_height, input_channel, output_size):
        super(ConvolutionalEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_channel, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear((input_width-6)*(input_height-6)*8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, output_size)
        self.fc_logsd = nn.Linear(256, output_size)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, (self.input_height-6)*(self.input_width-6)*8)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        mean = self.fc_mean(x)
        logsd = self.fc_logsd(x)

        return mean, logsd

class FullyConnecteDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        #x = torch.sigmoid(self.fc5(x))
        x = self.fc5(x)
        return x
