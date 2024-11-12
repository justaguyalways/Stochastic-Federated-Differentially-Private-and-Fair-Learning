import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models

class LogisticRegression(nn.Module):
    def __init__(self, input_num_attr):
        super().__init__()
        self.layer = nn.Linear(input_num_attr, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer(x)
        return x, self.sigmoid(x)

class NeuralNetwork(nn.Module):
    def __init__(self, input_num_attr, num_output, num_layers = 2):
        super().__init__()
        if num_layers < 2:
            raise Exception("For the neural network, please enter number of layers greater than 1")
        
        modules = []
        for i in range(num_layers - 1):
            modules.append((f'layer_{i+1}', nn.Linear(input_num_attr, input_num_attr)))
            modules.append((f'relu_{i+1}', nn.LeakyReLU(inplace = True)))
        modules.append((f'layer_{num_layers}', nn.Linear(input_num_attr, num_output)))
        
        self.model = nn.Sequential(OrderedDict(modules))
    
    def forward(self, x):
        out = self.model(x)
        return out, out

class FeedForward(nn.Module):
    def __init__(self, classes = 1):
        super(FeedForward, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.inst1_1 = nn.InstanceNorm2d(num_features=16, track_running_stats=True)
        self.inst1_2 = nn.InstanceNorm2d(num_features=16, track_running_stats=True)
        
        self.inst2_1 = nn.InstanceNorm2d(num_features=64, track_running_stats=True)
        self.inst2_2 = nn.InstanceNorm2d(num_features=64, track_running_stats=True)

        self.inst3_1 = nn.InstanceNorm2d(num_features=256, track_running_stats=True)
        self.inst3_2 = nn.InstanceNorm2d(num_features=256, track_running_stats=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=65536, out_features=classes, bias=True)
        self.midpoints = torch.tensor([[(i+5)/120 for i in range(0, 120, 10)]]).cuda()

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.inst1_1(self.conv1_2(x)))
        x = self.relu(self.inst1_2(self.conv1(x)))

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.inst2_1(self.conv2_2(x)))
        x = self.relu(self.inst2_2(self.conv2(x)))

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.inst3_1(self.conv3_2(x)))
        x = self.relu(self.inst3_2(self.conv3(x)))

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        logits = self.fc1(x)
        out = self.softmax(logits)
        return logits, out