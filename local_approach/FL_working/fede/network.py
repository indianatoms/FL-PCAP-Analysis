from torch import nn
import torch.nn.functional as F
import math

class Net2nn(nn.Module):
    def __init__(self, input_neurons):
        hidden_neurons = math.floor((input_neurons+2)/2)
        super(Net2nn, self).__init__()
        self.fc1=nn.Linear(input_neurons,hidden_neurons)
        self.fc2=nn.Linear(hidden_neurons,2)
        # self.fc3=nn.Linear(10,2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
        x=self.fc2(x)
        return x