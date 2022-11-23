from torch import nn
import torch.nn.functional as F

class Net2nn(nn.Module):
    def __init__(self, input_neurons):
        super(Net2nn, self).__init__()
        self.fc1=nn.Linear(input_neurons,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x