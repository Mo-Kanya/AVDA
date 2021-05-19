import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,path=None):
        if path is None:
            name='result/best_model.pth'
            torch.save(self.state_dict(),name)
            return name
        else:
            torch.save(self.state_dict(),path)
            return path

class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,input_features=64):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features,10)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)

class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,64)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)

        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)

        return out
