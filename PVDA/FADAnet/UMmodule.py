import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

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
    def __init__(self,input_features=84):
        super(Classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, 10),
        )

    def forward(self,input):
        return F.softmax(self.classifier(input),dim=1)

class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Flatten(),
            # nn.Linear(4608, 120),
            nn.Linear(1152, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True)
        )

    def forward(self,input):
        return self.feature(input)  
