import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.BasicModule import BasicModule


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class DCD(nn.Module):
    def __init__(self, h_features=2048, input_features=2048 * 2):
        super(DCD, self).__init__()

        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, 1024)
        self.fc3 = nn.Linear(1024, 4)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        return F.softmax(self.fc3(out), dim=1)


class Attention(BasicModule):
    def __init__(self, h_features=4096, input_features=2048):
        super(Attention, self).__init__()

        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features, input_features)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out
        # return F.InstanceNorm1d(out)

class Classifier(BasicModule):
    def __init__(self, input_features=2048):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 31),
        )

    def forward(self, input):
        return self.classifier(input)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load('./models/resnet50-19c8e357.pth'))
        # resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = out.view(out.size(0), -1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
