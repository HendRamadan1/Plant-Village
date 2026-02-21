import torch
import torch.nn as nn 
from torchvision import models

class PlantResNet50(nn.Module):
    def __init__(self, num_classes=15):
        super(PlantResNet50, self).__init__()
        

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT,)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            

        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


model_resent=PlantResNet50(num_classes=15)