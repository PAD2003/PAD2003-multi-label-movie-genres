import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch

class CustomResnet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # features extractor
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        # for param in self.features[-2][1].parameters():
        #     param.requires_grad = True
        
        # output layer
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x)