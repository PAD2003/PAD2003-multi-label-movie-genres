import torch
import torch.nn as nn
from torchvision import models

class CustomVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained VGG19 model with batch normalization
        vgg19_bn = models.vgg19_bn(pretrained=True)

        # Freeze all layers
        for param in vgg19_bn.parameters():
            param.requires_grad = False

        # Extract features from VGG19
        self.features = vgg19_bn.features

        # Define custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Replace first FC layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # Replace last FC layer with num_classes output
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)