import torch
import torch.nn as nn
import torchvision.models as models

# Using pre-trained ResNet model
class ResNetTransferLearning(nn.Module):
    def __init__(self):
        super(ResNetTransferLearning, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        return self.resnet(x)

# Example usage in training script
# model = ResNetTransferLearning().to(device)
