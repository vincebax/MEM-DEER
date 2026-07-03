import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImageEncoder(nn.Module):
    """
    Input : b x 3 x h x w
    Output : b x img_d

    """

    def __init__(self):
        super().__init__()

        self.backbone = models.densenet201(weights='DenseNet201_Weights.IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()

    def forward(self, image):
        return self.backbone(image)

class ScanpathEncoder(nn.Module):
    """
    Input : b x t x scanpath_feat
    Output : b x scanpath_d

    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, scanpath):
        _, yhat = self.model(scanpath)
        return yhat[-1]

class Fusion(nn.Module):
    """
    Input : 
        image_features : b x img_d
        scanpath_features : b x scanpath_d

    Output : b x (img_d + scanpath_d)

    """

    def __init__(self):
        super().__init__()

    def forward(self, image_features, scanpath_features):
        return torch.cat([image_features, scanpath_features], dim=1)

class TaskClassifier(nn.Module):
    """
    Input : b x (img_d + scanpath_d)
    Output : b x num_categories

    """

    def __init__(self, num_categories):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(num_categories)
        )

    def forward(self, fused_features):
        return self.net(fused_features)