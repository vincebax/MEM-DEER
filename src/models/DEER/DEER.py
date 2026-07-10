import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import ImageEncoder, ScanpathEncoder, Fusion, TaskClassifier

# TODO change to appropriate values
SCANPATH_INPUT_DIM = 4
SCANPATH_OUTPUT_DIM = 128
NUM_CATEGORIES = 2

class DEER(nn.Module):
    def __init__(self):

        super().__init__()

        self.image_encoder = ImageEncoder()
        self.scanpath_encoder = ScanpathEncoder(SCANPATH_INPUT_DIM, SCANPATH_OUTPUT_DIM)
        self.fusion = Fusion()
        self.task_classifier = TaskClassifier(NUM_CATEGORIES)
    
    def freeze_module(self, module_name):

        assert hasattr(self, module_name), f'Module "{module_name}" does not exist'

        module = getattr(self, module_name)
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module_name):

        assert hasattr(self, module_name), f'Module "{module_name}" does not exist'

        module = getattr(self, module_name)
        for param in module.parameters():
            param.requires_grad = True
    
    def forward(self, image, scanpath):

        image_features = self.image_encoder(image)
        scanpath_features = self.scanpath_encoder(scanpath)

        fused_features = self.fusion(image_features, scanpath_features)

        return self.task_classifier(fused_features)