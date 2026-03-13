import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
 
NUM_CLASSES = 2   # NORMAL, PNEUMONIA
 
def build_densenet121(pretrained: bool = True, freeze_layers: bool = True) -> nn.Module:
    '''
    Load DenseNet121 pretrained on ImageNet.
    Replace final classifier with a 2-class head.
    Args:
        pretrained: Load ImageNet weights (True for transfer learning)
        freeze_layers: Freeze all layers except final dense block + classifier
    '''
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
 
    if freeze_layers:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last dense block (denseblock4) and classifier
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
 
    # Replace classifier: DenseNet121 has 1024 output features
    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    return model
 
def build_resnet50(pretrained: bool = True) -> nn.Module:
    '''Alternative model for comparison experiments.'''
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    return model
 
def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    '''Lightweight alternative - good for CPU inference deployment.'''
    from torchvision.models import EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model
 
MODEL_REGISTRY = {
    'densenet121': build_densenet121,
    'resnet50': build_resnet50,
    'efficientnet_b0': build_efficientnet_b0,
}
