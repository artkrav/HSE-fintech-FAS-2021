import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones import EncoderResNet, build_encoder
    
    
class BaselineClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate):
        super().__init__()
        self.bottleneck_features = build_encoder(encoder)["features"]
        self.encoder = build_encoder(encoder)["init_op"]()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_features, self.bottleneck_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bottleneck_features // 4, 1)
        )
        
    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pooling(x).flatten(1)
        cl_out = self.classifier(x)
        return cl_out