import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .backbones import EncoderResNet, build_encoder
from .revgrad import GradientReversal


class DiscriminatorBlock(nn.Module):
    def __init__(self, features_in, features_out, dropout_rate):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(features_in, features_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features_in // 2, features_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features_in // 2, features_out)
        )
        
    def forward(self, x):
        x = self.discriminator(x)
        return x
    
    
class DAFL_FAS(nn.Module):
    def __init__(self, encoder, dropout_rate, revgrad_lamba, num_domains, single_discriminator=False):
        super().__init__()
        self.single_discriminator = single_discriminator
        self.bottleneck_features = build_encoder(encoder)["features"]
        self.encoder = build_encoder(encoder)["init_op"]()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_features, self.bottleneck_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bottleneck_features // 4, 1)
        )
        self.revgrad = GradientReversal(revgrad_lamba)
        if self.single_discriminator:
            self.dd = DiscriminatorBlock(self.bottleneck_features, num_domains, dropout_rate)
        else:
            self.ccdd_live = DiscriminatorBlock(self.bottleneck_features, num_domains, dropout_rate)
            self.ccdd_spoof = DiscriminatorBlock(self.bottleneck_features, num_domains, dropout_rate)
        
    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pooling(x).flatten(1)
        cl_out = self.classifier(x)
        revgrad_out = self.revgrad(x)
        if self.single_discriminator:
            dd_out = self.dd(revgrad_out)
            return cl_out, dd_out
        ccdd_live_out = self.ccdd_live(revgrad_out)
        ccdd_spoof_out = self.ccdd_spoof(revgrad_out)
        return cl_out, ccdd_live_out, ccdd_spoof_out
    
    
class DAFL_FAS_mixup(nn.Module):
    def __init__(self, encoder, dropout_rate, revgrad_lamba, num_domains):
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
        self.revgrad = GradientReversal(revgrad_lamba)
        self.ccdd_live = DiscriminatorBlock(self.bottleneck_features, num_domains, dropout_rate)
        self.ccdd_spoof = DiscriminatorBlock(self.bottleneck_features, num_domains, dropout_rate)
        
    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pooling(x).flatten(1)
        cl_out = self.classifier(x)
        return cl_out, x
    
    def forward_mix_live(self, x1, x2, lam):
        x = lam * x1 + (1.0 - lam) * x2
        revgrad_out = self.revgrad(x)
        ccdd_live_out = self.ccdd_live(revgrad_out)
        return ccdd_live_out
    
    def forward_mix_spoof(self, x1, x2, lam):
        x = lam * x1 + (1.0 - lam) * x2
        revgrad_out = self.revgrad(x)
        ccdd_spoof_out = self.ccdd_spoof(revgrad_out)
        return ccdd_spoof_out