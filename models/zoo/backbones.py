import torch
import torch.nn as nn
from functools import partial
from torchvision.models import resnet18, resnet50


class EncoderResNet(nn.Module):
    def __init__(self, resnet, pretrained):
        super().__init__()
        self.tv_resnet = resnet(pretrained)
        
    def forward_features(self, x):
        x = self.tv_resnet.conv1(x)
        x = self.tv_resnet.bn1(x)
        x = self.tv_resnet.relu(x)
        x = self.tv_resnet.maxpool(x)

        x = self.tv_resnet.layer1(x)
        x = self.tv_resnet.layer2(x)
        x = self.tv_resnet.layer3(x)
        x = self.tv_resnet.layer4(x)
        return x
    
    
encoder_params = {   
    "resnet18": {
        "features": 512,
        "init_op": partial(EncoderResNet, resnet=resnet18, pretrained=True)
    },
    
    "resnet50": {
        "features": 2048,
        "init_op": partial(EncoderResNet, resnet=resnet50, pretrained=True)
    }
}


def build_encoder(encoder_name):
    return encoder_params[encoder_name]