import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layers=[2, 7, 16, 25, 34], use_bn=False):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            vgg = models.vgg19_bn(pretrained=True)
        else:
            vgg = models.vgg19(pretrained=True)

        self.feature_layers = feature_layers
        self.features = vgg.features

        for param in self.features.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features