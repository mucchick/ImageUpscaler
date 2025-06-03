import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, stride=1, use_spectral_norm=True):
            layers = []
            conv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1)

            # Apply spectral normalization for training stability
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)

            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, stride=1, use_spectral_norm=False),
            *block(64, 64, stride=2),
            *block(64, 128, stride=1),
            *block(128, 128, stride=2),
            *block(128, 256, stride=1),
            *block(256, 256, stride=2),
            *block(256, 512, stride=1),
            *block(512, 512, stride=2),
        )

        # Final layer with spectral norm
        final_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.final_conv = nn.utils.spectral_norm(final_conv)

    def forward(self, img):
        features = self.model(img)
        return self.final_conv(features)