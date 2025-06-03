import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_channels = growth_channels
        self.in_channels = in_channels

        for i in range(5):
            self.layers.append(
                nn.Conv2d(in_channels + i * growth_channels, growth_channels,
                         kernel_size=3, padding=1)
            )

        self.final = nn.Conv2d(in_channels + 5 * growth_channels, in_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        inputs = x
        features = [x]

        for conv in self.layers:
            concat_feat = torch.cat(features, 1)
            out = self.relu(conv(concat_feat))
            features.append(out)

        concat_feat = torch.cat(features, 1)
        out = self.final(concat_feat)
        return out * 0.2 + x

class ChannelAttention(nn.Module):
    """Channel attention mechanism for enhanced feature representation"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Both average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.rdb1 = DenseBlock(channels)
        self.rdb2 = DenseBlock(channels)
        self.rdb3 = DenseBlock(channels)

        # Add channel attention
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.ca(out)  # Apply channel attention
        return out * 0.2 + x

class ESRGANGenerator(nn.Module):
    def __init__(self, num_rrdb=16, scale=4):
        super(ESRGANGenerator, self).__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)

        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb)])
        self.conv_trunk = nn.Conv2d(64, 64, 3, 1, 1)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final conv
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_trunk(self.rrdb_blocks(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        out = self.conv_last(out)
        return out