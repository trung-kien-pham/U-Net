import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super().__init__()

        if bn:
            self.doubleconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        else:
            self.doubleconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.doubleconv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super().__init__()

        init_channels = 64
        self.out_channels = out_channels

        self.encoder1 = ConvBlock(in_channels, init_channels, bn)
        self.encoder2 = ConvBlock(init_channels, 2*init_channels, bn)
        self.encoder3 = ConvBlock(2*init_channels, 4*init_channels, bn)
        self.encoder4 = ConvBlock(4*init_channels, 8*init_channels, bn)

        self.bottleneck = ConvBlock(8*init_channels, 16*init_channels, bn)

        self.decoder1 = ConvBlock((16+8)*init_channels, 8*init_channels, bn)
        self.decoder2 = ConvBlock((8+4)*init_channels, 4*init_channels, bn)
        self.decoder3 = ConvBlock((4+2)*init_channels, 2*init_channels, bn)
        self.decoder4 = ConvBlock((2+1)*init_channels, init_channels, bn)

        self.final_conv = nn.Conv2d(init_channels, out_channels, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))

        btn = self.bottleneck(self.maxpool(e4))

        d1 = self.decoder1(torch.cat([self.upsample(btn), e4], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.decoder4(torch.cat([self.upsample(d3), e1], dim=1))

        output = self.final_conv(d4)

        return output
