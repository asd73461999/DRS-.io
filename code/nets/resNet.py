import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from torchvision import models

nonlinearity = partial(F.relu,inplace=True)

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self,in_ch,n_filters):
        super(DecoderBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,in_ch//4,1)
        self.norm1 = nn.BatchNorm2d(in_ch//4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_ch//4,in_ch//4,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_ch//4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_ch//4,n_filters,1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
class resnet34_unet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, pretrained=False):
        super(resnet34_unet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)