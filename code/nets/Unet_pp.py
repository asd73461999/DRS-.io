from torch import nn
from torch.nn import functional as F
import torch

#double convolutional layer
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

#network block
class NestedUNet(nn.Module):
    def __init__(self,args,in_ch,out_ch):
        super(NestedUNet,self).__init__()
        nb_filter=[32,64,128,256,512]
        self.args=args
        self.Maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_ch, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()

        if self.args.deep_supervision == True:
            self.final1 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.Maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], 1))

        x2_0 = self.conv2_0(self.Maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.Maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], 1))

        x4_0 = self.conv4_0(self.Maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], 1))

        if self.args.deep_supervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output