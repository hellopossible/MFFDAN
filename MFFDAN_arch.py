'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''
from thop import profile
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY

class MSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros"):
        super().__init__()
        self.dim_conv7 = self.dim_conv5 = self.dim_conv3 = out_channels // 4
        self.dim_conv9 = out_channels - self.dim_conv3 * 3
        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )
        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.dw3 = torch.nn.Conv2d(
                in_channels=self.dim_conv5,
                out_channels=self.dim_conv5,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=self.dim_conv5,
                bias=bias,
                padding_mode=padding_mode,
        )
        self.dw5 = torch.nn.Conv2d(
            in_channels=self.dim_conv7,
            out_channels=self.dim_conv7,
            kernel_size=kernel_size,
            stride=stride,
            padding=2,
            dilation=2,
            groups=self.dim_conv7,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.dw7 = torch.nn.Conv2d(
            in_channels=self.dim_conv9,
            out_channels=self.dim_conv9,
            kernel_size=kernel_size,
            stride=stride,
            padding=3,
            dilation=3,
            groups=self.dim_conv9,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        x1, x2, x3, x4 = torch.split(fea, [self.dim_conv3, self.dim_conv5, self.dim_conv7, self.dim_conv9], dim=1)
        x2 = self.dw3(x2)
        x3 = self.dw5(x3)
        x4 = self.dw7(x4)
        fea = torch.cat((x1, x2, x3, x4), 1)
        return fea
class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class DSA_1(nn.Module):
    def __init__(self, in_channels):
        super(DSA_1, self).__init__()
        f = in_channels // 4
        self.dc = f // 2
        self.conv1 = nn.Conv2d(in_channels, f, 1)
        self.c0_d = nn.Conv2d(f, self.dc, 1)

        self.c1_r = BSConvU(f, f, 3, 2, 0)
        self.c2_r = BSConvU(f, f, 3, 2, 0)
        self.c3_r = BSConvU(f, f, 3, 2, 0)

        self.c1_d = nn.Conv2d(f, self.dc, 1)
        self.c2_d = nn.Conv2d(f, self.dc, 1)
        self.c3_d = nn.Conv2d(f, self.dc, 1)

        self.conv3_1 = BSConvU(self.dc, self.dc, kernel_size=3)
        self.conv3_2 = BSConvU(self.dc, self.dc, kernel_size=3)
        self.conv3_3 = BSConvU(self.dc, self.dc, kernel_size=3)

        self.c3 = nn.Conv2d(self.dc * 3, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1 = self.conv1(input)
        d_c0 = self.c0_d(c1)

        r_c1 = self.c1_r(c1)

        d_c1 = self.c1_d(r_c1)
        d_c1 = self.GELU(self.conv3_1(d_c1))
        d_c1 = F.interpolate(d_c1, (input.size(2), input.size(3)), mode='bilinear', align_corners=False) + d_c0

        r_c2 = self.c2_r(r_c1)

        d_c2 = self.c2_d(r_c2)
        d_c2 = self.GELU(self.conv3_2(d_c2))
        d_c2 = F.interpolate(d_c2, (input.size(2), input.size(3)), mode='bilinear', align_corners=False) + d_c0

        r_c3 = self.c3_r(r_c2)

        d_c3 = self.c3_d(r_c3)
        d_c3 = self.GELU(self.conv3_3(d_c3))
        d_c3 = F.interpolate(d_c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False) + d_c0

        out = torch.cat([d_c1, d_c2, d_c3], dim=1)
        out = self.c3(out)
        s = self.sigmoid(out)
        x = input * s
        return x

class DCA(nn.Module):
    def __init__(self, channel):
        super(DCA, self).__init__()
        self.dw = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        channel_out = self.dw(x)
        channel_out = self.sigmoid(channel_out)
        x = channel_out * x
        return x
class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ESDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = MSConv(in_channels, self.rc, kernel_size=3)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = MSConv(self.remaining_channels, self.rc, kernel_size=3)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = MSConv(self.remaining_channels, self.rc, kernel_size=3)

        self.c4 = MSConv(self.remaining_channels, self.dc, kernel_size=3)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.dsa = DSA_1(in_channels)
        self.dca = DCA(in_channels)
    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out = self.dca(self.dsa(out))
        return out + input

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

@ARCH_REGISTRY.register()
class MFFDAN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                upsampler='pixelshuffledirect'):
        super(MFFDAN, self).__init__()
        self.fea_conv = BSConvU(num_in_ch * 4, num_feat, kernel_size=3,padding=1)

        self.B1 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B2 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B3 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B4 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B5 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B6 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B7 = ESDB(in_channels=num_feat, out_channels=num_feat)
        self.B8 = ESDB(in_channels=num_feat, out_channels=num_feat)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3,padding=1)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

# net = MFFDAN(num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
#                 upsampler='pixelshuffledirect')  # 定义好的网络模型,实例化
# print(net)
# input = torch.randn(1, 3, 320, 180)  # 1280*720
# flops, params = profile(net, (input,))
# print('flops[G]: ', flops/1e9, 'params[K]: ', params/1e3)
