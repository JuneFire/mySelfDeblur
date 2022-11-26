import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# from .non_local_embedded_gaussian import NONLocalBlock2D
# from .non_local_concatenation import NONLocalBlock2D
# from .non_local_gaussian import NONLocalBlock2D
from .non_local_dot_product import NONLocalBlock2D

actfun = nn.LeakyReLU


class Skip_gen(nn.Module):
    def __init__(self, input_channels=8, output_channels=16):
        super(Skip_gen, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = actfun()

    def forward(self, input):
        x = self.conv1(input)
        x = self.act(self.bn(x))
        return x

class Down_block(nn.Module):
    def __init__(self, input_channels=8, output_channels=128, pad='reflect', isNONLocalBlock2D=False):
        super(Down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, padding_mode=pad)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.act1 = actfun()

        self.isNONLocalBlock2D = isNONLocalBlock2D
        if self.isNONLocalBlock2D:
            self.NONLocalBlock2D = NONLocalBlock2D(input_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, padding_mode=pad)  # stride=1,
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.act2 = actfun()


    def forward(self, input):
        x = self.conv1(input)
        x = self.act1(self.bn1(x))
        if self.isNONLocalBlock2D:
            x = self.NONLocalBlock2D(x)
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x


class Up_block(nn.Module):
    def __init__(self, input_channels=128, output_channels=128, pad='reflect'):
        super(Up_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1, padding_mode=pad)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.act = actfun()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 1, padding=0, padding_mode=pad)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.us = nn.Upsample(scale_factor=2.0, mode='bilinear')

    def forward(self, input):
        x = self.conv1(self.bn1(input))
        x = self.act(self.bn2(x))
        x = self.conv2(x)
        x = self.act(self.bn2(x))
        x = self.us(x)

        return x



class My_Skip(nn.Module):
    def __init__(self, input_channels=3, output_channels=3,
                 num_channels_down=[128, 128, 128, 128, 128],
                 num_channels_up=[128, 128, 128, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 pad='reflect'):
        super(My_Skip, self).__init__()

        self.down0 = Down_block(input_channels=input_channels, output_channels=num_channels_down[0], isNONLocalBlock2D=False)
        self.down1 = Down_block(input_channels=num_channels_down[0], output_channels=num_channels_down[1], isNONLocalBlock2D=False)
        self.down2 = Down_block(input_channels=num_channels_down[1], output_channels=num_channels_down[2], isNONLocalBlock2D=True)
        self.down3 = Down_block(input_channels=num_channels_down[2], output_channels=num_channels_down[3], isNONLocalBlock2D=True)
        self.down4 = Down_block(input_channels=num_channels_down[3], output_channels=num_channels_down[4], isNONLocalBlock2D=True)

        self.skip0 = Skip_gen(input_channels, num_channels_skip[0])
        self.skip1 = Skip_gen(num_channels_down[0], num_channels_skip[1])
        self.skip2 = Skip_gen(num_channels_down[1], num_channels_skip[2])
        self.skip3 = Skip_gen(num_channels_down[2], num_channels_skip[3])
        self.skip4 = Skip_gen(num_channels_down[3], num_channels_skip[4])

        self.us = nn.Upsample(scale_factor=2.0, mode='bilinear')   # 对应 deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        self.up4 = Up_block(num_channels_down[4] + num_channels_skip[4], num_channels_up[4])
        self.up3 = Up_block(num_channels_up[4] + num_channels_skip[3], num_channels_up[3])
        self.up2 = Up_block(num_channels_up[3] + num_channels_skip[2], num_channels_up[2])
        self.up1 = Up_block(num_channels_up[2] + num_channels_skip[1], num_channels_up[1])
        # self.up0 = Up_block(num_channels_up[1]+num_channels_skip[0], num_channels_up[0])

        # self.temp = nn.Conv2d(num_channels_up[4], 1024, 1)

        # self.temp = nn.Sequential(
        #     nn.Conv2d(num_channels_up[4], 1024, 1),
        #     actfun()
        # )

        ### use in deep image prior
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(num_channels_up[1] + num_channels_skip[0]),
            nn.Conv2d(num_channels_up[1] + num_channels_skip[0], num_channels_up[0], 3, padding=1, padding_mode=pad),
            nn.BatchNorm2d(num_channels_up[0]),
            actfun(),
            nn.Conv2d(num_channels_up[0], num_channels_up[0], 1),
            nn.BatchNorm2d(num_channels_up[0]),
            actfun(),
            nn.Conv2d(num_channels_up[0], output_channels, 1)
        )
        # self.last_conv = nn.Sequential(  # error absence bn
        #     nn.Conv2d(num_channels_up[1] + num_channels_skip[0], num_channels_up[0], 3, padding=1, padding_mode=pad),
        #     nn.BatchNorm2d(num_channels_up[0]),
        #     actfun(),
        #     nn.Conv2d(num_channels_up[0], num_channels_up[0], 1),
        #     nn.BatchNorm2d(num_channels_up[0]),
        #     actfun(),
        #     nn.Conv2d(num_channels_up[0], 3, 1)  # error : 1
        # )

    def forward(self, input):
        S0 = self.skip0(input)
        # print(S0.shape)
        X0 = self.down0(input)
        # print(X0.shape)
        S1 = self.skip1(X0)
        X1 = self.down1(X0)
        S2 = self.skip2(X1)
        X2 = self.down2(X1)
        S3 = self.skip3(X2)
        X3 = self.down3(X2)
        X4 = self.down4(X3)
        S4 = self.skip4(X3)

        Y4 = self.us(X4)
        Y3 = self.up4(Concat(S4, Y4))
        Y2 = self.up3(Concat(S3, Y3))
        Y1 = self.up2(Concat(S2, Y2))
        Y0 = self.up1(Concat(S1, Y1))
        # Y3 = self.up4(torch.cat((S4, Y4), dim=1))
        # Y2 = self.up3(torch.cat((S3, Y3), dim=1))
        # Y1 = self.up2(torch.cat((S2, Y2), dim=1))
        # Y0 = self.up1(torch.cat((S1, Y1), dim=1))
        # output = self.last_conv(torch.cat((S0, Y0), dim=1))
        output = self.last_conv(Concat(S0, Y0))
        output = torch.sigmoid(output)

        return output

'''
下采样时,增加了padding,导致下采样后的数据要比不加padding的大一些.(这样会导致向上取整).
所以在上采样concat的时候,取二者min值进行拼接,保证了最后的输出大小和输入大小一致.
这里每次都要计算尺寸大小,有点浪费资源,个人希望给图像进行大小限制.
'''

def Concat(input1, input2):
    inputs = [input1, input2]
    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]
    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs:
            diff2 = (inp.size(2) - target_shape2) // 2
            diff3 = (inp.size(3) - target_shape3) // 2
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
    return torch.cat(inputs_, 1)  # dim=0 上下竖向cat, dim=0 左右横向cat

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = My_Skip()
    x = torch.randn(1, 3, 256, 256)
    net = net.to(device)
    x = x.to(device)
    out = net(x)

    print(out.shape)
