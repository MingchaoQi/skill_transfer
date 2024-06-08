#在这里面我搭建了一个shufflenet_v2网络

from typing import List, Callable
import torch
from torch import Tensor
import torch.nn as nn

#channel shuffle操作的实现
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    #实现Shufflenet最重要的思想channel shuffle（通道重排）操作
    batch_size, num_channels, height, width = x.size()
    #获取传入进函数的feature map的size
    #在pytorch中获取得到的tensor排列顺序为
    #[batch_size, channels, 特征矩阵高度，特征矩阵宽度]
    channels_per_group = num_channels // groups
    #对分组卷积结果每个组内的通道进行划分

    x = x.view(batch_size, groups, channels_per_group, height, width)
    #通过view()方法重新定义tensor的维度
    #[batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = torch.transpose(x, 1, 2).contiguous()
    #通过transpose()方法对维度1和维度2进行互换，可认为是矩阵的转置

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

#Block的搭建
class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        #传入的参数分别为输入特征矩阵的通道数，输出特征矩阵的通道数以及DW卷积采用的步长
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride
        #判断步长是否等于1或2，因为在shufflenet网络中步长只能为1或2

        assert output_c % 2 == 0
        #判断输出特征矩阵通道数是否为2的整数倍，因为在shufflenet V2网络block中
        #在concat拼接前左右分支通道数相等
        branch_features = output_c // 2
        #当stride为1时，input_channel应该是branch_features的两倍
        #因为当stride=1时，input_channel需要经过channel spilt操作
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:  #步长为2
            self.branch1 = nn.Sequential(
                #branch1代表示意图中左边的分支
                self.depthwise_conv(
                    input_c,
                    input_c,  #因为DW卷积输出和输出特征矩阵通道数相同
                    kernel_s=3,
                    stride=self.stride,
                    padding=1),
                nn.BatchNorm2d(input_c),  #DW卷积后feature map通道数=输入通道数
                nn.Conv2d(input_c,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()
        #stride=1时，对左边的分支没有做任何处理

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                input_c if self.stride > 1 else branch_features,
                #不论是stride=1或2，右边分支结构基本相同，不同的只是DW卷积的步长，在这里判断一下
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_s=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    #静态方法，实现深度可分离卷积，即DWConv
    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c,
                         out_channels=output_c,
                         kernel_size=kernel_s,
                         stride=stride,
                         padding=padding,
                         bias=bias,
                         groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            #当stride=1时，需要对input channel进行Channel split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
            #实现concat拼接功能
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

#神经网络完整层结构的搭建
class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],#代表每层结构的重复次数
                 stages_out_channels: List[int],#代表每层结构的输出通道数
                 num_classes: int = 7,#代表进行分类的数目，我这里对七种飞机进行分类
                 inverted_residual: Callable[...,
                                             nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        #检查输入的变量个数是否符合网络需要的参数个数
        if len(stages_repeats) != 3:
            raise ValueError(
                "expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError(
                "expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # 输入的数据集图片均为RGB图片，RGB格式图片有3个channels
        #输出的通道数为人工输入变量的第一个，即_stage_out_channels[0]
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels,
                      output_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))
        input_channels = output_channels
        #表示下一层的输入通道数等于Conv1层的输出通道数。

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
        #同时遍历stage_names, stages_repeats, self._stage_out_channels
            seq = [inverted_residual(input_channels, output_channels, 2)]
        #对于每个stage层结构首先使用的都是stride=2的Block
        #每个stage进行完上面步骤后都会重复使用三个stride=1的Block，故使用循环
            for i in range(repeats - 1):
                seq.append(
                    inverted_residual(output_channels, output_channels, 1))
                #在搭建Block时已经说明当stride=1，input channels=output channels
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            #下一个stage的输入通道数等于上一个stage的输出通道数

        output_channels = self._stage_out_channels[-1]
        #输入通道数为需要输入列表的最后一个参数
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels,
                      output_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False), nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(output_channels, num_classes)
        #通过nn.Linear()方法实现全连接层
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  #全局池化
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=7):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=7):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=7):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=7):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model
