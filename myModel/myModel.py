import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(3, 96, 11, 4),  # in_channels, out_channels, kernel_size, stride, padding

            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #由于使用CPU镜像，精简网络，若为GPU镜像可添加该层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()
#         self.cnn = nn.Sequential(
#             # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
#             # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
#             # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
#             nn.Conv2d(3, 96, 7, 2, 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, 0),
#
#             # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
#             # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
#             # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
#             nn.Conv2d(96, 256, 5, 1, 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, 0),
#
#             # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(256, 384, 3, 1, 1),
#             nn.ReLU(inplace=True),
#
#             # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(384, 384, 3, 1, 1),
#             nn.ReLU(inplace=True),
#
#             # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
#             # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
#             nn.Conv2d(384, 256, 3, 1, 1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.fc = nn.Sequential(
#             # 256个feature，每个feature 3*3
#             nn.Linear(256*3*3, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         x = self.cnn(x)
#         # x.size()[0]: batch size
#         x = x.view(x.size()[0], -1)
#         x = self.fc(x)
#
#         return x


# net = AlexNet()
# print(net)