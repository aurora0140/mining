from torch import nn
import torch

# CIFAR100网络模型
class Cifarnet(nn.Module):
    def __init__(self):
        super(Cifarnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1,
                      padding=2),  # stride=1,padding=2卷积后尺寸不变
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1,
                      padding=2),  # stride=1,padding=2卷积后尺寸不变
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,
                      padding=2),  # stride=1,padding=2卷积后尺寸不变
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.to(torch.float32)
        print(x.shape)
        return x


if __name__ == '__main__':
    net = Cifarnet()
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    # torch.Size([64, 10])  每一张图片的预测概率为一个10大小的数组
    print(output.shape)